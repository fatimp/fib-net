from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from .image_features import (
    feature_channels,
    image_stack_to_feature_tensor,
    image_to_feature_tensor,
)
from .model import build_model
from .probability_io import save_probability_map

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained pore segmentation model."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Optional image stems to process from an input directory.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=("grayscale", "relief", "stack_relief"),
        default=None,
    )
    parser.add_argument(
        "--model-arch", type=str, choices=("unet", "resunet"), default=None
    )
    parser.add_argument("--mode", choices=("tile", "whole"), default="tile")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hysteresis-low-threshold", type=float, default=None)
    parser.add_argument(
        "--stack-aggregation", choices=("none", "mean", "max"), default="none"
    )
    parser.add_argument("--stack-radius", type=int, default=1)
    parser.add_argument("--stack-blend", type=float, default=0.35)
    parser.add_argument("--min-object-size", type=int, default=0)
    parser.add_argument("--fill-hole-size", type=int, default=0)
    parser.add_argument("--bright-veto-threshold", type=int, default=None)
    parser.add_argument("--bright-veto-radius", type=int, default=0)
    parser.add_argument("--save-probability", action="store_true")
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def collect_image_paths(directory: Path) -> list[Path]:
    input_paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    input_paths.sort(key=natural_sort_key)
    return input_paths


def build_context_lookup(context_dir: Path | None) -> dict[Path, tuple[Path, Path]]:
    if context_dir is None:
        return {}
    context_paths = collect_image_paths(context_dir)
    lookup: dict[Path, tuple[Path, Path]] = {}
    for index, path in enumerate(context_paths):
        previous_path = context_paths[index - 1] if index > 0 else path
        next_path = context_paths[index + 1] if index < len(context_paths) - 1 else path
        lookup[path.resolve()] = (previous_path, next_path)
    return lookup


def image_to_tensor(
    image: Image.Image,
    feature_mode: str,
    previous_image: Image.Image | None = None,
    next_image: Image.Image | None = None,
) -> torch.Tensor:
    if feature_mode == "stack_relief":
        previous_image = previous_image or image
        next_image = next_image or image
        return image_stack_to_feature_tensor(
            previous_image, image, next_image
        ).unsqueeze(0)
    return image_to_feature_tensor(image, feature_mode=feature_mode).unsqueeze(0)


def load_resized_image(
    path: Path,
    image_size: int,
    feature_mode: str,
    previous_path: Path | None = None,
    next_path: Path | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("L")
    original_size = image.size
    if image_size > 0:
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    previous_image = None
    next_image = None
    if feature_mode == "stack_relief":
        previous_image = Image.open(previous_path or path).convert("L")
        next_image = Image.open(next_path or path).convert("L")
        if image_size > 0:
            previous_image = previous_image.resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            next_image = next_image.resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
    return image_to_tensor(
        image, feature_mode, previous_image, next_image
    ), original_size


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    mask = mask.astype(np.uint8) * 255
    image = Image.fromarray(mask, mode="L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_probability(probability: np.ndarray, output_path: Path) -> None:
    save_probability_map(output_path, probability)


def save_probability_preview(probability: np.ndarray, output_path: Path) -> None:
    probability_image = (np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(probability_image, mode="L").save(output_path)


def save_overlay(
    source_path: Path, probability: np.ndarray, output_path: Path, threshold: float
) -> None:
    image = Image.open(source_path).convert("RGB")
    image_array = np.asarray(image, dtype=np.float32)
    mask = probability >= threshold
    overlay_color = np.array([255.0, 48.0, 48.0], dtype=np.float32)
    image_array[mask] = 0.55 * image_array[mask] + 0.45 * overlay_color
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image_array, 0.0, 255.0).astype(np.uint8), mode="RGB").save(
        output_path
    )


def predict_whole(
    model: torch.nn.Module,
    path: Path,
    image_size: int,
    feature_mode: str,
    device: str,
    previous_path: Path | None = None,
    next_path: Path | None = None,
) -> np.ndarray:
    tensor, original_size = load_resized_image(
        path, image_size, feature_mode, previous_path, next_path
    )
    logits = model(tensor.to(device))
    probability = torch.sigmoid(logits).squeeze().cpu().numpy()
    probability_image = Image.fromarray(
        (probability * 255.0).astype(np.uint8), mode="L"
    )
    probability_image = probability_image.resize(
        original_size, Image.Resampling.BILINEAR
    )
    return np.asarray(probability_image, dtype=np.float32) / 255.0


def _axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _tile_weight(tile_size: int) -> np.ndarray:
    axis = np.arange(tile_size, dtype=np.float32)
    distance_to_edge = np.minimum(axis + 1.0, tile_size - axis)
    weight_1d = distance_to_edge / distance_to_edge.max()
    return np.outer(weight_1d, weight_1d).astype(np.float32)


def accumulate_weighted_tile(
    probability_sum: np.ndarray,
    weight_sum: np.ndarray,
    probability: np.ndarray,
    weight: np.ndarray,
    y: int,
    x: int,
) -> None:
    """Accumulate one tile using the same blending weights in numerator and denominator."""
    if probability.shape != weight.shape:
        raise ValueError("probability and weight must have identical shapes.")
    if probability_sum.shape != weight_sum.shape:
        raise ValueError("probability_sum and weight_sum must have identical shapes.")

    height, width = probability.shape
    probability_sum[y : y + height, x : x + width] += probability * weight
    weight_sum[y : y + height, x : x + width] += weight


def normalize_weighted_probability(
    probability_sum: np.ndarray, weight_sum: np.ndarray
) -> np.ndarray:
    """Return the weighted mean wherever at least one tile contributed."""
    if probability_sum.shape != weight_sum.shape:
        raise ValueError("probability_sum and weight_sum must have identical shapes.")
    return probability_sum / np.maximum(weight_sum, np.finfo(np.float32).eps)


def predict_tiled(
    model: torch.nn.Module,
    path: Path,
    tile_size: int,
    overlap: int,
    feature_mode: str,
    device: str,
    previous_path: Path | None = None,
    next_path: Path | None = None,
) -> np.ndarray:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in the range [0, tile_size).")

    image = Image.open(path).convert("L")
    previous_image = (
        Image.open(previous_path or path).convert("L")
        if feature_mode == "stack_relief"
        else None
    )
    next_image = (
        Image.open(next_path or path).convert("L")
        if feature_mode == "stack_relief"
        else None
    )
    width, height = image.size
    stride = tile_size - overlap
    xs = _axis_positions(width, tile_size, stride)
    ys = _axis_positions(height, tile_size, stride)

    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    tile_weight = _tile_weight(tile_size)

    for y in ys:
        for x in xs:
            crop_width = min(tile_size, width - x)
            crop_height = min(tile_size, height - y)
            tile = image.crop((x, y, x + crop_width, y + crop_height))
            previous_tile = (
                previous_image.crop((x, y, x + crop_width, y + crop_height))
                if previous_image
                else None
            )
            next_tile = (
                next_image.crop((x, y, x + crop_width, y + crop_height))
                if next_image
                else None
            )
            if tile.size != (tile_size, tile_size):
                padded = Image.new("L", (tile_size, tile_size), color=0)
                padded.paste(tile, (0, 0))
                tile = padded
                if previous_tile is not None:
                    padded_previous = Image.new("L", (tile_size, tile_size), color=0)
                    padded_previous.paste(previous_tile, (0, 0))
                    previous_tile = padded_previous
                if next_tile is not None:
                    padded_next = Image.new("L", (tile_size, tile_size), color=0)
                    padded_next.paste(next_tile, (0, 0))
                    next_tile = padded_next

            tensor = image_to_tensor(tile, feature_mode, previous_tile, next_tile)
            logits = model(tensor.to(device))
            probability = torch.sigmoid(logits).squeeze().cpu().numpy()
            probability = probability[:crop_height, :crop_width]
            weight = tile_weight[:crop_height, :crop_width]

            accumulate_weighted_tile(
                probability_sum, weight_sum, probability, weight, y, x
            )

    return normalize_weighted_probability(probability_sum, weight_sum)


def _component_positions(
    mask: np.ndarray, start_y: int, start_x: int, visited: np.ndarray
) -> list[tuple[int, int]]:
    height, width = mask.shape
    target_value = bool(mask[start_y, start_x])
    stack = [(start_y, start_x)]
    visited[start_y, start_x] = True
    positions: list[tuple[int, int]] = []

    while stack:
        y, x = stack.pop()
        positions.append((y, x))
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                continue
            if visited[next_y, next_x] or bool(mask[next_y, next_x]) != target_value:
                continue
            visited[next_y, next_x] = True
            stack.append((next_y, next_x))

    return positions


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask

    cleaned = mask.copy()
    visited = np.zeros(mask.shape, dtype=bool)
    foreground_y, foreground_x = np.where(mask)
    for y, x in zip(foreground_y, foreground_x, strict=False):
        if visited[y, x]:
            continue
        positions = _component_positions(mask, int(y), int(x), visited)
        if len(positions) < min_size:
            for pos_y, pos_x in positions:
                cleaned[pos_y, pos_x] = False
    return cleaned


def fill_small_holes(mask: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        return mask

    filled = mask.copy()
    background = ~mask
    visited = np.zeros(mask.shape, dtype=bool)
    background_y, background_x = np.where(background)
    height, width = mask.shape
    for y, x in zip(background_y, background_x, strict=False):
        if visited[y, x]:
            continue
        positions = _component_positions(background, int(y), int(x), visited)
        touches_border = any(
            pos_y == 0 or pos_y == height - 1 or pos_x == 0 or pos_x == width - 1
            for pos_y, pos_x in positions
        )
        if not touches_border and len(positions) <= max_size:
            for pos_y, pos_x in positions:
                filled[pos_y, pos_x] = True
    return filled


def apply_hysteresis_threshold(
    probability: np.ndarray, high_threshold: float, low_threshold: float | None
) -> np.ndarray:
    if low_threshold is None:
        return probability >= high_threshold
    if low_threshold > high_threshold:
        raise ValueError(
            "hysteresis-low-threshold must be less than or equal to threshold."
        )

    seeds = probability >= high_threshold
    candidates = probability >= low_threshold
    if not seeds.any():
        return seeds

    mask = np.zeros(probability.shape, dtype=bool)
    visited = np.zeros(probability.shape, dtype=bool)
    seed_y, seed_x = np.where(seeds)
    for y, x in zip(seed_y, seed_x, strict=False):
        if visited[y, x]:
            continue
        positions = _component_positions(candidates, int(y), int(x), visited)
        for pos_y, pos_x in positions:
            mask[pos_y, pos_x] = True
    return mask


def build_mask(
    probability: np.ndarray,
    threshold: float,
    min_object_size: int,
    fill_hole_size: int,
    hysteresis_low_threshold: float | None = None,
) -> np.ndarray:
    mask = apply_hysteresis_threshold(probability, threshold, hysteresis_low_threshold)
    mask = remove_small_objects(mask, min_object_size)
    mask = fill_small_holes(mask, fill_hole_size)
    return mask


def apply_bright_veto(
    probability: np.ndarray,
    source_path: Path,
    threshold: int | None,
    radius: int,
) -> np.ndarray:
    if threshold is None:
        return probability
    if not 0 <= threshold <= 255:
        raise ValueError("bright-veto-threshold must be in the range [0, 255].")
    if radius < 0:
        raise ValueError("bright-veto-radius must be non-negative.")

    image = Image.open(source_path).convert("L")
    bright_mask = (np.asarray(image, dtype=np.uint8) > threshold).astype(np.uint8) * 255
    if radius > 0:
        filter_size = (radius * 2) + 1
        bright_mask = np.asarray(
            Image.fromarray(bright_mask, mode="L").filter(
                ImageFilter.MaxFilter(filter_size)
            ),
            dtype=np.uint8,
        )
    veto = bright_mask > 0
    if veto.shape != probability.shape:
        veto_image = Image.fromarray((veto.astype(np.uint8) * 255), mode="L")
        veto_image = veto_image.resize(
            (probability.shape[1], probability.shape[0]), Image.Resampling.NEAREST
        )
        veto = np.asarray(veto_image, dtype=np.uint8) > 0

    filtered = probability.copy()
    filtered[veto] = 0.0
    return filtered


def aggregate_stack_probabilities(
    probabilities: list[np.ndarray],
    aggregation: str,
    radius: int,
    blend: float,
) -> list[np.ndarray]:
    if aggregation == "none":
        return probabilities
    if radius < 1:
        raise ValueError(
            "stack-radius must be at least 1 when stack aggregation is enabled."
        )
    if not 0.0 <= blend <= 1.0:
        raise ValueError("stack-blend must be in the range [0.0, 1.0].")

    aggregated: list[np.ndarray] = []
    for index, probability in enumerate(probabilities):
        start = max(0, index - radius)
        end = min(len(probabilities), index + radius + 1)
        window = np.stack(probabilities[start:end], axis=0)
        if aggregation == "mean":
            stack_probability = window.mean(axis=0)
        elif aggregation == "max":
            stack_probability = window.max(axis=0)
        else:
            raise ValueError(f"Unsupported stack aggregation: {aggregation}")
        aggregated.append(((1.0 - blend) * probability) + (blend * stack_probability))
    return aggregated


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    checkpoint_args = checkpoint.get("args", {})
    image_size = (
        args.image_size
        if args.image_size is not None
        else checkpoint_args.get("image_size", 256)
    )
    feature_mode = args.feature_mode or checkpoint_args.get("feature_mode", "grayscale")
    model_arch = args.model_arch or checkpoint_args.get("model_arch", "unet")
    tile_size = args.tile_size or image_size
    model = build_model(model_arch, in_channels=feature_channels(feature_mode)).to(
        args.device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.input.is_file():
        input_paths = [args.input]
    else:
        input_paths = collect_image_paths(args.input)
        if args.stems:
            selected_stems = set(args.stems)
            input_paths = [path for path in input_paths if path.stem in selected_stems]
            missing_stems = sorted(selected_stems - {path.stem for path in input_paths})
            if missing_stems:
                raise FileNotFoundError(
                    f"No input images found for stems: {', '.join(missing_stems)}"
                )

    context_lookup = build_context_lookup(args.context_dir)

    with torch.no_grad():
        probabilities: list[np.ndarray] = []
        output_stems: list[Path] = []
        for index, path in enumerate(input_paths):
            if args.input.is_file():
                output_stem = args.output_dir / path.stem
            else:
                relative_path = path.relative_to(args.input)
                output_stem = args.output_dir / relative_path.parent / path.stem
            output_stems.append(output_stem)

            previous_path = (
                input_paths[index - 1]
                if index > 0 and not args.input.is_file()
                else path
            )
            next_path = (
                input_paths[index + 1]
                if index < len(input_paths) - 1 and not args.input.is_file()
                else path
            )
            previous_path, next_path = context_lookup.get(
                path.resolve(), (previous_path, next_path)
            )
            if args.mode == "whole":
                probability = predict_whole(
                    model,
                    path,
                    image_size,
                    feature_mode,
                    args.device,
                    previous_path=previous_path,
                    next_path=next_path,
                )
            else:
                probability = predict_tiled(
                    model,
                    path,
                    tile_size,
                    args.overlap,
                    feature_mode,
                    args.device,
                    previous_path=previous_path,
                    next_path=next_path,
                )
            probabilities.append(probability)

        probabilities = aggregate_stack_probabilities(
            probabilities,
            aggregation=args.stack_aggregation,
            radius=args.stack_radius,
            blend=args.stack_blend,
        )

        for path, output_stem, probability in zip(
            input_paths, output_stems, probabilities, strict=False
        ):
            probability = apply_bright_veto(
                probability,
                path,
                threshold=args.bright_veto_threshold,
                radius=args.bright_veto_radius,
            )
            mask = build_mask(
                probability,
                threshold=args.threshold,
                min_object_size=args.min_object_size,
                fill_hole_size=args.fill_hole_size,
                hysteresis_low_threshold=args.hysteresis_low_threshold,
            )
            mask_path = output_stem.with_name(f"{output_stem.name}_pred.png")
            save_mask(mask, mask_path)
            print(f"Saved prediction to {mask_path}")

            if args.save_probability:
                probability_path = output_stem.with_name(f"{output_stem.name}_prob.npy")
                save_probability(probability, probability_path)
                print(f"Saved probability map to {probability_path}")
                preview_path = output_stem.with_name(
                    f"{output_stem.name}_prob_preview.png"
                )
                save_probability_preview(probability, preview_path)
                print(f"Saved probability preview to {preview_path}")

            if args.save_overlay:
                overlay_path = output_stem.with_name(f"{output_stem.name}_overlay.png")
                save_overlay(path, mask.astype(np.float32), overlay_path, threshold=0.5)
                print(f"Saved overlay to {overlay_path}")


if __name__ == "__main__":
    main()
