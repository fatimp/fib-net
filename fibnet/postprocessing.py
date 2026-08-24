from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .probability_io import load_probability_map, save_probability_map

POSTPROCESSING_OPERATION_ORDER = [
    "raw_probability_maps",
    "inter_slice_probability_smoothing",
    "fixed_probability_threshold",
    "adjacent_slice_support_filtering_if_enabled",
    "small_2d_connected_component_removal",
    "final_binary_masks",
]
COMPONENT_CONNECTIVITY = 4


def component_area_um2_to_pixels(
    min_component_area_um2: float,
    pixel_size_nm: float | None,
) -> int:
    if pixel_size_nm is None:
        raise ValueError(
            "Cannot convert physical component area because pixel-size metadata are missing."
        )
    if min_component_area_um2 < 0:
        raise ValueError("min-component-area-um2 must be non-negative.")
    pixel_size_um = pixel_size_nm / 1000.0
    pixel_area_um2 = pixel_size_um * pixel_size_um
    return max(1, int(np.ceil(min_component_area_um2 / pixel_area_um2)))


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def collect_probability_paths(prob_dir: Path) -> list[Path]:
    paths = sorted(prob_dir.glob("*_prob.npy"), key=natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No *_prob.npy files found in {prob_dir}.")
    return paths


def load_probability_stack(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    stems = [path.name.removesuffix("_prob.npy") for path in paths]
    arrays = [load_probability_map(path) for path in paths]
    return stems, np.stack(arrays, axis=0)


def temporal_average(
    probabilities: np.ndarray, radius: int, blend: float
) -> np.ndarray:
    if radius <= 0 or blend <= 0.0:
        return probabilities
    smoothed = np.empty_like(probabilities)
    for index in range(probabilities.shape[0]):
        start = max(0, index - radius)
        end = min(probabilities.shape[0], index + radius + 1)
        local_mean = probabilities[start:end].mean(axis=0)
        smoothed[index] = ((1.0 - blend) * probabilities[index]) + (blend * local_mean)
    return smoothed


def temporal_support(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.uint8)
    support = np.zeros(mask.shape, dtype=np.uint8)
    for index in range(mask.shape[0]):
        start = max(0, index - radius)
        end = min(mask.shape[0], index + radius + 1)
        support[index] = mask[start:end].sum(axis=0)
    return support


def component_positions(
    mask: np.ndarray, start_y: int, start_x: int, visited: np.ndarray
) -> list[tuple[int, int]]:
    height, width = mask.shape
    stack = [(start_y, start_x)]
    visited[start_y, start_x] = True
    positions: list[tuple[int, int]] = []
    while stack:
        y, x = stack.pop()
        positions.append((y, x))
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                continue
            if visited[next_y, next_x] or not mask[next_y, next_x]:
                continue
            visited[next_y, next_x] = True
            stack.append((next_y, next_x))
    return positions


def remove_small_objects_2d(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask
    cleaned = mask.copy()
    for z in range(mask.shape[0]):
        visited = np.zeros(mask[z].shape, dtype=bool)
        ys, xs = np.where(mask[z])
        for y, x in zip(ys, xs, strict=False):
            if visited[y, x]:
                continue
            positions = component_positions(mask[z], int(y), int(x), visited)
            if len(positions) < min_size:
                for pos_y, pos_x in positions:
                    cleaned[z, pos_y, pos_x] = False
    return cleaned


def hysteresis_threshold_2d(
    probabilities: np.ndarray, high_threshold: float, low_threshold: float | None
) -> np.ndarray:
    if low_threshold is None:
        return probabilities >= high_threshold
    if low_threshold > high_threshold:
        raise ValueError("hysteresis-low-threshold must be <= threshold.")

    seeds = probabilities >= high_threshold
    candidates = probabilities >= low_threshold
    output = np.zeros(probabilities.shape, dtype=bool)
    for z in range(probabilities.shape[0]):
        visited = np.zeros(probabilities[z].shape, dtype=bool)
        seed_y, seed_x = np.where(seeds[z])
        for y, x in zip(seed_y, seed_x, strict=False):
            if visited[y, x]:
                continue
            positions = component_positions(candidates[z], int(y), int(x), visited)
            for pos_y, pos_x in positions:
                output[z, pos_y, pos_x] = True
    return output


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output_path)


def save_overlay(source_path: Path, mask: np.ndarray, output_path: Path) -> None:
    image = Image.open(source_path).convert("RGB")
    image_array = np.asarray(image, dtype=np.float32)
    if mask.shape != image_array.shape[:2]:
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_image = mask_image.resize(
            (image_array.shape[1], image_array.shape[0]), Image.Resampling.NEAREST
        )
        mask = np.asarray(mask_image, dtype=np.uint8) > 0
    overlay_color = np.array([255.0, 48.0, 48.0], dtype=np.float32)
    image_array[mask] = 0.55 * image_array[mask] + 0.45 * overlay_color
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode="RGB").save(
        output_path
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply conservative cross-section sequence post-processing to probability maps."
    )
    parser.add_argument("--prob-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=Path("target"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.47)
    parser.add_argument("--hysteresis-low-threshold", type=float, default=None)
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument("--smooth-blend", type=float, default=0.25)
    parser.add_argument("--support-radius", type=int, default=1)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--min-object-size-2d", type=int, default=0)
    parser.add_argument("--min-component-area-um2", type=float, default=None)
    parser.add_argument("--pixel-size-nm", type=float, default=None)
    parser.add_argument("--save-smoothed-probability", action="store_true")
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--overlay-stems", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_probability_paths(args.prob_dir)
    stems, probabilities = load_probability_stack(paths)
    smoothed = temporal_average(
        probabilities, radius=args.smooth_radius, blend=args.smooth_blend
    )
    mask = hysteresis_threshold_2d(
        smoothed,
        high_threshold=args.threshold,
        low_threshold=args.hysteresis_low_threshold,
    )
    if args.min_support > 1:
        support = temporal_support(mask, radius=args.support_radius)
        mask = mask & (support >= args.min_support)
    min_component_size_px = args.min_object_size_2d
    if args.min_component_area_um2 is not None:
        min_component_size_px = component_area_um2_to_pixels(
            args.min_component_area_um2,
            args.pixel_size_nm,
        )
    mask = remove_small_objects_2d(mask, min_size=min_component_size_px)

    overlay_stems = set(args.overlay_stems or [])
    for index, stem in enumerate(stems):
        output_stem = args.output_dir / stem
        save_mask(mask[index], output_stem.with_name(f"{stem}_pred.png"))
        if args.save_smoothed_probability:
            save_probability_map(
                output_stem.with_name(f"{stem}_prob.npy"), smoothed[index]
            )
        if args.save_overlay and (not overlay_stems or stem in overlay_stems):
            source_path = args.target_dir / f"{stem}.tiff"
            save_overlay(
                source_path, mask[index], output_stem.with_name(f"{stem}_overlay.png")
            )

    summary = {
        "num_slices": len(stems),
        "operation_order": POSTPROCESSING_OPERATION_ORDER,
        "threshold": args.threshold,
        "hysteresis_low_threshold": args.hysteresis_low_threshold,
        "inter_slice_smoothing_radius": args.smooth_radius,
        "inter_slice_smoothing_blend": args.smooth_blend,
        "boundary_rule": "first_and_last_cross_sections_use_available_neighbours_only",
        "support_radius": args.support_radius,
        "min_support": args.min_support,
        "adjacent_slice_support_filtering_enabled": args.min_support > 1,
        "component_connectivity": COMPONENT_CONNECTIVITY,
        "requested_min_component_area_um2": args.min_component_area_um2,
        "historical_min_component_size_px": args.min_object_size_2d,
        "resolved_min_component_size_px": min_component_size_px,
        "pixel_size_nm": args.pixel_size_nm,
        "morphological_closing_enabled": False,
        "manual_correction_after_prediction": False,
        "mean_mask_fraction": float(mask.mean()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "postprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
