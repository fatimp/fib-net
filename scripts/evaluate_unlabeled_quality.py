from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from fibnet.probability_io import load_probability_map


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def collect_stems(prob_dir: Path, pred_dir: Path) -> list[str]:
    prob_stems = {
        path.name.removesuffix("_prob.npy") for path in prob_dir.glob("*_prob.npy")
    }
    pred_stems = {
        path.name.removesuffix("_pred.png") for path in pred_dir.glob("*_pred.png")
    }
    stems = sorted(
        prob_stems & pred_stems,
        key=lambda stem: [
            int(part) if part.isdigit() else part for part in re.split(r"(\d+)", stem)
        ],
    )
    if not stems:
        raise FileNotFoundError(
            f"No matching *_prob.npy and *_pred.png files found in {prob_dir} / {pred_dir}."
        )
    return stems


def load_probability(prob_dir: Path, stem: str) -> np.ndarray:
    return load_probability_map(prob_dir / f"{stem}_prob.npy")


def load_mask(pred_dir: Path, stem: str) -> np.ndarray:
    return np.asarray(Image.open(pred_dir / f"{stem}_pred.png").convert("L")) > 0


def component_sizes(mask: np.ndarray) -> list[int]:
    visited = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []
    ys, xs = np.where(mask)
    height, width = mask.shape
    for y, x in zip(ys, xs, strict=False):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
        sizes.append(size)
    return sizes


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b_image = Image.fromarray(b.astype(np.uint8) * 255, mode="L")
        b_image = b_image.resize((a.shape[1], a.shape[0]), Image.Resampling.NEAREST)
        b = np.asarray(b_image, dtype=np.uint8) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def entropy(probability: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(probability, eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def score_row(row: dict[str, float]) -> float:
    penalty = 0.0
    penalty += min(row["uncertain_fraction"] / 0.08, 1.5) * 35.0
    penalty += min(row["entropy"] / 0.45, 1.5) * 25.0
    penalty += min(max(0.0, 0.55 - row["neighbor_iou"]) / 0.55, 1.0) * 25.0
    penalty += min(row["small_component_fraction"] / 0.08, 1.0) * 10.0
    penalty += min(abs(row["area_zscore"]) / 3.0, 1.0) * 5.0
    return float(max(0.0, 100.0 - penalty))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate segmentation quality without ground-truth masks."
    )
    parser.add_argument("--prob-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--stems", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.47)
    parser.add_argument("--uncertainty-margin", type=float, default=0.08)
    parser.add_argument("--small-component-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stems = collect_stems(args.prob_dir, args.pred_dir)
    stem_to_index = {stem: index for index, stem in enumerate(stems)}
    selected_stems = args.stems or stems

    probabilities = {stem: load_probability(args.prob_dir, stem) for stem in stems}
    masks = {stem: load_mask(args.pred_dir, stem) for stem in stems}
    areas = np.asarray([masks[stem].mean() for stem in stems], dtype=np.float32)
    area_mean = float(areas.mean())
    area_std = float(areas.std() + 1e-6)

    rows = []
    for stem in selected_stems:
        if stem not in stem_to_index:
            raise FileNotFoundError(
                f"No prediction/probability found for stem {stem!r}."
            )
        probability = probabilities[stem]
        mask = masks[stem]
        sizes = component_sizes(mask)
        mask_pixels = int(mask.sum())
        small_pixels = sum(size for size in sizes if size < args.small_component_size)
        index = stem_to_index[stem]
        neighbor_ious = []
        if index > 0:
            neighbor_ious.append(mask_iou(mask, masks[stems[index - 1]]))
        if index < len(stems) - 1:
            neighbor_ious.append(mask_iou(mask, masks[stems[index + 1]]))

        uncertain = np.abs(probability - args.threshold) <= args.uncertainty_margin
        row = {
            "stem": stem,
            "quality_score": 0.0,
            "mask_fraction": float(mask.mean()),
            "area_zscore": float((mask.mean() - area_mean) / area_std),
            "mean_confidence": float(np.maximum(probability, 1.0 - probability).mean()),
            "uncertain_fraction": float(uncertain.mean()),
            "entropy": float(entropy(probability).mean()),
            "neighbor_iou": float(np.mean(neighbor_ious)) if neighbor_ious else 1.0,
            "component_count": float(len(sizes)),
            "largest_component_fraction": float(max(sizes) / mask_pixels)
            if mask_pixels
            else 0.0,
            "small_component_fraction": float(small_pixels / mask_pixels)
            if mask_pixels
            else 0.0,
        }
        row["quality_score"] = score_row(row)
        rows.append(row)

    rows.sort(key=lambda row: row["stem"])
    for row in rows:
        print(
            "stem={stem} score={quality_score:.1f} area={mask_fraction:.4f} "
            "conf={mean_confidence:.4f} uncertain={uncertain_fraction:.4f} "
            "entropy={entropy:.4f} neighbor_iou={neighbor_iou:.4f} "
            "components={component_count:.0f} small={small_component_fraction:.4f}".format(
                **row
            )
        )

    summary = {
        "rows": rows,
        "notes": {
            "quality_score": "Proxy score from 0 to 100; higher is more stable, not Dice/IoU.",
            "uncertain_fraction": "Fraction of pixels close to the threshold.",
            "neighbor_iou": "Mask overlap with adjacent stack slices.",
            "small_component_fraction": "Fraction of predicted pore pixels in small isolated components.",
        },
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
