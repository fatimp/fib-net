from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def natural_key(value: str) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", value.lower())
    ]


def collect_manual_stems(manual_dir: Path) -> set[str]:
    return {
        path.stem
        for path in manual_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def collect_prediction_stems(pred_dir: Path) -> set[str]:
    stems = set()
    for path in pred_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.name.endswith("_pred.png"):
            stems.add(path.name.removesuffix("_pred.png"))
        else:
            stems.add(path.stem)
    return stems


def find_manual_path(manual_dir: Path, stem: str) -> Path:
    matches = sorted(
        path
        for path in manual_dir.glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not matches:
        raise FileNotFoundError(
            f"No manual mask found for stem {stem!r} in {manual_dir}."
        )
    return matches[0]


def find_prediction_path(pred_dir: Path, stem: str) -> Path:
    pred_path = pred_dir / f"{stem}_pred.png"
    if pred_path.exists():
        return pred_path
    matches = sorted(
        path
        for path in pred_dir.glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not matches:
        raise FileNotFoundError(
            f"No predicted mask found for stem {stem!r} in {pred_dir}."
        )
    return matches[0]


def load_mask(path: Path, pore_value: str) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"))
    if pore_value == "zero":
        return array == 0
    if pore_value == "nonzero":
        return array > 0
    raise ValueError("pore_value must be 'zero' or 'nonzero'.")


def resize_to(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def neighbors(y: int, x: int, height: int, width: int, connectivity: int):
    candidates = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
    if connectivity == 8:
        candidates.extend(
            [
                (y - 1, x - 1),
                (y - 1, x + 1),
                (y + 1, x - 1),
                (y + 1, x + 1),
            ]
        )
    for ny, nx in candidates:
        if 0 <= ny < height and 0 <= nx < width:
            yield ny, nx


def component_sizes(
    mask: np.ndarray, connectivity: int
) -> tuple[list[int], list[bool]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []
    touches_border: list[bool] = []
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs, strict=False):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        size = 0
        border = False
        while stack:
            cy, cx = stack.pop()
            size += 1
            border = border or cy == 0 or cx == 0 or cy == height - 1 or cx == width - 1
            for ny, nx in neighbors(cy, cx, height, width, connectivity):
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
        sizes.append(size)
        touches_border.append(border)
    return sizes, touches_border


def perimeter_4(mask: np.ndarray) -> int:
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    perimeter = 0
    perimeter += np.logical_and(center, ~padded[:-2, 1:-1]).sum()
    perimeter += np.logical_and(center, ~padded[2:, 1:-1]).sum()
    perimeter += np.logical_and(center, ~padded[1:-1, :-2]).sum()
    perimeter += np.logical_and(center, ~padded[1:-1, 2:]).sum()
    return int(perimeter)


def scalar_metrics(mask: np.ndarray, connectivity: int) -> dict[str, float]:
    pore_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    foreground_sizes, _ = component_sizes(mask, connectivity)
    background_sizes, background_touches_border = component_sizes(~mask, connectivity)
    hole_sizes = [
        size
        for size, touches_border in zip(
            background_sizes, background_touches_border, strict=False
        )
        if not touches_border
    ]
    component_count = len(foreground_sizes)
    hole_count = len(hole_sizes)
    perimeter = perimeter_4(mask)
    largest_component = max(foreground_sizes, default=0)
    mean_component_size = float(np.mean(foreground_sizes)) if foreground_sizes else 0.0
    median_component_size = float(median(foreground_sizes)) if foreground_sizes else 0.0
    equivalent_diameter = 2.0 * math.sqrt(pore_pixels / math.pi) if pore_pixels else 0.0
    return {
        "porosity": pore_pixels / total_pixels if total_pixels else 0.0,
        "pore_pixels": float(pore_pixels),
        "component_count": float(component_count),
        "hole_count": float(hole_count),
        "euler_number": float(component_count - hole_count),
        "largest_component_fraction": largest_component / pore_pixels
        if pore_pixels
        else 0.0,
        "mean_component_size": mean_component_size,
        "median_component_size": median_component_size,
        "perimeter": float(perimeter),
        "specific_perimeter": perimeter / total_pixels if total_pixels else 0.0,
        "perimeter_per_pore_pixel": perimeter / pore_pixels if pore_pixels else 0.0,
        "equivalent_diameter": equivalent_diameter,
    }


def overlap_metrics(manual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    intersection = np.logical_and(manual, prediction).sum()
    union = np.logical_or(manual, prediction).sum()
    manual_sum = manual.sum()
    pred_sum = prediction.sum()
    fp = np.logical_and(prediction, ~manual).sum()
    fn = np.logical_and(~prediction, manual).sum()
    tn = np.logical_and(~prediction, ~manual).sum()
    return {
        "dice": (2.0 * intersection / (manual_sum + pred_sum))
        if manual_sum + pred_sum
        else 1.0,
        "iou": (intersection / union) if union else 1.0,
        "precision": (intersection / pred_sum) if pred_sum else 1.0,
        "recall": (intersection / manual_sum) if manual_sum else 1.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / manual_sum if manual_sum else 0.0,
    }


def compare_scalar_metrics(
    manual: dict[str, float], prediction: dict[str, float]
) -> dict[str, float]:
    compared = {}
    for key, manual_value in manual.items():
        pred_value = prediction[key]
        compared[f"manual_{key}"] = manual_value
        compared[f"pred_{key}"] = pred_value
        compared[f"delta_{key}"] = pred_value - manual_value
        compared[f"relative_delta_{key}"] = (
            (pred_value - manual_value) / manual_value
            if abs(manual_value) > 1e-12
            else 0.0
        )
    return compared


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare manual masks with predicted masks using overlap and morphology metrics."
    )
    parser.add_argument("--manual-dir", type=Path, default=Path("ideal"))
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/segmentation_metric_comparison.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/segmentation_metric_comparison.csv"),
    )
    parser.add_argument("--stems", nargs="*", default=None)
    parser.add_argument(
        "--manual-pore-value", choices=("zero", "nonzero"), default="zero"
    )
    parser.add_argument(
        "--pred-pore-value", choices=("zero", "nonzero"), default="nonzero"
    )
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stems:
        stems = sorted(args.stems, key=natural_key)
    else:
        stems = sorted(
            collect_manual_stems(args.manual_dir)
            & collect_prediction_stems(args.pred_dir),
            key=natural_key,
        )
    if not stems:
        raise FileNotFoundError("No matching manual/predicted mask stems found.")

    rows: list[dict[str, float | str]] = []
    for stem in stems:
        manual = load_mask(
            find_manual_path(args.manual_dir, stem), args.manual_pore_value
        )
        prediction = load_mask(
            find_prediction_path(args.pred_dir, stem), args.pred_pore_value
        )
        prediction = resize_to(prediction, manual.shape)
        manual_metrics = scalar_metrics(manual, args.connectivity)
        pred_metrics = scalar_metrics(prediction, args.connectivity)
        row: dict[str, float | str] = {
            "stem": stem,
            **overlap_metrics(manual, prediction),
            **compare_scalar_metrics(manual_metrics, pred_metrics),
        }
        rows.append(row)

    numeric_keys = [key for key in rows[0] if key != "stem"]
    summary = {
        "num_slices": len(rows),
        "stems": stems,
        "connectivity": args.connectivity,
        "mean": {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in numeric_keys
        },
        "rows": rows,
        "notes": {
            "porosity": "Fraction of pixels labeled as pore.",
            "euler_number": "2D Euler number = foreground connected components - enclosed background holes.",
            "specific_perimeter": "4-neighbor pore/solid boundary length divided by total pixel count.",
            "equivalent_diameter": "Diameter of a circle with the same total pore area, in pixels.",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)

    print(f"slices={summary['num_slices']} connectivity={args.connectivity}")
    mean = summary["mean"]
    print(
        "mean dice={dice:.4f} iou={iou:.4f} "
        "manual_porosity={manual_porosity:.4f} pred_porosity={pred_porosity:.4f} "
        "delta_porosity={delta_porosity:+.4f}".format(**mean)
    )
    print(
        "mean manual_euler={manual_euler_number:.2f} pred_euler={pred_euler_number:.2f} "
        "delta_euler={delta_euler_number:+.2f}".format(**mean)
    )
    print(
        "mean manual_components={manual_component_count:.2f} pred_components={pred_component_count:.2f} "
        "manual_holes={manual_hole_count:.2f} pred_holes={pred_hole_count:.2f}".format(
            **mean
        )
    )
    print(f"Saved JSON to {args.output_json}")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
