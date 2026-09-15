"""Benchmark sparse-scribble RF segmentation on a fully labeled FIB-SEM pair."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import label as connected_components
from skimage.morphology import binary_erosion, disk

from fibnet.evaluation import load_mask
from fibnet.interactive import (
    PORE,
    SOLID,
    UNLABELED,
    extract_multiscale_features,
    predict_pore_probability,
    train_classifier,
)
from fibnet.interactive.io import load_grayscale_image, save_outputs

DEFAULT_BUDGETS = (3, 5, 10, 20)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
QUALITY_METRICS = (
    "dice",
    "iou",
    "accuracy",
    "precision",
    "recall",
    "pore_fraction_gt",
    "pore_fraction_prediction",
    "absolute_pore_fraction_error",
)
EFFORT_METRICS = (
    "labeled_pixel_fraction",
    "pore_labeled_pixels",
    "solid_labeled_pixels",
)
TIME_METRICS = (
    "feature_extraction_time",
    "rf_fit_time",
    "rf_prediction_time",
    "total_time",
)


def load_ground_truth(path: str | Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load a manual mask using FIB-NET's established black-pore convention."""
    pore = load_mask(Path(path), pore_value="zero")
    if pore.shape != expected_shape:
        raise ValueError(
            "Image and ground-truth mask must have identical shapes: "
            f"image={expected_shape}, mask={pore.shape}."
        )
    if not np.any(pore):
        raise ValueError("Ground truth does not contain any pore pixels (value 0).")
    if np.all(pore):
        raise ValueError("Ground truth does not contain any solid pixels (nonzero).")
    return pore


def _validate_stroke_options(
    stroke_count: int, stroke_length: int, brush_width: int
) -> None:
    if stroke_count < 1:
        raise ValueError("stroke_count must be at least 1 per class.")
    if stroke_length < 1:
        raise ValueError("stroke_length must be at least 1 pixel.")
    if brush_width < 1 or brush_width % 2 == 0:
        raise ValueError("brush_width must be a positive odd number of pixels.")


def _eligible_components(
    class_mask: np.ndarray,
    *,
    stroke_length: int,
    brush_radius: int,
    class_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    footprint = disk(brush_radius)
    safe_centers = binary_erosion(class_mask, footprint=footprint, mode="min")
    components, count = connected_components(
        safe_centers, connectivity=2, return_num=True
    )
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    eligible_ids = np.flatnonzero(sizes >= stroke_length)
    eligible_ids = eligible_ids[eligible_ids != 0]
    eligible = np.isin(components, eligible_ids)
    if not np.any(eligible):
        raise ValueError(
            f"Cannot generate a {class_name} stroke of length {stroke_length} "
            f"with brush width {2 * brush_radius + 1}: the class is too small "
            "after erosion."
        )
    return eligible, components


def _random_walk(
    eligible: np.ndarray,
    components: np.ndarray,
    *,
    length: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    candidates = np.argwhere(eligible)
    row, column = candidates[rng.integers(candidates.shape[0])]
    component_id = components[row, column]
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)
    path = [(int(row), int(column))]
    visited = {path[0]}
    neighbor_steps = np.array(
        [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ],
        dtype=np.int32,
    )

    for _ in range(length - 1):
        neighbor_rows = row + neighbor_steps[:, 0]
        neighbor_columns = column + neighbor_steps[:, 1]
        inside = (
            (neighbor_rows >= 0)
            & (neighbor_rows < eligible.shape[0])
            & (neighbor_columns >= 0)
            & (neighbor_columns < eligible.shape[1])
        )
        steps = neighbor_steps[inside]
        neighbor_rows = neighbor_rows[inside]
        neighbor_columns = neighbor_columns[inside]
        same_component = components[neighbor_rows, neighbor_columns] == component_id
        steps = steps[same_component]
        neighbor_rows = neighbor_rows[same_component]
        neighbor_columns = neighbor_columns[same_component]
        if not steps.size:
            raise RuntimeError("An eligible stroke component unexpectedly has no path.")

        unvisited = np.array(
            [
                (int(next_row), int(next_column)) not in visited
                for next_row, next_column in zip(
                    neighbor_rows, neighbor_columns, strict=True
                )
            ]
        )
        if np.any(unvisited):
            steps = steps[unvisited]
            neighbor_rows = neighbor_rows[unvisited]
            neighbor_columns = neighbor_columns[unvisited]

        unit_steps = steps / np.linalg.norm(steps, axis=1, keepdims=True)
        alignment = unit_steps @ direction
        weights = np.exp(2.5 * alignment)
        selected = int(rng.choice(len(steps), p=weights / weights.sum()))
        row = int(neighbor_rows[selected])
        column = int(neighbor_columns[selected])
        step_direction = unit_steps[selected]
        direction = 0.75 * direction + 0.25 * step_direction
        direction /= np.linalg.norm(direction)
        point = (row, column)
        path.append(point)
        visited.add(point)
    return path


def _paint_strokes(
    class_mask: np.ndarray,
    *,
    stroke_count: int,
    stroke_length: int,
    brush_width: int,
    class_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    brush_radius = brush_width // 2
    eligible, components = _eligible_components(
        class_mask,
        stroke_length=stroke_length,
        brush_radius=brush_radius,
        class_name=class_name,
    )
    brush_offsets = np.argwhere(disk(brush_radius)) - brush_radius
    painted = np.zeros(class_mask.shape, dtype=bool)
    for _ in range(stroke_count):
        path = _random_walk(
            eligible,
            components,
            length=stroke_length,
            rng=rng,
        )
        for row, column in path:
            rows = row + brush_offsets[:, 0]
            columns = column + brush_offsets[:, 1]
            painted[rows, columns] = True
    if np.any(painted & ~class_mask):
        raise RuntimeError(f"Generated {class_name} strokes crossed a class boundary.")
    return painted


def generate_scribbles(
    pore_ground_truth: np.ndarray,
    *,
    stroke_count: int,
    stroke_length: int = 80,
    brush_width: int = 7,
    seed: int = 0,
) -> np.ndarray:
    """Generate reproducible, connected pore and solid brush strokes."""
    _validate_stroke_options(stroke_count, stroke_length, brush_width)
    pore = np.asarray(pore_ground_truth, dtype=bool)
    if pore.ndim != 2 or pore.size == 0:
        raise ValueError("pore_ground_truth must be a non-empty 2D array.")

    rng = np.random.default_rng(seed)
    pore_strokes = _paint_strokes(
        pore,
        stroke_count=stroke_count,
        stroke_length=stroke_length,
        brush_width=brush_width,
        class_name="pore",
        rng=rng,
    )
    solid_strokes = _paint_strokes(
        ~pore,
        stroke_count=stroke_count,
        stroke_length=stroke_length,
        brush_width=brush_width,
        class_name="solid",
        rng=rng,
    )
    scribbles = np.full(pore.shape, UNLABELED, dtype=np.uint8)
    scribbles[pore_strokes] = PORE
    scribbles[solid_strokes] = SOLID
    if not np.any(scribbles == UNLABELED):
        raise ValueError(
            "Stroke settings label the entire image; reduce the budget, length, "
            "or brush width."
        )
    return scribbles


def segmentation_metrics(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """Compute pore-class overlap and pore-fraction metrics."""
    ground_truth = np.asarray(ground_truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if ground_truth.shape != prediction.shape:
        raise ValueError("Ground truth and prediction must have identical shapes.")
    true_positive = int(np.count_nonzero(ground_truth & prediction))
    false_positive = int(np.count_nonzero(~ground_truth & prediction))
    false_negative = int(np.count_nonzero(ground_truth & ~prediction))
    true_negative = int(np.count_nonzero(~ground_truth & ~prediction))
    ground_truth_positive = true_positive + false_negative
    predicted_positive = true_positive + false_positive
    dice_denominator = ground_truth_positive + predicted_positive
    union = true_positive + false_positive + false_negative
    total = ground_truth.size
    pore_fraction_gt = ground_truth_positive / total
    pore_fraction_prediction = predicted_positive / total
    return {
        "dice": 2.0 * true_positive / dice_denominator if dice_denominator else 1.0,
        "iou": true_positive / union if union else 1.0,
        "accuracy": (true_positive + true_negative) / total,
        "precision": true_positive / predicted_positive if predicted_positive else 1.0,
        "recall": true_positive / ground_truth_positive
        if ground_truth_positive
        else 1.0,
        "pore_fraction_gt": pore_fraction_gt,
        "pore_fraction_prediction": pore_fraction_prediction,
        "absolute_pore_fraction_error": abs(
            pore_fraction_prediction - pore_fraction_gt
        ),
    }


def _error_image(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    errors = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    errors[ground_truth & prediction] = (210, 210, 210)
    errors[~ground_truth & prediction] = (255, 64, 64)  # False positive: red.
    errors[ground_truth & ~prediction] = (64, 128, 255)  # False negative: blue.
    return errors


def _grayscale_preview(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, (0.5, 99.5))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


def _scribble_preview(image: np.ndarray, scribbles: np.ndarray) -> np.ndarray:
    preview = np.repeat(_grayscale_preview(image)[..., None], 3, axis=2)
    preview = np.rint(preview.astype(np.float32) * 0.55).astype(np.uint8)
    preview[scribbles == PORE] = (0, 255, 255)
    preview[scribbles == SOLID] = (255, 215, 0)
    return preview


def save_errors(
    path: str | Path, ground_truth: np.ndarray, prediction: np.ndarray
) -> Path:
    """Save correct pores in gray, false positives in red, and false negatives in blue."""
    output_path = Path(path)
    Image.fromarray(_error_image(ground_truth, prediction)).save(output_path)
    return output_path


def save_overview(
    path: str | Path,
    image: np.ndarray,
    scribbles: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    max_panel_height: int = 900,
) -> Path:
    """Save the five-panel benchmark overview requested for one budget."""
    panels = [
        np.repeat(_grayscale_preview(image)[..., None], 3, axis=2),
        _scribble_preview(image, scribbles),
        np.repeat((prediction.astype(np.uint8) * 255)[..., None], 3, axis=2),
        np.repeat((ground_truth.astype(np.uint8) * 255)[..., None], 3, axis=2),
        _error_image(ground_truth, prediction),
    ]
    titles = (
        "Original",
        "Scribbles (pore cyan, solid yellow)",
        "Prediction",
        "Ground truth",
        "Errors (FP red, FN blue)",
    )
    scale = min(1.0, max_panel_height / image.shape[0])
    panel_size = (
        max(1, round(image.shape[1] * scale)),
        max(1, round(image.shape[0] * scale)),
    )
    header_height = 34
    canvas = Image.new(
        "RGB", (panel_size[0] * len(panels), panel_size[1] + header_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        panel_image = Image.fromarray(panel).resize(
            panel_size,
            Image.Resampling.BILINEAR if index < 2 else Image.Resampling.NEAREST,
        )
        left = index * panel_size[0]
        canvas.paste(panel_image, (left, header_height))
        draw.text((left + 5, 10), title, fill="black")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty results to '{path}'.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate numeric benchmark measurements by budget using sample std."""
    summaries: list[dict[str, object]] = []
    metrics = (*QUALITY_METRICS, *EFFORT_METRICS, *TIME_METRICS)
    budgets = sorted({int(row["budget"]) for row in rows})
    for budget in budgets:
        group = [row for row in rows if int(row["budget"]) == budget]
        summary: dict[str, object] = {
            "budget": budget,
            "runs": len(group),
            "feature_tensor_shape": group[0]["feature_tensor_shape"],
            "feature_tensor_nbytes": group[0]["feature_tensor_nbytes"],
        }
        for metric in metrics:
            values = np.array([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def run_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, object]], Path]:
    """Execute all configured budgets and seeds and write benchmark artifacts."""
    benchmark_started = perf_counter()
    image = load_grayscale_image(args.image)
    ground_truth = load_ground_truth(args.mask, image.shape)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_started = perf_counter()
    features = extract_multiscale_features(image)
    feature_extraction_time = perf_counter() - feature_started
    feature_shape = "x".join(str(dimension) for dimension in features.shape)
    print(
        f"image={image.shape} features={features.shape} "
        f"feature_memory={features.nbytes / 1024**2:.2f} MiB "
        f"feature_time={feature_extraction_time:.3f} s"
    )

    rows: list[dict[str, object]] = []
    overview_paths: list[Path] = []
    for budget in args.budgets:
        for seed in args.seeds:
            run_started = perf_counter()
            scribbles = generate_scribbles(
                ground_truth,
                stroke_count=budget,
                stroke_length=args.stroke_length,
                brush_width=args.brush_width,
                seed=seed,
            )
            run_dir = args.output_dir / f"budget_{budget:02d}" / f"seed_{seed:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(scribbles).save(run_dir / "scribbles.png")

            fit_started = perf_counter()
            classifier = train_classifier(features, scribbles)
            rf_fit_time = perf_counter() - fit_started
            prediction_started = perf_counter()
            probability = predict_pore_probability(classifier, features)
            rf_prediction_time = perf_counter() - prediction_started
            prediction = probability >= args.threshold
            save_outputs(run_dir, probability, prediction)
            save_errors(run_dir / "errors.png", ground_truth, prediction)

            metrics = segmentation_metrics(ground_truth, prediction)
            pore_labeled = int(np.count_nonzero(scribbles == PORE))
            solid_labeled = int(np.count_nonzero(scribbles == SOLID))
            labeled_fraction = (pore_labeled + solid_labeled) / scribbles.size
            measured_run_time = perf_counter() - run_started
            row: dict[str, object] = {
                "budget": budget,
                "seed": seed,
                "pore_strokes": budget,
                "solid_strokes": budget,
                "stroke_length": args.stroke_length,
                "brush_width": args.brush_width,
                **metrics,
                "labeled_pixel_fraction": labeled_fraction,
                "pore_labeled_pixels": pore_labeled,
                "solid_labeled_pixels": solid_labeled,
                "feature_extraction_time": feature_extraction_time,
                "rf_fit_time": rf_fit_time,
                "rf_prediction_time": rf_prediction_time,
                "total_time": feature_extraction_time + measured_run_time,
                "measured_run_time_without_shared_features": measured_run_time,
                "feature_tensor_shape": feature_shape,
                "feature_tensor_nbytes": features.nbytes,
                "output_dir": str(run_dir),
            }
            rows.append(row)
            _write_csv(args.output_dir / "runs.csv", rows)
            if seed == args.overview_seed:
                overview_path = save_overview(
                    args.output_dir
                    / f"budget_{budget:02d}"
                    / f"overview_seed_{seed:03d}.png",
                    image,
                    scribbles,
                    prediction,
                    ground_truth,
                )
                overview_paths.append(overview_path)
            print(
                f"budget={budget:02d} seed={seed:03d} "
                f"dice={metrics['dice']:.4f} iou={metrics['iou']:.4f} "
                f"labeled={labeled_fraction:.4%} fit={rf_fit_time:.3f}s "
                f"predict={rf_prediction_time:.3f}s"
            )

    summary_path = args.output_dir / "summary.csv"
    _write_csv(summary_path, summarize_runs(rows))
    print(
        f"Completed {len(rows)} runs in {perf_counter() - benchmark_started:.3f} s; "
        f"runs={args.output_dir / 'runs.csv'} summary={summary_path}"
    )
    for overview_path in overview_paths:
        print(f"overview={overview_path}")
    return rows, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("target/60.tiff"))
    parser.add_argument("--mask", type=Path, default=Path("ideal/60.tiff"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rf_benchmark"))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--stroke-length", type=int, default=80)
    parser.add_argument("--brush-width", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overview-seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in the range [0, 1].")
    if any(budget < 1 for budget in args.budgets):
        raise ValueError("All budgets must be positive integers.")
    if args.overview_seed not in args.seeds:
        raise ValueError("overview_seed must be included in seeds.")
    run_benchmark(args)


if __name__ == "__main__":
    main()
