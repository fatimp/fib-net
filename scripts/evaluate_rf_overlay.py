"""Evaluate RF segmentation from a real red/green user scribble overlay."""

from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from fibnet.interactive import (
    FEATURE_MODES,
    PORE,
    SOLID,
    UNLABELED,
    FeatureMode,
    extract_features,
    predict_pore_probability,
    train_classifier,
)
from fibnet.interactive.io import load_grayscale_image, save_outputs, validate_scribbles
from scripts.benchmark_rf_scribbles import (
    load_ground_truth,
    save_errors,
    segmentation_metrics,
)

COMPARISON_METRICS = (
    "pore_scribble_purity",
    "solid_scribble_purity",
    "dice",
    "iou",
    "precision",
    "recall",
    "pore_fraction_prediction",
    "absolute_pore_fraction_error",
    "labeled_pixel_fraction",
)
FEATURE_MODE_COMPARISON_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "accuracy",
    "pore_fraction_gt",
    "pore_fraction_prediction",
    "absolute_pore_fraction_error",
    "pore_scribble_purity",
    "solid_scribble_purity",
    "pore_labeled_pixels",
    "solid_labeled_pixels",
    "labeled_pixel_fraction",
    "feature_extraction_time",
    "rf_fit_time",
    "rf_prediction_time",
    "total_time",
    "feature_tensor_shape",
    "feature_tensor_nbytes",
)


@dataclass(frozen=True)
class ModePrediction:
    probability: np.ndarray
    prediction: np.ndarray
    feature_names: tuple[str, ...]
    feature_importances: np.ndarray
    feature_extraction_time: float
    rf_fit_time: float
    rf_prediction_time: float
    total_time: float
    feature_tensor_shape: tuple[int, int, int]
    feature_tensor_nbytes: int


def extract_overlay_scribbles(
    overlay: np.ndarray, *, expected_shape: tuple[int, int] | None = None
) -> np.ndarray:
    """Map green/red overlay pixels to semantic pore/solid labels."""
    array = np.asarray(overlay)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(
            f"Expected an RGB or RGBA scribble overlay, received shape {array.shape}."
        )
    if expected_shape is not None and array.shape[:2] != expected_shape:
        raise ValueError(
            "Image and scribble overlay must have identical spatial shapes: "
            f"image={expected_shape}, overlay={array.shape[:2]}."
        )
    rgb = array[..., :3].astype(np.int16, copy=False)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    green_pixels = (green > red) & (green > blue)
    red_pixels = (red > green) & (red > blue)
    scribbles = np.full(array.shape[:2], UNLABELED, dtype=np.uint8)
    scribbles[green_pixels] = PORE
    scribbles[red_pixels] = SOLID
    return validate_scribbles(scribbles, expected_shape=expected_shape)


def load_overlay(path: str | Path) -> np.ndarray:
    input_path = Path(path)
    try:
        with Image.open(input_path) as opened:
            return np.asarray(opened.convert("RGBA"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not read scribble overlay '{input_path}': {error}"
        ) from error


def _grayscale_preview(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(array, (0.5, 99.5))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.rint(np.clip((array - low) / (high - low), 0.0, 1.0) * 255).astype(
        np.uint8
    )


def _label_preview(scribbles: np.ndarray) -> np.ndarray:
    preview = np.zeros((*scribbles.shape, 3), dtype=np.uint8)
    preview[scribbles == PORE] = (34, 177, 76)
    preview[scribbles == SOLID] = (237, 28, 36)
    return preview


def _error_preview(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    errors = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    errors[ground_truth & prediction] = (210, 210, 210)
    errors[~ground_truth & prediction] = (255, 64, 64)
    errors[ground_truth & ~prediction] = (64, 128, 255)
    return errors


def save_overview(
    path: str | Path,
    image: np.ndarray,
    overlay: np.ndarray,
    scribbles: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    max_panel_height: int = 900,
) -> Path:
    """Save Original | Scribbles | Labels | Prediction | GT | Errors."""
    panels = [
        np.repeat(_grayscale_preview(image)[..., None], 3, axis=2),
        overlay[..., :3],
        _label_preview(scribbles),
        np.repeat((prediction.astype(np.uint8) * 255)[..., None], 3, axis=2),
        np.repeat((ground_truth.astype(np.uint8) * 255)[..., None], 3, axis=2),
        _error_preview(ground_truth, prediction),
    ]
    titles = (
        "Original",
        "Scribbles",
        "Extracted labels (pore green, solid red)",
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
        resampling = (
            Image.Resampling.BILINEAR if index < 2 else Image.Resampling.NEAREST
        )
        panel_image = Image.fromarray(panel).resize(panel_size, resampling)
        left = index * panel_size[0]
        canvas.paste(panel_image, (left, header_height))
        draw.text((left + 5, 10), title, fill="black")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def save_metrics(path: str | Path, metrics: dict[str, object]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    return output_path


def save_comparison(
    path: str | Path,
    previous_metrics_path: str | Path,
    current_metrics: dict[str, object],
) -> Path:
    """Save a compact metric-by-metric comparison with an earlier overlay run."""
    previous_path = Path(previous_metrics_path)
    with previous_path.open(encoding="utf-8", newline="") as handle:
        previous = next(csv.DictReader(handle), None)
    if previous is None:
        raise ValueError(f"No metric row found in '{previous_path}'.")
    missing = [metric for metric in COMPARISON_METRICS if metric not in previous]
    if missing:
        raise ValueError(
            f"Previous metrics file '{previous_path}' is missing: {', '.join(missing)}."
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["metric", "scribbles_v1", "scribbles_v2"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "metric": metric,
                "scribbles_v1": previous[metric],
                "scribbles_v2": current_metrics[metric],
            }
            for metric in COMPARISON_METRICS
        )
    return output_path


def save_feature_importance(
    path: str | Path,
    feature_names: tuple[str, ...],
    feature_importances: np.ndarray,
) -> Path:
    """Save channel-aligned RF feature importances in descending order."""
    importances = np.asarray(feature_importances, dtype=np.float64)
    if importances.shape != (len(feature_names),):
        raise ValueError(
            "Feature names and RF importances must have the same length: "
            f"{len(feature_names)} != {importances.shape}."
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(importances)[::-1]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "importance"])
        writer.writeheader()
        writer.writerows(
            {
                "feature": feature_names[index],
                "importance": float(importances[index]),
            }
            for index in order
        )
    return output_path


def save_feature_mode_comparison(
    path: str | Path, metrics_by_mode: dict[str, dict[str, object]]
) -> Path:
    """Save generic versus ParticleSeg3D-style results as metric rows."""
    missing_modes = [mode for mode in FEATURE_MODES if mode not in metrics_by_mode]
    if missing_modes:
        raise ValueError(
            f"Feature comparison is missing mode(s): {', '.join(missing_modes)}."
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["metric", "generic", "particleseg3d_style"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "metric": metric,
                "generic": metrics_by_mode["generic"][metric],
                "particleseg3d_style": metrics_by_mode["particleseg3d_style"][metric],
            }
            for metric in FEATURE_MODE_COMPARISON_METRICS
        )
    return output_path


def save_feature_mode_overview(
    path: str | Path,
    image: np.ndarray,
    overlay: np.ndarray,
    generic_prediction: np.ndarray,
    particleseg3d_prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    max_panel_height: int = 900,
) -> Path:
    """Save the requested six-panel generic/new-backend comparison."""
    panels = [
        np.repeat(_grayscale_preview(image)[..., None], 3, axis=2),
        overlay[..., :3],
        np.repeat((generic_prediction.astype(np.uint8) * 255)[..., None], 3, axis=2),
        np.repeat(
            (particleseg3d_prediction.astype(np.uint8) * 255)[..., None], 3, axis=2
        ),
        np.repeat((ground_truth.astype(np.uint8) * 255)[..., None], 3, axis=2),
        _error_preview(ground_truth, particleseg3d_prediction),
    ]
    titles = (
        "Original",
        "Scribbles",
        "Generic RF",
        "ParticleSeg3D-style RF",
        "Ground truth",
        "Errors for ParticleSeg3D-style (FP red, FN blue)",
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
        resampling = (
            Image.Resampling.BILINEAR if index < 2 else Image.Resampling.NEAREST
        )
        panel_image = Image.fromarray(panel).resize(panel_size, resampling)
        left = index * panel_size[0]
        canvas.paste(panel_image, (left, header_height))
        draw.text((left + 5, 10), title, fill="black")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def predict_with_feature_mode(
    image: np.ndarray,
    scribbles: np.ndarray,
    mode: FeatureMode,
    threshold: float,
) -> ModePrediction:
    """Extract one backend, fit the unchanged RF, and densely predict."""
    total_started = perf_counter()
    feature_started = perf_counter()
    extracted = extract_features(image, mode)
    feature_extraction_time = perf_counter() - feature_started
    features = extracted.feature_tensor

    fit_started = perf_counter()
    classifier = train_classifier(features, scribbles)
    rf_fit_time = perf_counter() - fit_started
    prediction_started = perf_counter()
    probability = predict_pore_probability(classifier, features)
    rf_prediction_time = perf_counter() - prediction_started
    total_time = perf_counter() - total_started
    result = ModePrediction(
        probability=probability,
        prediction=probability >= threshold,
        feature_names=extracted.feature_names,
        feature_importances=np.asarray(
            classifier.feature_importances_, dtype=np.float64
        ).copy(),
        feature_extraction_time=feature_extraction_time,
        rf_fit_time=rf_fit_time,
        rf_prediction_time=rf_prediction_time,
        total_time=total_time,
        feature_tensor_shape=features.shape,
        feature_tensor_nbytes=features.nbytes,
    )
    del classifier, extracted, features
    gc.collect()
    return result


def run(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    # Ground truth is deliberately not loaded until every dense prediction is complete.
    image = load_grayscale_image(args.image)
    overlay = load_overlay(args.scribbles)
    scribbles = extract_overlay_scribbles(overlay, expected_shape=image.shape)
    pore_labeled_pixels = int(np.count_nonzero(scribbles == PORE))
    solid_labeled_pixels = int(np.count_nonzero(scribbles == SOLID))
    modes = tuple(dict.fromkeys(args.feature_modes))
    predictions = {
        mode: predict_with_feature_mode(image, scribbles, mode, args.threshold)
        for mode in modes
    }
    ground_truth = load_ground_truth(args.mask, image.shape)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    multiple_modes = len(predictions) > 1
    metrics_by_mode: dict[str, dict[str, object]] = {}
    for mode, result in predictions.items():
        mode_dir = args.output_dir / mode if multiple_modes else args.output_dir
        metrics: dict[str, object] = {
            **segmentation_metrics(ground_truth, result.prediction),
            "pore_scribble_purity": float(ground_truth[scribbles == PORE].mean()),
            "solid_scribble_purity": float((~ground_truth)[scribbles == SOLID].mean()),
            "pore_labeled_pixels": pore_labeled_pixels,
            "solid_labeled_pixels": solid_labeled_pixels,
            "labeled_pixel_fraction": (pore_labeled_pixels + solid_labeled_pixels)
            / scribbles.size,
            "feature_extraction_time": result.feature_extraction_time,
            "rf_fit_time": result.rf_fit_time,
            "rf_prediction_time": result.rf_prediction_time,
            "total_time": result.total_time,
            "feature_tensor_shape": "x".join(
                str(value) for value in result.feature_tensor_shape
            ),
            "feature_tensor_nbytes": result.feature_tensor_nbytes,
        }
        mode_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(scribbles).save(mode_dir / "scribbles.png")
        save_outputs(mode_dir, result.probability, result.prediction)
        save_errors(mode_dir / "errors.png", ground_truth, result.prediction)
        overview_path = save_overview(
            mode_dir / "overview.png",
            image,
            overlay,
            scribbles,
            result.prediction,
            ground_truth,
        )
        metrics["overview"] = str(overview_path)
        save_metrics(mode_dir / "metrics.csv", metrics)
        if mode == "particleseg3d_style":
            save_feature_importance(
                mode_dir / "feature_importance.csv",
                result.feature_names,
                result.feature_importances,
            )
        metrics_by_mode[mode] = metrics

    if multiple_modes:
        save_feature_mode_comparison(
            args.output_dir / "comparison.csv", metrics_by_mode
        )
        save_feature_mode_overview(
            args.output_dir / "comparison_overview.png",
            image,
            overlay,
            predictions["generic"].prediction,
            predictions["particleseg3d_style"].prediction,
            ground_truth,
        )
    elif args.compare_with is not None:
        only_metrics = next(iter(metrics_by_mode.values()))
        save_comparison(
            args.output_dir / "comparison.csv", args.compare_with, only_metrics
        )
    return metrics_by_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("target/60.tiff"))
    parser.add_argument("--scribbles", type=Path, default=Path("60_scribbles.png"))
    parser.add_argument("--mask", type=Path, default=Path("ideal/60.tiff"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/rf_user_scribbles_60")
    )
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        choices=FEATURE_MODES,
        default=["generic"],
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--compare-with", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in the range [0, 1].")
    if args.compare_with is not None and len(set(args.feature_modes)) > 1:
        raise ValueError("--compare-with is supported only for a single feature mode.")
    metrics_by_mode = run(args)
    for mode, metrics in metrics_by_mode.items():
        print(f"[{mode}]")
        print(
            f"labels: pore={metrics['pore_labeled_pixels']} "
            f"solid={metrics['solid_labeled_pixels']} "
            f"fraction={metrics['labeled_pixel_fraction']:.4%}"
        )
        print(
            f"Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"pore_fraction_error={metrics['absolute_pore_fraction_error']:.4f}"
        )
        print(
            f"scribble purity: pore={metrics['pore_scribble_purity']:.4%} "
            f"solid={metrics['solid_scribble_purity']:.4%}"
        )
        print(
            f"features={metrics['feature_tensor_shape']} "
            f"feature_time={metrics['feature_extraction_time']:.3f}s "
            f"fit={metrics['rf_fit_time']:.3f}s "
            f"predict={metrics['rf_prediction_time']:.3f}s "
            f"total={metrics['total_time']:.3f}s"
        )
        print(f"overview={metrics['overview']}")
    if len(metrics_by_mode) > 1:
        print(f"comparison={args.output_dir / 'comparison.csv'}")
        print(f"comparison_overview={args.output_dir / 'comparison_overview.png'}")


if __name__ == "__main__":
    main()
