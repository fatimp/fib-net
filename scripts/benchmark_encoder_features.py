"""Benchmark native-resolution tiled FIB-NET features with real scribbles."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from fibnet.inference import predict_tiled
from fibnet.interactive import (
    MANUSCRIPT_OVERLAP,
    MANUSCRIPT_TILE_SIZE,
    PORE,
    SOLID,
    build_training_set,
    extract_tiled_deep_features,
    load_frozen_fibnet,
    predict_pore_probability,
)
from fibnet.interactive.io import load_grayscale_image, save_outputs
from scripts.benchmark_rf_scribbles import (
    load_ground_truth,
    save_errors,
    segmentation_metrics,
)
from scripts.evaluate_rf_overlay import (
    extract_overlay_scribbles,
    load_overlay,
    save_metrics,
    save_overview,
)

THRESHOLD = 0.5
VARIANTS = (
    "source_model",
    "shallow_encoder_lr_balanced",
    "shallow_encoder_lr_unweighted",
    "decoder_lr_balanced",
    "decoder_lr_unweighted",
    "decoder_plus_logit_lr_balanced",
    "decoder_plus_logit_lr_unweighted",
)
COMPARISON_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "accuracy",
    "pore_fraction_prediction",
    "absolute_pore_fraction_error",
)


@dataclass(frozen=True)
class DensePrediction:
    probability: np.ndarray
    prediction: np.ndarray
    fit_time: float
    prediction_time: float


def fit_logistic_regression(
    features: np.ndarray,
    scribbles: np.ndarray,
    *,
    balanced: bool,
) -> DensePrediction:
    """Fit only on scribbles and predict all pixels at the fixed threshold."""
    training_features, training_labels = build_training_set(features, scribbles)
    fit_started = perf_counter()
    classifier = LogisticRegression(
        class_weight="balanced" if balanced else None,
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(training_features, training_labels)
    fit_time = perf_counter() - fit_started
    del training_features, training_labels

    prediction_started = perf_counter()
    probability = predict_pore_probability(classifier, features)
    prediction_time = perf_counter() - prediction_started
    result = DensePrediction(
        probability=probability,
        prediction=probability >= THRESHOLD,
        fit_time=fit_time,
        prediction_time=prediction_time,
    )
    del classifier
    gc.collect()
    return result


def _run_weight_variants(
    representation: str,
    features: np.ndarray,
    scribbles: np.ndarray,
    predictions: dict[str, DensePrediction],
) -> None:
    for balanced in (True, False):
        weighting = "balanced" if balanced else "unweighted"
        predictions[f"{representation}_lr_{weighting}"] = fit_logistic_regression(
            features,
            scribbles,
            balanced=balanced,
        )


def _prediction_metrics(
    ground_truth: np.ndarray,
    scribbles: np.ndarray,
    result: DensePrediction,
    *,
    representation: str,
    feature_shape: tuple[int, int, int] | None,
    feature_nbytes: int,
    feature_extraction_time: float,
    class_weight: str,
) -> dict[str, object]:
    pore_labeled = int(np.count_nonzero(scribbles == PORE))
    solid_labeled = int(np.count_nonzero(scribbles == SOLID))
    return {
        **segmentation_metrics(ground_truth, result.prediction),
        "representation": representation,
        "class_weight": class_weight,
        "threshold": THRESHOLD,
        "pore_scribble_purity": float(ground_truth[scribbles == PORE].mean()),
        "solid_scribble_purity": float((~ground_truth)[scribbles == SOLID].mean()),
        "pore_labeled_pixels": pore_labeled,
        "solid_labeled_pixels": solid_labeled,
        "labeled_pixel_fraction": (pore_labeled + solid_labeled) / scribbles.size,
        "feature_extraction_time": feature_extraction_time,
        "fit_time": result.fit_time,
        "prediction_time": result.prediction_time,
        "feature_tensor_shape": (
            "" if feature_shape is None else "x".join(map(str, feature_shape))
        ),
        "feature_tensor_nbytes": feature_nbytes,
        "feature_channels": 0 if feature_shape is None else feature_shape[-1],
    }


def _save_comparison(
    path: Path, metrics_by_variant: dict[str, dict[str, object]]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", *VARIANTS])
        writer.writeheader()
        writer.writerows(
            {
                "metric": metric,
                **{
                    variant: metrics_by_variant[variant][metric] for variant in VARIANTS
                },
            }
            for metric in COMPARISON_METRICS
        )
    return path


def _save_variant(
    output_dir: Path,
    image: np.ndarray,
    overlay: np.ndarray,
    scribbles: np.ndarray,
    ground_truth: np.ndarray,
    result: DensePrediction,
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scribbles).save(output_dir / "scribbles.png")
    save_outputs(output_dir, result.probability, result.prediction)
    save_errors(output_dir / "errors.png", ground_truth, result.prediction)
    overview = save_overview(
        output_dir / "overview.png",
        image,
        overlay,
        scribbles,
        result.prediction,
        ground_truth,
    )
    metrics["overview"] = str(overview)
    save_metrics(output_dir / "metrics.csv", metrics)


def run(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    image = load_grayscale_image(args.current)
    overlay = load_overlay(args.scribbles)
    scribbles = extract_overlay_scribbles(overlay, expected_shape=image.shape)
    frozen = load_frozen_fibnet(args.checkpoint, device=args.device)
    if frozen.feature_mode != "stack_relief" or frozen.model_arch != "resunet":
        raise ValueError(
            "The source benchmark requires the production resunet/stack_relief "
            f"checkpoint, received {frozen.model_arch}/{frozen.feature_mode}."
        )
    if frozen.image_size != MANUSCRIPT_TILE_SIZE:
        raise ValueError(
            "Checkpoint patch size differs from the locked manuscript tile size: "
            f"{frozen.image_size} != {MANUSCRIPT_TILE_SIZE}."
        )

    source_started = perf_counter()
    with torch.no_grad():
        source_probability = predict_tiled(
            frozen.model,
            args.current,
            MANUSCRIPT_TILE_SIZE,
            MANUSCRIPT_OVERLAP,
            frozen.feature_mode,
            args.device,
            previous_path=args.previous,
            next_path=args.next,
        ).astype(np.float32, copy=False)
    source_prediction_time = perf_counter() - source_started
    predictions: dict[str, DensePrediction] = {
        "source_model": DensePrediction(
            probability=source_probability,
            prediction=source_probability >= THRESHOLD,
            fit_time=0.0,
            prediction_time=source_prediction_time,
        )
    }

    extracted = extract_tiled_deep_features(
        frozen,
        args.previous,
        args.current,
        args.next,
        tile_size=MANUSCRIPT_TILE_SIZE,
        overlap=MANUSCRIPT_OVERLAP,
    )
    feature_extraction_time = extracted.extraction_time
    tile_count = extracted.tile_count
    shallow = extracted.shallow_encoder.feature_tensor
    decoder = extracted.decoder.feature_tensor
    del extracted

    representation_metadata: dict[str, tuple[tuple[int, int, int], int]] = {
        "shallow_encoder": (shallow.shape, shallow.nbytes),
        "decoder": (decoder.shape, decoder.nbytes),
    }
    _run_weight_variants("shallow_encoder", shallow, scribbles, predictions)
    del shallow
    gc.collect()
    _run_weight_variants("decoder", decoder, scribbles, predictions)

    decoder_plus_logit = np.empty(
        (*decoder.shape[:2], decoder.shape[-1] + 1), dtype=np.float32
    )
    decoder_plus_logit[..., :-1] = decoder
    decoder_plus_logit[..., -1] = source_probability
    representation_metadata["decoder_plus_logit"] = (
        decoder_plus_logit.shape,
        decoder_plus_logit.nbytes,
    )
    del decoder
    gc.collect()
    _run_weight_variants(
        "decoder_plus_logit", decoder_plus_logit, scribbles, predictions
    )
    del decoder_plus_logit
    gc.collect()

    # Ground truth is loaded only after the source and all six LR predictions.
    ground_truth = load_ground_truth(args.mask, image.shape)
    metrics_by_variant: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        if variant == "source_model":
            representation = "source_model"
            shape = None
            nbytes = 0
            extraction_time = 0.0
            class_weight = "not_applicable"
        else:
            representation = variant.split("_lr_", maxsplit=1)[0]
            shape, nbytes = representation_metadata[representation]
            extraction_time = feature_extraction_time
            class_weight = "balanced" if variant.endswith("balanced") else "none"
        metrics = _prediction_metrics(
            ground_truth,
            scribbles,
            predictions[variant],
            representation=representation,
            feature_shape=shape,
            feature_nbytes=nbytes,
            feature_extraction_time=extraction_time,
            class_weight=class_weight,
        )
        _save_variant(
            args.output_dir / variant,
            image,
            overlay,
            scribbles,
            ground_truth,
            predictions[variant],
            metrics,
        )
        metrics_by_variant[variant] = metrics

    _save_comparison(args.output_dir / "comparison.csv", metrics_by_variant)
    metadata = {
        "checkpoint": str(frozen.checkpoint),
        "model_arch": frozen.model_arch,
        "feature_mode": frozen.feature_mode,
        "native_image_shape": list(image.shape),
        "tile_size": MANUSCRIPT_TILE_SIZE,
        "overlap": MANUSCRIPT_OVERLAP,
        "overlap_weighting": "production triangular edge-distance weights",
        "stack_aggregation": "none",
        "threshold": THRESHOLD,
        "tile_count": tile_count,
        "pca": None,
        "feature_extraction_time": feature_extraction_time,
        "source_model_prediction_time": source_prediction_time,
        "representations": {
            name: {"shape": list(shape), "nbytes": nbytes}
            for name, (shape, nbytes) in representation_metadata.items()
        },
        "decoder_plus_logit_added_channel": "production source-model probability",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metrics_by_variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=Path("target/59.tiff"))
    parser.add_argument("--current", type=Path, default=Path("target/60.tiff"))
    parser.add_argument("--next", type=Path, default=Path("target/61.tiff"))
    parser.add_argument("--scribbles", type=Path, default=Path("60_scribbles2.png"))
    parser.add_argument("--mask", type=Path, default=Path("ideal/60.tiff"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("weights/fibnet_source_v0.1.pt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/rf_tiled_deep_features_60")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_by_variant = run(args)
    for variant in VARIANTS:
        metrics = metrics_by_variant[variant]
        print(
            f"{variant}: Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"pore_fraction={metrics['pore_fraction_prediction']:.4f} "
            f"pore_fraction_error={metrics['absolute_pore_fraction_error']:.4f}"
        )
    print(f"comparison={args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
