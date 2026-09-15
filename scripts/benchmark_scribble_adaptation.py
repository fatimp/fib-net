"""Benchmark short source-prior adaptation from a real scribble overlay."""

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

from fibnet.inference import predict_tiled
from fibnet.interactive import (
    MANUSCRIPT_OVERLAP,
    MANUSCRIPT_TILE_SIZE,
    PORE,
    SOLID,
    AdaptationResult,
    adapt_from_scribbles,
    load_frozen_fibnet,
    prepare_scribble_training_tiles,
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
LAMBDA_VALUES = (0.0, 0.1, 1.0, 10.0)
SOURCE_PRIOR_ADAPTATION_MODES = ("head_only", "last_decoder_head")


def _lambda_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


VARIANTS = (
    "source_model",
    *(
        f"{mode}_lambda_{_lambda_slug(lambda_value)}"
        for mode in SOURCE_PRIOR_ADAPTATION_MODES
        for lambda_value in LAMBDA_VALUES
    ),
)
COMPARISON_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "accuracy",
    "pore_fraction_prediction",
    "absolute_pore_fraction_error",
    "adaptation_time",
    "inference_time",
)


@dataclass(frozen=True)
class AdaptedPrediction:
    probability: np.ndarray
    prediction: np.ndarray
    adaptation: AdaptationResult | None
    inference_time: float


def _predict_native(
    model: torch.nn.Module,
    current: Path,
    previous: Path,
    next_path: Path,
    feature_mode: str,
    device: str,
) -> tuple[np.ndarray, float]:
    started = perf_counter()
    model.eval()
    with torch.no_grad():
        probability = predict_tiled(
            model,
            current,
            MANUSCRIPT_TILE_SIZE,
            MANUSCRIPT_OVERLAP,
            feature_mode,
            device,
            previous_path=previous,
            next_path=next_path,
        ).astype(np.float32, copy=False)
    return probability, perf_counter() - started


def _save_history(path: Path, adaptation: AdaptationResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "tile_index",
        "labeled_pixels",
        "total_loss",
        "scribble_loss",
        "consistency_loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: getattr(step, field) for field in fieldnames}
            for step in adaptation.history
        )
    return path


def _metrics(
    ground_truth: np.ndarray,
    scribbles: np.ndarray,
    result: AdaptedPrediction,
) -> dict[str, object]:
    pore_labeled = int(np.count_nonzero(scribbles == PORE))
    solid_labeled = int(np.count_nonzero(scribbles == SOLID))
    adaptation = result.adaptation
    return {
        **segmentation_metrics(ground_truth, result.prediction),
        "threshold": THRESHOLD,
        "pore_scribble_purity": float(ground_truth[scribbles == PORE].mean()),
        "solid_scribble_purity": float((~ground_truth)[scribbles == SOLID].mean()),
        "pore_labeled_pixels": pore_labeled,
        "solid_labeled_pixels": solid_labeled,
        "labeled_pixel_fraction": (pore_labeled + solid_labeled) / scribbles.size,
        "adaptation_mode": "none" if adaptation is None else adaptation.mode,
        "lambda_consistency": (
            "" if adaptation is None else adaptation.lambda_consistency
        ),
        "steps": 0 if adaptation is None else adaptation.steps,
        "learning_rate": "" if adaptation is None else adaptation.learning_rate,
        "trainable_parameters": (
            0 if adaptation is None else adaptation.trainable_parameters
        ),
        "adaptation_time": 0.0 if adaptation is None else adaptation.adaptation_time,
        "inference_time": result.inference_time,
        "initial_scribble_loss": (
            "" if adaptation is None else adaptation.history[0].scribble_loss
        ),
        "final_scribble_loss": (
            "" if adaptation is None else adaptation.history[-1].scribble_loss
        ),
        "initial_consistency_loss": (
            "" if adaptation is None else adaptation.history[0].consistency_loss
        ),
        "final_consistency_loss": (
            "" if adaptation is None else adaptation.history[-1].consistency_loss
        ),
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
    result: AdaptedPrediction,
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
    if result.adaptation is not None:
        _save_history(output_dir / "training.csv", result.adaptation)


def run(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    image = load_grayscale_image(args.current)
    overlay = load_overlay(args.scribbles)
    scribbles = extract_overlay_scribbles(overlay, expected_shape=image.shape)
    frozen = load_frozen_fibnet(args.checkpoint, device=args.device)
    if frozen.feature_mode != "stack_relief" or frozen.model_arch != "resunet":
        raise ValueError(
            "Adaptation requires the production resunet/stack_relief source "
            f"checkpoint, received {frozen.model_arch}/{frozen.feature_mode}."
        )
    if frozen.image_size != MANUSCRIPT_TILE_SIZE:
        raise ValueError(
            "Checkpoint patch size differs from the locked manuscript tile size: "
            f"{frozen.image_size} != {MANUSCRIPT_TILE_SIZE}."
        )

    source_probability, source_inference_time = _predict_native(
        frozen.model,
        args.current,
        args.previous,
        args.next,
        frozen.feature_mode,
        args.device,
    )
    predictions: dict[str, AdaptedPrediction] = {
        "source_model": AdaptedPrediction(
            probability=source_probability,
            prediction=source_probability >= THRESHOLD,
            adaptation=None,
            inference_time=source_inference_time,
        )
    }
    training_tiles = prepare_scribble_training_tiles(
        frozen,
        args.previous,
        args.current,
        args.next,
        scribbles,
        tile_size=MANUSCRIPT_TILE_SIZE,
        overlap=MANUSCRIPT_OVERLAP,
    )
    source_state = {
        name: value.detach().cpu().clone()
        for name, value in frozen.model.state_dict().items()
    }

    for mode in SOURCE_PRIOR_ADAPTATION_MODES:
        for lambda_value in LAMBDA_VALUES:
            frozen.model.load_state_dict(source_state)
            adaptation = adapt_from_scribbles(
                frozen,
                training_tiles,
                mode,
                lambda_value,
                steps=args.steps,
                learning_rate=args.learning_rate,
                random_state=42,
            )
            probability, inference_time = _predict_native(
                frozen.model,
                args.current,
                args.previous,
                args.next,
                frozen.feature_mode,
                args.device,
            )
            variant = f"{mode}_lambda_{_lambda_slug(lambda_value)}"
            predictions[variant] = AdaptedPrediction(
                probability=probability,
                prediction=probability >= THRESHOLD,
                adaptation=adaptation,
                inference_time=inference_time,
            )
            if frozen.device.type == "cuda":
                torch.cuda.empty_cache()

    # Evaluation data is deliberately unavailable during adaptation and inference.
    ground_truth = load_ground_truth(args.mask, image.shape)
    metrics_by_variant = {}
    for variant in VARIANTS:
        result = predictions[variant]
        metrics = _metrics(ground_truth, scribbles, result)
        _save_variant(
            args.output_dir / variant,
            image,
            overlay,
            scribbles,
            ground_truth,
            result,
            metrics,
        )
        metrics_by_variant[variant] = metrics
    _save_comparison(args.output_dir / "comparison.csv", metrics_by_variant)

    metadata = {
        "checkpoint": str(frozen.checkpoint),
        "native_image_shape": list(image.shape),
        "feature_mode": frozen.feature_mode,
        "model_arch": frozen.model_arch,
        "tile_size": MANUSCRIPT_TILE_SIZE,
        "overlap": MANUSCRIPT_OVERLAP,
        "overlap_weighting": "production triangular edge-distance weights",
        "stack_aggregation": "none",
        "threshold": THRESHOLD,
        "optimization_steps": args.steps,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "scribble_loss": "binary cross entropy with logits on labeled pixels",
        "consistency_loss": "probability MSE on valid unlabeled pixels",
        "lambda_values": list(LAMBDA_VALUES),
        "random_state": 42,
        "training_tiles": len(training_tiles),
        "unique_labeled_pixels": int(np.count_nonzero(scribbles)),
        "ground_truth_used_for": "final evaluation only",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    del source_state, training_tiles
    gc.collect()
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
        "--output-dir", type=Path, default=Path("outputs/scribble_adaptation_60")
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
            f"pore_fraction_error={metrics['absolute_pore_fraction_error']:.4f} "
            f"adapt={metrics['adaptation_time']:.3f}s "
            f"infer={metrics['inference_time']:.3f}s"
        )
    print(f"comparison={args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
