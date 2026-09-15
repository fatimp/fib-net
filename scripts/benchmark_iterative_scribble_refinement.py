"""Run sequential source -> round 1 -> round 2 interactive adaptation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image, ImageDraw

from fibnet.inference import predict_tiled
from fibnet.interactive import (
    INTERACTIVE_ADAPTATION_BASELINE,
    MANUSCRIPT_OVERLAP,
    MANUSCRIPT_TILE_SIZE,
    PORE,
    SOLID,
    AdaptationResult,
    FrozenFibNet,
    adapt_from_scribbles,
    load_frozen_fibnet,
    prepare_scribble_training_tiles,
)
from fibnet.interactive.io import load_grayscale_image, save_outputs
from scripts.benchmark_rf_scribbles import load_ground_truth, segmentation_metrics
from scripts.evaluate_rf_overlay import extract_overlay_scribbles, load_overlay


@dataclass(frozen=True)
class RoundPrediction:
    probability: np.ndarray
    prediction: np.ndarray
    adaptation: AdaptationResult | None
    inference_time: float


def validate_cumulative_scribbles(round1: np.ndarray, round2: np.ndarray) -> np.ndarray:
    """Require round 2 to retain every round-1 label and add corrections."""
    first = np.asarray(round1)
    second = np.asarray(round2)
    if first.shape != second.shape:
        raise ValueError(
            f"Cumulative scribble shapes differ: {first.shape} != {second.shape}."
        )
    original = first != 0
    changed = original & (second != first)
    if np.any(changed):
        removed = int(np.count_nonzero(changed & (second == 0)))
        relabeled = int(np.count_nonzero(changed & (second != 0)))
        raise ValueError(
            "Round-2 cumulative scribbles must preserve every round-1 label; "
            f"removed={removed}, relabeled={relabeled}."
        )
    new_labels = (~original) & (second != 0)
    if not np.any(new_labels):
        raise ValueError(
            "Round-2 cumulative overlay contains no new correction scribbles."
        )
    return second.astype(np.uint8, copy=False)


def _predict_native(
    model: torch.nn.Module,
    current: Path,
    previous: Path,
    next_path: Path,
    feature_mode: str,
    device: str,
) -> tuple[np.ndarray, float]:
    model.eval()
    started = perf_counter()
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


def _adapt_round(
    frozen: FrozenFibNet,
    args: argparse.Namespace,
    scribbles: np.ndarray,
) -> AdaptationResult:
    config = INTERACTIVE_ADAPTATION_BASELINE
    training_tiles = prepare_scribble_training_tiles(
        frozen,
        args.previous,
        args.current,
        args.next,
        scribbles,
        tile_size=MANUSCRIPT_TILE_SIZE,
        overlap=MANUSCRIPT_OVERLAP,
    )
    return adapt_from_scribbles(
        frozen,
        training_tiles,
        config.mode,
        config.lambda_consistency,
        steps=config.steps,
        learning_rate=config.learning_rate,
        random_state=42,
    )


def _rgb_mask(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def _gray_preview(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(values, (0.5, 99.5))
    if high <= low:
        gray = np.zeros(values.shape, dtype=np.uint8)
    else:
        gray = np.rint(np.clip((values - low) / (high - low), 0, 1) * 255).astype(
            np.uint8
        )
    return np.repeat(gray[..., None], 3, axis=2)


def _save_panels(
    path: Path,
    panels: tuple[np.ndarray, ...],
    titles: tuple[str, ...],
    *,
    native_height: int,
    max_panel_height: int = 900,
) -> Path:
    scale = min(1.0, max_panel_height / native_height)
    panel_size = (
        max(1, round(panels[0].shape[1] * scale)),
        max(1, round(panels[0].shape[0] * scale)),
    )
    header_height = 34
    canvas = Image.new(
        "RGB", (panel_size[0] * len(panels), panel_size[1] + header_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        resized = Image.fromarray(panel[..., :3]).resize(
            panel_size, Image.Resampling.NEAREST
        )
        left = index * panel_size[0]
        canvas.paste(resized, (left, header_height))
        draw.text((left + 5, 10), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def save_round1_review(
    output_dir: Path,
    image: np.ndarray,
    overlay1: np.ndarray,
    source: RoundPrediction,
    round1: RoundPrediction,
) -> tuple[Path, Path]:
    """Save a GT-free review and cumulative overlay template for manual edits."""
    review = _save_panels(
        output_dir / "round1_review_no_gt.png",
        (
            _gray_preview(image),
            _rgb_mask(source.prediction),
            overlay1[..., :3],
            _rgb_mask(round1.prediction),
        ),
        (
            "Original",
            "Source prediction",
            "Cumulative scribbles round 1",
            "Prediction round 1",
        ),
        native_height=image.shape[0],
    )
    template = output_dir / "60_scribbles3_cumulative_template.png"
    Image.fromarray(overlay1).save(template)
    return review, template


def save_iterative_overview(
    path: Path,
    overlay1: np.ndarray,
    overlay2: np.ndarray,
    source: RoundPrediction,
    round1: RoundPrediction,
    round2: RoundPrediction,
    ground_truth: np.ndarray,
) -> Path:
    return _save_panels(
        path,
        (
            _rgb_mask(source.prediction),
            overlay1[..., :3],
            _rgb_mask(round1.prediction),
            overlay2[..., :3],
            _rgb_mask(round2.prediction),
            _rgb_mask(ground_truth),
        ),
        (
            "Source prediction",
            "Scribbles round 1",
            "Prediction round 1",
            "Cumulative scribbles round 2",
            "Prediction round 2",
            "Ground truth",
        ),
        native_height=ground_truth.shape[0],
    )


def _save_prediction(output_dir: Path, result: RoundPrediction) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_outputs(output_dir, result.probability, result.prediction)


def _save_history(path: Path, adaptation: AdaptationResult) -> None:
    fields = (
        "step",
        "tile_index",
        "labeled_pixels",
        "total_loss",
        "scribble_loss",
        "consistency_loss",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: getattr(step, field) for field in fields}
            for step in adaptation.history
        )


def _round_metrics(
    ground_truth: np.ndarray,
    result: RoundPrediction,
) -> dict[str, object]:
    return {
        **segmentation_metrics(ground_truth, result.prediction),
        "adaptation_time": (
            0.0 if result.adaptation is None else result.adaptation.adaptation_time
        ),
        "inference_time": result.inference_time,
    }


def _save_final_results(
    args: argparse.Namespace,
    image: np.ndarray,
    overlay1: np.ndarray,
    overlay2: np.ndarray,
    scribbles1: np.ndarray,
    scribbles2: np.ndarray,
    results: dict[str, RoundPrediction],
) -> dict[str, dict[str, object]]:
    ground_truth = load_ground_truth(args.mask, image.shape)
    metrics_by_round = {
        name: _round_metrics(ground_truth, result) for name, result in results.items()
    }
    for name, result in results.items():
        output_dir = args.output_dir / name
        _save_prediction(output_dir, result)
        metrics = metrics_by_round[name]
        with (output_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
        if result.adaptation is not None:
            _save_history(output_dir / "training.csv", result.adaptation)

    comparison_path = args.output_dir / "comparison.csv"
    metric_names = (
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
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", *results])
        writer.writeheader()
        writer.writerows(
            {
                "metric": metric,
                **{name: metrics_by_round[name][metric] for name in results},
            }
            for metric in metric_names
        )
    save_iterative_overview(
        args.output_dir / "iterative_overview.png",
        overlay1,
        overlay2,
        results["source"],
        results["round_1"],
        results["round_2"],
        ground_truth,
    )
    metadata = {
        "round2_scribble_format": "cumulative",
        "round1_labeled_pixels": int(np.count_nonzero(scribbles1)),
        "round1_pore_labeled_pixels": int(np.count_nonzero(scribbles1 == PORE)),
        "round1_solid_labeled_pixels": int(np.count_nonzero(scribbles1 == SOLID)),
        "round1_labeled_fraction": float(
            np.count_nonzero(scribbles1) / scribbles1.size
        ),
        "round2_labeled_pixels": int(np.count_nonzero(scribbles2)),
        "round2_pore_labeled_pixels": int(np.count_nonzero(scribbles2 == PORE)),
        "round2_solid_labeled_pixels": int(np.count_nonzero(scribbles2 == SOLID)),
        "round2_labeled_fraction": float(
            np.count_nonzero(scribbles2) / scribbles2.size
        ),
        "new_round2_labeled_pixels": int(
            np.count_nonzero((scribbles1 == 0) & (scribbles2 != 0))
        ),
        "ground_truth_used_for": "final evaluation only",
    }
    with (args.output_dir / "iteration_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
    return metrics_by_round


def run(args: argparse.Namespace) -> dict[str, dict[str, object]] | None:
    config = INTERACTIVE_ADAPTATION_BASELINE
    image = load_grayscale_image(args.current)
    overlay1 = load_overlay(args.round1_scribbles)
    scribbles1 = extract_overlay_scribbles(overlay1, expected_shape=image.shape)
    frozen = load_frozen_fibnet(args.checkpoint, device=args.device)
    if frozen.feature_mode != "stack_relief" or frozen.model_arch != "resunet":
        raise ValueError("Iterative adaptation requires resunet/stack_relief.")
    if frozen.image_size != MANUSCRIPT_TILE_SIZE:
        raise ValueError(
            "Checkpoint patch size differs from the locked interactive tile size: "
            f"{frozen.image_size} != {MANUSCRIPT_TILE_SIZE}."
        )

    source_probability, source_time = _predict_native(
        frozen.model,
        args.current,
        args.previous,
        args.next,
        frozen.feature_mode,
        args.device,
    )
    source = RoundPrediction(
        source_probability,
        source_probability >= config.threshold,
        None,
        source_time,
    )
    round1_adaptation = _adapt_round(frozen, args, scribbles1)
    round1_probability, round1_time = _predict_native(
        frozen.model,
        args.current,
        args.previous,
        args.next,
        frozen.feature_mode,
        args.device,
    )
    round1 = RoundPrediction(
        round1_probability,
        round1_probability >= config.threshold,
        round1_adaptation,
        round1_time,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_prediction(args.output_dir / "source", source)
    _save_prediction(args.output_dir / "round_1", round1)
    Image.fromarray(scribbles1).save(args.output_dir / "round_1" / "scribbles.png")
    Image.fromarray(overlay1).save(args.output_dir / "round_1" / "overlay.png")
    _save_history(args.output_dir / "round_1" / "training.csv", round1_adaptation)
    review, template = save_round1_review(
        args.output_dir, image, overlay1, source, round1
    )
    session_metadata = {
        "status": "awaiting_round2_scribbles",
        "round2_scribble_format": "cumulative",
        "checkpoint": str(frozen.checkpoint),
        "trainable": "last decoder block + segmentation head",
        "lambda_consistency": config.lambda_consistency,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "steps_per_round": config.steps,
        "threshold": config.threshold,
        "tile_size": MANUSCRIPT_TILE_SIZE,
        "overlap": MANUSCRIPT_OVERLAP,
        "round1_labeled_pixels": int(np.count_nonzero(scribbles1)),
        "ground_truth_loaded": False,
    }
    with (args.output_dir / "session_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(session_metadata, handle, indent=2)
    if not args.round2_scribbles.exists():
        print(f"Round 1 ready for manual review: {review}")
        print(f"Add corrections to this cumulative template: {template}")
        print(f"Save the edited overlay as: {args.round2_scribbles}")
        return None

    overlay2 = load_overlay(args.round2_scribbles)
    scribbles2 = extract_overlay_scribbles(overlay2, expected_shape=image.shape)
    scribbles2 = validate_cumulative_scribbles(scribbles1, scribbles2)
    round2_adaptation = _adapt_round(frozen, args, scribbles2)
    round2_probability, round2_time = _predict_native(
        frozen.model,
        args.current,
        args.previous,
        args.next,
        frozen.feature_mode,
        args.device,
    )
    round2 = RoundPrediction(
        round2_probability,
        round2_probability >= config.threshold,
        round2_adaptation,
        round2_time,
    )
    (args.output_dir / "round_2").mkdir(parents=True, exist_ok=True)
    Image.fromarray(scribbles2).save(args.output_dir / "round_2" / "scribbles.png")
    Image.fromarray(overlay2).save(args.output_dir / "round_2" / "overlay.png")
    metrics = _save_final_results(
        args,
        image,
        overlay1,
        overlay2,
        scribbles1,
        scribbles2,
        {"source": source, "round_1": round1, "round_2": round2},
    )
    session_metadata["status"] = "complete"
    session_metadata["round2_labeled_pixels"] = int(np.count_nonzero(scribbles2))
    session_metadata["ground_truth_loaded"] = True
    with (args.output_dir / "session_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(session_metadata, handle, indent=2)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=Path("target/59.tiff"))
    parser.add_argument("--current", type=Path, default=Path("target/60.tiff"))
    parser.add_argument("--next", type=Path, default=Path("target/61.tiff"))
    parser.add_argument(
        "--round1-scribbles", type=Path, default=Path("60_scribbles2.png")
    )
    parser.add_argument(
        "--round2-scribbles", type=Path, default=Path("60_scribbles3.png")
    )
    parser.add_argument("--mask", type=Path, default=Path("ideal/60.tiff"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("weights/fibnet_source_v0.1.pt")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/iterative_scribble_refinement_60"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run(args)
    if metrics is None:
        return
    for name, values in metrics.items():
        print(
            f"{name}: Dice={values['dice']:.4f} IoU={values['iou']:.4f} "
            f"precision={values['precision']:.4f} recall={values['recall']:.4f} "
            f"accuracy={values['accuracy']:.4f} "
            f"pore_fraction={values['pore_fraction_prediction']:.4f} "
            f"pore_fraction_error={values['absolute_pore_fraction_error']:.4f}"
        )


if __name__ == "__main__":
    main()
