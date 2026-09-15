"""Compare single- and multi-slice sparse adaptation on a FIB-SEM stack.

Ground-truth pixels are loaded only after every regime has finished adaptation
and native tiled inference. Manual-mask filenames are used solely to identify
the evaluation slices before that boundary.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image, ImageDraw

from fibnet.inference import IMAGE_EXTENSIONS, predict_tiled
from fibnet.interactive import (
    MANUSCRIPT_OVERLAP,
    MANUSCRIPT_TILE_SIZE,
    PORE,
    SOLID,
    ScribbleSliceTrainingTiles,
    adapt_from_multislice_scribbles,
    load_frozen_fibnet,
    prepare_scribble_training_tiles,
)
from fibnet.interactive.io import load_grayscale_image
from fibnet.probability_io import load_probability_map, save_probability_map
from scripts.benchmark_rf_scribbles import load_ground_truth, segmentation_metrics
from scripts.evaluate_rf_overlay import extract_overlay_scribbles, load_overlay

TRAINING_SLICE_IDS = ("46", "60", "86")
THRESHOLD = 0.5
LEARNING_RATE = 3e-5
FREEZE_POLICY = "full_decoder"
LAMBDA_CONSISTENCY = 0.0
RANDOM_STATE = 42

RESULT_FIELDS = (
    "regime",
    "slice",
    "seen_unseen",
    "dice",
    "iou",
    "precision",
    "recall",
    "accuracy",
    "gt_pore_fraction",
    "predicted_pore_fraction",
    "absolute_pore_fraction_error",
    "inference_time",
)
SUMMARY_FIELDS = (
    "regime",
    "number_training_slices",
    "labeled_pixels",
    "optimization_steps",
    "adaptation_time",
    "mean_seen_dice",
    "mean_unseen_dice",
    "mean_all_dice",
    "mean_seen_iou",
    "mean_unseen_iou",
    "mean_seen_pore_fraction_error",
    "mean_unseen_pore_fraction_error",
)
SCRIBBLE_FIELDS = (
    "slice",
    "image_path",
    "previous_path",
    "next_path",
    "scribble_path",
    "pore_labeled_pixels",
    "solid_labeled_pixels",
    "total_labeled_pixels",
    "labeled_fraction",
    "pore_scribble_purity",
    "solid_scribble_purity",
    "overall_scribble_purity",
)


@dataclass(frozen=True)
class StackContext:
    """One stack image and its production stack-relief neighbours."""

    slice_id: str
    image_path: Path
    previous_path: Path
    next_path: Path


@dataclass(frozen=True)
class ScribbleInput:
    """Adaptation-only paths; deliberately contains no ground-truth path."""

    context: StackContext
    scribble_path: Path


@dataclass(frozen=True)
class EvaluationInput:
    """Evaluation-only image context and manual mask path."""

    context: StackContext
    mask_path: Path


@dataclass(frozen=True)
class RegimeInfo:
    name: str
    training_slice_ids: tuple[str, ...]
    labeled_pixels: int
    optimization_steps: int
    adaptation_time: float


def _slice_sort_key(slice_id: str) -> tuple[int, int | str]:
    return (0, int(slice_id)) if slice_id.isdigit() else (1, slice_id.lower())


def _unique_paths_by_stem(directory: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.lower()
        if key in paths:
            raise ValueError(
                f"Duplicate image stem {path.stem!r} in {directory}: "
                f"{paths[key].name}, {path.name}."
            )
        paths[key] = path
    return paths


def discover_stack_contexts(target_dir: Path) -> dict[str, StackContext]:
    """Resolve every target image and its neighbours without extension guesses."""
    paths_by_stem = _unique_paths_by_stem(target_dir)
    if not paths_by_stem:
        raise FileNotFoundError(f"No target images found in {target_dir}.")
    ordered_ids = sorted(paths_by_stem, key=_slice_sort_key)
    contexts = {}
    for index, slice_id in enumerate(ordered_ids):
        previous_id = ordered_ids[index - 1] if index else slice_id
        next_id = ordered_ids[index + 1] if index + 1 < len(ordered_ids) else slice_id
        contexts[slice_id] = StackContext(
            slice_id=slice_id,
            image_path=paths_by_stem[slice_id],
            previous_path=paths_by_stem[previous_id],
            next_path=paths_by_stem[next_id],
        )
    return contexts


def discover_evaluation_inputs(
    contexts: Mapping[str, StackContext], ideal_dir: Path
) -> tuple[EvaluationInput, ...]:
    """Find all manual masks with matching targets, using filenames only."""
    masks_by_stem = _unique_paths_by_stem(ideal_dir)
    shared = sorted(set(contexts) & set(masks_by_stem), key=_slice_sort_key)
    if not shared:
        raise FileNotFoundError(
            f"No manual masks in {ideal_dir} have matching images in the target stack."
        )
    return tuple(
        EvaluationInput(context=contexts[slice_id], mask_path=masks_by_stem[slice_id])
        for slice_id in shared
    )


def resolve_scribble_inputs(
    contexts: Mapping[str, StackContext],
    scribble_dir: Path,
    slice_ids: Sequence[str] = TRAINING_SLICE_IDS,
) -> tuple[ScribbleInput, ...]:
    """Associate each requested overlay with the exact image/context triplet."""
    resolved = []
    for slice_id in slice_ids:
        normalized = slice_id.lower()
        if normalized not in contexts:
            raise FileNotFoundError(
                f"Target image for scribble slice {slice_id!r} was not found."
            )
        scribble_path = scribble_dir / f"{slice_id}_scribbles.png"
        if not scribble_path.is_file():
            raise FileNotFoundError(f"Missing scribble overlay: {scribble_path}")
        resolved.append(
            ScribbleInput(context=contexts[normalized], scribble_path=scribble_path)
        )
    return tuple(resolved)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _predict_native(
    model: torch.nn.Module,
    context: StackContext,
    feature_mode: str,
    device: str,
) -> tuple[np.ndarray, float]:
    started = perf_counter()
    model.eval()
    with torch.no_grad():
        probability = predict_tiled(
            model,
            context.image_path,
            MANUSCRIPT_TILE_SIZE,
            MANUSCRIPT_OVERLAP,
            feature_mode,
            device,
            previous_path=context.previous_path,
            next_path=context.next_path,
        ).astype(np.float32, copy=False)
    return probability, perf_counter() - started


def _regime_dir(output_dir: Path, regime: str) -> Path:
    return output_dir / "regimes" / regime


def _save_prediction(
    output_dir: Path,
    regime: str,
    slice_id: str,
    probability: np.ndarray,
) -> None:
    regime_dir = _regime_dir(output_dir, regime)
    save_probability_map(regime_dir / "probabilities" / f"{slice_id}.npy", probability)
    segmentation_path = regime_dir / "segmentations" / f"{slice_id}.png"
    segmentation_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(probability >= THRESHOLD, 255, 0).astype(np.uint8)).save(
        segmentation_path
    )


def predict_regime(
    output_dir: Path,
    regime: str,
    model: torch.nn.Module,
    evaluations: Sequence[EvaluationInput],
    feature_mode: str,
    device: str,
) -> dict[str, float]:
    """Run production-aligned inference and persist only evaluation slices."""
    timings = {}
    for index, evaluation in enumerate(evaluations, start=1):
        slice_id = evaluation.context.slice_id
        probability, elapsed = _predict_native(
            model, evaluation.context, feature_mode, device
        )
        _save_prediction(output_dir, regime, slice_id, probability)
        timings[slice_id] = elapsed
        print(
            f"  {regime}: inference {index}/{len(evaluations)} "
            f"slice={slice_id} time={elapsed:.2f}s",
            flush=True,
        )
        del probability
    return timings


def _save_history(path: Path, histories: Sequence[object]) -> None:
    rows = []
    global_step = 0
    for phase, adaptation in enumerate(histories, start=1):
        for step in adaptation.history:
            global_step += 1
            rows.append(
                {
                    "global_step": global_step,
                    "phase": phase,
                    "phase_step": step.step,
                    "slice": step.slice_id,
                    "tile_index": step.tile_index,
                    "labeled_pixels": step.labeled_pixels,
                    "total_loss": step.total_loss,
                    "scribble_loss": step.scribble_loss,
                    "consistency_loss": step.consistency_loss,
                    "weight_anchor_loss": step.weight_anchor_loss,
                    "weighted_weight_anchor_loss": step.weighted_weight_anchor_loss,
                    "functional_anchor_loss": step.functional_anchor_loss,
                    "weighted_functional_anchor_loss": (
                        step.weighted_functional_anchor_loss
                    ),
                    "functional_anchor_slice": step.functional_anchor_slice_id,
                    "functional_confident_pixels": step.functional_confident_pixels,
                    "functional_confident_fraction": (
                        step.functional_confident_fraction
                    ),
                    "functional_active_pixels": step.functional_active_pixels,
                    "functional_active_fraction": step.functional_active_fraction,
                }
            )
    _write_csv(
        path,
        (
            "global_step",
            "phase",
            "phase_step",
            "slice",
            "tile_index",
            "labeled_pixels",
            "total_loss",
            "scribble_loss",
            "consistency_loss",
            "weight_anchor_loss",
            "weighted_weight_anchor_loss",
            "functional_anchor_loss",
            "weighted_functional_anchor_loss",
            "functional_anchor_slice",
            "functional_confident_pixels",
            "functional_confident_fraction",
            "functional_active_pixels",
            "functional_active_fraction",
        ),
        rows,
    )


def _load_scribbles_and_tiles(
    frozen,
    inputs: Sequence[ScribbleInput],
) -> tuple[
    dict[str, np.ndarray],
    tuple[ScribbleSliceTrainingTiles, ...],
    list[dict[str, object]],
]:
    scribbles_by_slice = {}
    tile_groups = []
    stats = []
    for item in inputs:
        context = item.context
        image = load_grayscale_image(context.image_path)
        scribbles = extract_overlay_scribbles(
            load_overlay(item.scribble_path), expected_shape=image.shape
        )
        tiles = prepare_scribble_training_tiles(
            frozen,
            context.previous_path,
            context.image_path,
            context.next_path,
            scribbles,
            tile_size=MANUSCRIPT_TILE_SIZE,
            overlap=MANUSCRIPT_OVERLAP,
            compute_source_probability=False,
        )
        pore_pixels = int(np.count_nonzero(scribbles == PORE))
        solid_pixels = int(np.count_nonzero(scribbles == SOLID))
        total_pixels = pore_pixels + solid_pixels
        stats.append(
            {
                "slice": context.slice_id,
                "image_path": str(context.image_path),
                "previous_path": str(context.previous_path),
                "next_path": str(context.next_path),
                "scribble_path": str(item.scribble_path),
                "pore_labeled_pixels": pore_pixels,
                "solid_labeled_pixels": solid_pixels,
                "total_labeled_pixels": total_pixels,
                "labeled_fraction": total_pixels / scribbles.size,
                "pore_scribble_purity": "",
                "solid_scribble_purity": "",
                "overall_scribble_purity": "",
            }
        )
        print(
            f"scribbles slice={context.slice_id}: pore={pore_pixels}, "
            f"solid={solid_pixels}, total={total_pixels}, "
            f"fraction={total_pixels / scribbles.size:.8f}, tiles={len(tiles)}",
            flush=True,
        )
        scribbles_by_slice[context.slice_id] = scribbles
        tile_groups.append(
            ScribbleSliceTrainingTiles(slice_id=context.slice_id, tiles=tiles)
        )
    return scribbles_by_slice, tuple(tile_groups), stats


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float | str:
    if not rows:
        return ""
    return float(np.mean([float(row[field]) for row in rows]))


def classify_seen(slice_id: str, training_slice_ids: Sequence[str]) -> str:
    return "seen" if slice_id in set(training_slice_ids) else "unseen"


def _evaluate_predictions(
    output_dir: Path,
    evaluations: Sequence[EvaluationInput],
    regimes: Sequence[RegimeInfo],
    inference_times: Mapping[str, Mapping[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    """Cross the GT boundary: load masks only here, after all predictions exist."""
    ground_truth_by_slice = {}
    for evaluation in evaluations:
        context = evaluation.context
        with Image.open(context.image_path) as image:
            shape = (image.height, image.width)
        ground_truth_by_slice[context.slice_id] = load_ground_truth(
            evaluation.mask_path, shape
        )

    result_rows = []
    summary_rows = []
    for regime in regimes:
        regime_rows = []
        for evaluation in evaluations:
            slice_id = evaluation.context.slice_id
            probability = load_probability_map(
                _regime_dir(output_dir, regime.name)
                / "probabilities"
                / f"{slice_id}.npy"
            )
            metrics = segmentation_metrics(
                ground_truth_by_slice[slice_id], probability >= THRESHOLD
            )
            row = {
                "regime": regime.name,
                "slice": slice_id,
                "seen_unseen": classify_seen(slice_id, regime.training_slice_ids),
                **metrics,
                "gt_pore_fraction": metrics["pore_fraction_gt"],
                "predicted_pore_fraction": metrics["pore_fraction_prediction"],
                "inference_time": inference_times[regime.name][slice_id],
            }
            row.pop("pore_fraction_gt")
            row.pop("pore_fraction_prediction")
            result_rows.append(row)
            regime_rows.append(row)
            del probability
        seen = [row for row in regime_rows if row["seen_unseen"] == "seen"]
        unseen = [row for row in regime_rows if row["seen_unseen"] == "unseen"]
        summary_rows.append(
            {
                "regime": regime.name,
                "number_training_slices": len(regime.training_slice_ids),
                "labeled_pixels": regime.labeled_pixels,
                "optimization_steps": regime.optimization_steps,
                "adaptation_time": regime.adaptation_time,
                "mean_seen_dice": _mean(seen, "dice"),
                "mean_unseen_dice": _mean(unseen, "dice"),
                "mean_all_dice": _mean(regime_rows, "dice"),
                "mean_seen_iou": _mean(seen, "iou"),
                "mean_unseen_iou": _mean(unseen, "iou"),
                "mean_seen_pore_fraction_error": _mean(
                    seen, "absolute_pore_fraction_error"
                ),
                "mean_unseen_pore_fraction_error": _mean(
                    unseen, "absolute_pore_fraction_error"
                ),
            }
        )
    return result_rows, summary_rows, ground_truth_by_slice


def rank_multislice(summary_rows: Sequence[Mapping[str, object]]) -> list[str]:
    candidates = [
        row
        for row in summary_rows
        if int(row["number_training_slices"]) > 1 and row["mean_unseen_dice"] != ""
    ]
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["mean_unseen_dice"]),
            -float(row["mean_all_dice"]),
            float(row["mean_unseen_pore_fraction_error"]),
            str(row["regime"]),
        ),
    )
    return [str(row["regime"]) for row in ranked]


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


def _mask_preview(mask: np.ndarray) -> np.ndarray:
    return np.repeat((np.asarray(mask, dtype=np.uint8) * 255)[..., None], 3, axis=2)


def _error_preview(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    errors = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    errors[ground_truth & prediction] = (210, 210, 210)
    errors[~ground_truth & prediction] = (255, 64, 64)
    errors[ground_truth & ~prediction] = (64, 128, 255)
    return errors


def _scribble_preview(image: np.ndarray, scribbles: np.ndarray | None) -> np.ndarray:
    preview = _gray_preview(image)
    if scribbles is None:
        return preview
    preview = np.rint(preview.astype(np.float32) * 0.55).astype(np.uint8)
    preview[scribbles == PORE] = (0, 220, 0)
    preview[scribbles == SOLID] = (240, 30, 30)
    return preview


def save_overviews(
    output_dir: Path,
    best_regime: str,
    evaluations: Sequence[EvaluationInput],
    scribbles_by_slice: Mapping[str, np.ndarray],
    ground_truth_by_slice: Mapping[str, np.ndarray],
) -> None:
    overview_dir = output_dir / "best_multislice_overviews"
    overview_dir.mkdir(parents=True, exist_ok=True)
    for evaluation in evaluations:
        context = evaluation.context
        slice_id = context.slice_id
        image = load_grayscale_image(context.image_path)
        ground_truth = ground_truth_by_slice[slice_id]
        probability = load_probability_map(
            _regime_dir(output_dir, best_regime) / "probabilities" / f"{slice_id}.npy"
        )
        prediction = probability >= THRESHOLD
        panels = (
            _gray_preview(image),
            _scribble_preview(image, scribbles_by_slice.get(slice_id)),
            _mask_preview(prediction),
            _mask_preview(ground_truth),
            _error_preview(ground_truth, prediction),
        )
        titles = (
            "Original",
            "Scribbles" if slice_id in scribbles_by_slice else "Unseen (no scribbles)",
            f"Prediction: {best_regime}",
            "Ground truth",
            "Errors (FP red, FN blue)",
        )
        scale = min(1.0, 620 / image.shape[0])
        panel_size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
        header = 34
        canvas = Image.new(
            "RGB", (panel_size[0] * len(panels), panel_size[1] + header), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
            left = index * panel_size[0]
            resized = Image.fromarray(panel).resize(
                panel_size, Image.Resampling.NEAREST
            )
            canvas.paste(resized, (left, header))
            draw.text((left + 5, 10), title, fill="black")
        canvas.save(overview_dir / f"{slice_id}.png")


def save_dice_plot(
    path: Path,
    result_rows: Sequence[Mapping[str, object]],
    best_regime: str,
) -> None:
    regimes = ("source", "single_60", best_regime)
    colors = ((80, 80, 80), (35, 105, 210), (20, 150, 75))
    slice_ids = sorted({str(row["slice"]) for row in result_rows}, key=_slice_sort_key)
    x_values = [
        int(value) if value.isdigit() else index
        for index, value in enumerate(slice_ids)
    ]
    selected = {
        (str(row["regime"]), str(row["slice"])): float(row["dice"])
        for row in result_rows
        if row["regime"] in regimes
    }
    y_values = list(selected.values())
    y_low = max(0.0, min(y_values) - 0.03)
    y_high = min(1.0, max(y_values) + 0.03)
    if y_high <= y_low:
        y_low, y_high = 0.0, 1.0
    width, height = 1200, 720
    left, right, top, bottom = 90, 1140, 60, 620

    def xy(slice_value: int, dice: float) -> tuple[int, int]:
        x_min, x_max = min(x_values), max(x_values)
        x = (
            left
            if x_max == x_min
            else left + (slice_value - x_min) / (x_max - x_min) * (right - left)
        )
        y = bottom - (dice - y_low) / (y_high - y_low) * (bottom - top)
        return round(x), round(y)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 20), "Dice vs slice index", fill="black")
    draw.text((left, bottom + 55), "Slice index", fill="black")
    draw.text((12, top), "Dice", fill="black")
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    for tick in np.linspace(y_low, y_high, 6):
        y = round(bottom - (tick - y_low) / (y_high - y_low) * (bottom - top))
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((35, y - 7), f"{tick:.3f}", fill="black")
    for slice_id, slice_value in zip(slice_ids, x_values, strict=True):
        x, _ = xy(slice_value, y_low)
        draw.text((x - 10, bottom + 12), slice_id, fill="black")
    for training_id in TRAINING_SLICE_IDS:
        if training_id not in slice_ids:
            continue
        x, _ = xy(int(training_id), y_low)
        for y in range(top, bottom, 12):
            draw.line((x, y, x, min(y + 6, bottom)), fill=(185, 185, 185), width=2)
    for regime, color in zip(regimes, colors, strict=True):
        points = [
            xy(value, selected[(regime, slice_id)])
            for value, slice_id in zip(x_values, slice_ids, strict=True)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
    for index, (regime, color) in enumerate(zip(regimes, colors, strict=True)):
        y = 70 + index * 25
        draw.line((780, y + 6, 815, y + 6), fill=color, width=4)
        draw.text((825, y), regime, fill="black")
    draw.text((780, 150), "Dashed: training scribble slices", fill=(100, 100, 100))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _add_scribble_purity(
    stats: list[dict[str, object]],
    scribbles_by_slice: Mapping[str, np.ndarray],
    ground_truth_by_slice: Mapping[str, np.ndarray],
) -> None:
    for row in stats:
        slice_id = str(row["slice"])
        scribbles = scribbles_by_slice[slice_id]
        ground_truth = ground_truth_by_slice[slice_id]
        pore = scribbles == PORE
        solid = scribbles == SOLID
        correct = (pore & ground_truth) | (solid & ~ground_truth)
        row["pore_scribble_purity"] = float(ground_truth[pore].mean())
        row["solid_scribble_purity"] = float((~ground_truth)[solid].mean())
        row["overall_scribble_purity"] = float(
            np.count_nonzero(correct) / np.count_nonzero(pore | solid)
        )


def _configuration_metadata() -> dict[str, object]:
    return {
        "freeze_policy": FREEZE_POLICY,
        "encoder": "frozen",
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam; one optimizer for each joint session, reset per sequential slice",
        "lambda_consistency": LAMBDA_CONSISTENCY,
        "threshold": THRESHOLD,
        "tile_size": MANUSCRIPT_TILE_SIZE,
        "overlap": MANUSCRIPT_OVERLAP,
        "preprocessing": "stack_relief",
        "random_state": RANDOM_STATE,
        "joint_step_budgets": [25, 75],
        "sequential_steps_per_slice": 25,
    }


def run_sequential_adaptation(
    frozen,
    tile_groups: Sequence[ScribbleSliceTrainingTiles],
    *,
    steps_per_slice: int = 25,
    learning_rate: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
) -> tuple[list[object], list[dict[str, str]]]:
    """Adapt ordered slices without resetting the shared model weights."""
    histories = []
    transitions = []
    for group in tile_groups:
        start_digest = state_dict_digest(frozen.model.state_dict())
        adaptation = adapt_from_multislice_scribbles(
            frozen,
            (group,),
            FREEZE_POLICY,
            LAMBDA_CONSISTENCY,
            steps=steps_per_slice,
            learning_rate=learning_rate,
            sampling_strategy="joint_pixel_uniform",
            random_state=random_state,
        )
        end_digest = state_dict_digest(frozen.model.state_dict())
        histories.append(adaptation)
        transitions.append(
            {
                "slice": group.slice_id,
                "start_state_sha256": start_digest,
                "end_state_sha256": end_digest,
            }
        )
    for previous, current in zip(transitions, transitions[1:], strict=False):
        if previous["end_state_sha256"] != current["start_state_sha256"]:
            raise RuntimeError("Sequential adaptation did not continue prior weights.")
    return histories, transitions


def run(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_digest_before = file_digest(args.checkpoint)
    contexts = discover_stack_contexts(args.target_dir)
    evaluations = discover_evaluation_inputs(contexts, args.ideal_dir)
    scribble_inputs = resolve_scribble_inputs(contexts, args.scribble_dir)
    print(
        "evaluation slices: "
        + ", ".join(evaluation.context.slice_id for evaluation in evaluations),
        flush=True,
    )

    frozen = load_frozen_fibnet(args.checkpoint, device=args.device)
    if frozen.feature_mode != "stack_relief" or frozen.model_arch != "resunet":
        raise ValueError(
            "Benchmark requires the production resunet/stack_relief checkpoint; "
            f"received {frozen.model_arch}/{frozen.feature_mode}."
        )
    if frozen.image_size != MANUSCRIPT_TILE_SIZE:
        raise ValueError(
            f"Checkpoint tile size {frozen.image_size} is not {MANUSCRIPT_TILE_SIZE}."
        )
    source_state = _clone_state(frozen.model)
    source_state_sha256 = state_dict_digest(source_state)

    # Required first stage: source inference on every evaluation slice.
    inference_times: dict[str, dict[str, float]] = {
        "source": predict_regime(
            args.output_dir,
            "source",
            frozen.model,
            evaluations,
            frozen.feature_mode,
            args.device,
        )
    }

    scribbles_by_slice, tile_groups, scribble_stats = _load_scribbles_and_tiles(
        frozen, scribble_inputs
    )
    groups_by_slice = {item.slice_id: item for item in tile_groups}
    labeled_by_slice = {
        str(row["slice"]): int(row["total_labeled_pixels"]) for row in scribble_stats
    }
    all_labeled_pixels = sum(labeled_by_slice.values())
    regime_infos = [RegimeInfo("source", (), 0, 0, 0.0)]

    def infer_adapted(regime: str) -> None:
        inference_times[regime] = predict_regime(
            args.output_dir,
            regime,
            frozen.model,
            evaluations,
            frozen.feature_mode,
            args.device,
        )
        if frozen.device.type == "cuda":
            torch.cuda.empty_cache()

    frozen.model.load_state_dict(source_state)
    single = adapt_from_multislice_scribbles(
        frozen,
        (groups_by_slice["60"],),
        FREEZE_POLICY,
        LAMBDA_CONSISTENCY,
        steps=25,
        learning_rate=LEARNING_RATE,
        sampling_strategy="joint_pixel_uniform",
        random_state=RANDOM_STATE,
    )
    _save_history(_regime_dir(args.output_dir, "single_60") / "history.csv", (single,))
    infer_adapted("single_60")
    regime_infos.append(
        RegimeInfo(
            "single_60", ("60",), labeled_by_slice["60"], 25, single.adaptation_time
        )
    )

    for sampling_strategy in ("joint_pixel_uniform", "joint_slice_balanced"):
        for steps in (25, 75):
            regime = f"{sampling_strategy}_{steps}"
            frozen.model.load_state_dict(source_state)
            adaptation = adapt_from_multislice_scribbles(
                frozen,
                tile_groups,
                FREEZE_POLICY,
                LAMBDA_CONSISTENCY,
                steps=steps,
                learning_rate=LEARNING_RATE,
                sampling_strategy=sampling_strategy,
                random_state=RANDOM_STATE,
            )
            _save_history(
                _regime_dir(args.output_dir, regime) / "history.csv", (adaptation,)
            )
            infer_adapted(regime)
            regime_infos.append(
                RegimeInfo(
                    regime,
                    TRAINING_SLICE_IDS,
                    all_labeled_pixels,
                    steps,
                    adaptation.adaptation_time,
                )
            )

    frozen.model.load_state_dict(source_state)
    sequential_histories, sequential_transitions = run_sequential_adaptation(
        frozen, tile_groups
    )
    sequential_name = "sequential_46_60_86"
    _save_history(
        _regime_dir(args.output_dir, sequential_name) / "history.csv",
        sequential_histories,
    )
    infer_adapted(sequential_name)
    regime_infos.append(
        RegimeInfo(
            sequential_name,
            TRAINING_SLICE_IDS,
            all_labeled_pixels,
            75,
            sum(item.adaptation_time for item in sequential_histories),
        )
    )

    # No ground-truth array has been loaded before this point.
    result_rows, summary_rows, ground_truth_by_slice = _evaluate_predictions(
        args.output_dir, evaluations, regime_infos, inference_times
    )
    _add_scribble_purity(scribble_stats, scribbles_by_slice, ground_truth_by_slice)
    _write_csv(args.output_dir / "results_per_slice.csv", RESULT_FIELDS, result_rows)
    _write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(args.output_dir / "scribble_stats.csv", SCRIBBLE_FIELDS, scribble_stats)

    ranking = rank_multislice(summary_rows)
    if not ranking:
        raise RuntimeError("No multi-slice regime could be ranked on unseen slices.")
    best_regime = ranking[0]
    save_dice_plot(args.output_dir / "dice_vs_slice.png", result_rows, best_regime)
    save_overviews(
        args.output_dir,
        best_regime,
        evaluations,
        scribbles_by_slice,
        ground_truth_by_slice,
    )

    checkpoint_digest_after = file_digest(args.checkpoint)
    if checkpoint_digest_after != checkpoint_digest_before:
        raise RuntimeError("Source checkpoint changed during the benchmark.")
    history_counts = {}
    for info in regime_infos:
        history_path = _regime_dir(args.output_dir, info.name) / "history.csv"
        if history_path.exists():
            with history_path.open(encoding="utf-8", newline="") as handle:
                counts = Counter(row["slice"] for row in csv.DictReader(handle))
            history_counts[info.name] = dict(counts)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_before": checkpoint_digest_before,
        "checkpoint_sha256_after": checkpoint_digest_after,
        "source_checkpoint_unchanged": True,
        "source_state_sha256": source_state_sha256,
        "evaluation_slices": [item.context.slice_id for item in evaluations],
        "training_slices": list(TRAINING_SLICE_IDS),
        "best_multislice_regime": best_regime,
        "multislice_ranking": ranking,
        "ranking_rule": (
            "mean unseen Dice desc, mean all Dice desc, "
            "mean unseen absolute pore-fraction error asc"
        ),
        "ground_truth_used_for": "metrics, scribble purity, and overviews after all predictions",
        "adaptation_inputs_contain_ground_truth_paths": False,
        "history_slice_counts": history_counts,
        "sequential_state_transitions": sequential_transitions,
        **_configuration_metadata(),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    del source_state, tile_groups, scribbles_by_slice, ground_truth_by_slice
    gc.collect()
    return result_rows, summary_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=Path("target"))
    parser.add_argument("--ideal-dir", type=Path, default=Path("ideal"))
    parser.add_argument("--scribble-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("weights/fibnet_source_v0.1.pt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/multislice_adaptation")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _rows, summary = run(args)
    print("\nSummary:")
    for row in summary:
        seen = row["mean_seen_dice"]
        unseen = row["mean_unseen_dice"]
        print(
            f"{row['regime']}: seen={seen if seen == '' else f'{float(seen):.5f}'} "
            f"unseen={unseen if unseen == '' else f'{float(unseen):.5f}'} "
            f"all={float(row['mean_all_dice']):.5f} "
            f"steps={row['optimization_steps']} "
            f"adapt={float(row['adaptation_time']):.2f}s"
        )
    print(f"outputs={args.output_dir}")


if __name__ == "__main__":
    main()
