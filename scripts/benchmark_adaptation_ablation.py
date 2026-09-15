"""Ablate two-round sparse adaptation without unfreezing the FIB-NET encoder."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image, ImageDraw

from fibnet.inference import predict_tiled
from fibnet.interactive import (
    MANUSCRIPT_OVERLAP,
    MANUSCRIPT_TILE_SIZE,
    AdaptationMode,
    AdaptationResult,
    FrozenFibNet,
    ScribbleTrainingTile,
    adapt_from_scribbles,
    describe_freeze_policy,
    load_frozen_fibnet,
    prepare_scribble_training_tiles,
)
from fibnet.interactive.io import load_grayscale_image, save_probability_map
from scripts.benchmark_iterative_scribble_refinement import (
    validate_cumulative_scribbles,
)
from scripts.benchmark_rf_scribbles import (
    load_ground_truth,
    save_errors,
    segmentation_metrics,
)
from scripts.evaluate_rf_overlay import extract_overlay_scribbles, load_overlay

FREEZE_POLICIES: tuple[AdaptationMode, ...] = (
    "last_decoder_head",
    "last2_decoder_head",
    "full_decoder",
)
LEARNING_RATES = (3e-5, 1e-4, 3e-4)
STEP_COUNTS = (25, 50, 100)
RANDOM_STATE = 42
THRESHOLD = 0.5
LAMBDA_CONSISTENCY = 0.0

METRIC_FIELDS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "accuracy",
    "pore_fraction_gt",
    "pore_fraction_prediction",
    "absolute_pore_fraction_error",
)
RESULT_FIELDS = (
    "config_id",
    "freeze_policy",
    "learning_rate",
    "steps",
    "round",
    *METRIC_FIELDS,
    "adaptation_time",
    "inference_time",
    "cumulative_adaptation_time",
    "trainable_parameters",
    "frozen_parameters",
    "total_parameters",
    "trainable_module_names",
    "optimizer",
    "lambda_consistency",
    "threshold",
    "random_state",
)
SUMMARY_FIELDS = (
    "rank",
    "config_id",
    "freeze_policy",
    "learning_rate",
    "steps",
    "source_dice",
    "round1_dice",
    "round1_iou",
    "round1_precision",
    "round1_recall",
    "round1_accuracy",
    "round1_pore_fraction_prediction",
    "round1_absolute_pore_fraction_error",
    "round2_dice",
    "round2_iou",
    "round2_precision",
    "round2_recall",
    "round2_accuracy",
    "round2_pore_fraction_prediction",
    "round2_absolute_pore_fraction_error",
    "delta_dice_source_to_round1",
    "delta_dice_round1_to_round2",
    "delta_dice_source_to_round2",
    "round1_adaptation_time",
    "round2_adaptation_time",
    "total_adaptation_time",
    "round1_inference_time",
    "round2_inference_time",
    "trainable_parameters",
    "frozen_parameters",
    "total_parameters",
    "trainable_module_names",
    "unstable_or_suspicious",
    "suspicion_reasons",
)


@dataclass(frozen=True)
class RoundOutput:
    probability: np.ndarray
    adaptation: AdaptationResult
    inference_time: float


@dataclass(frozen=True)
class SequentialOutput:
    round1: RoundOutput
    round2: RoundOutput
    round1_state_digest: str
    round2_start_state_digest: str


def configurations() -> tuple[tuple[AdaptationMode, float, int], ...]:
    """Return the stable 3 x 3 x 3 ablation grid."""
    return tuple(
        (policy, learning_rate, steps)
        for policy in FREEZE_POLICIES
        for learning_rate in LEARNING_RATES
        for steps in STEP_COUNTS
    )


def _config_id(policy: str, learning_rate: float, steps: int) -> str:
    lr_slug = f"{learning_rate:.0e}".replace("-", "m")
    return f"{policy}__lr_{lr_slug}__steps_{steps:03d}"


def state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Hash model values for reset/continuation audit records."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def run_sequential_configuration(
    frozen: FrozenFibNet,
    source_state: Mapping[str, torch.Tensor],
    round1_tiles: tuple[ScribbleTrainingTile, ...],
    round2_tiles: tuple[ScribbleTrainingTile, ...],
    policy: AdaptationMode,
    learning_rate: float,
    steps: int,
    predict: Callable[[torch.nn.Module], tuple[np.ndarray, float]],
) -> SequentialOutput:
    """Reset once, then adapt round 2 directly from round-1 weights."""
    frozen.model.load_state_dict(source_state)
    round1_adaptation = adapt_from_scribbles(
        frozen,
        round1_tiles,
        policy,
        LAMBDA_CONSISTENCY,
        steps=steps,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
    )
    round1_probability, round1_inference_time = predict(frozen.model)
    round1_digest = state_dict_digest(frozen.model.state_dict())
    round2_start_digest = state_dict_digest(frozen.model.state_dict())
    round2_adaptation = adapt_from_scribbles(
        frozen,
        round2_tiles,
        policy,
        LAMBDA_CONSISTENCY,
        steps=steps,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
    )
    round2_probability, round2_inference_time = predict(frozen.model)
    return SequentialOutput(
        round1=RoundOutput(
            round1_probability,
            round1_adaptation,
            round1_inference_time,
        ),
        round2=RoundOutput(
            round2_probability,
            round2_adaptation,
            round2_inference_time,
        ),
        round1_state_digest=round1_digest,
        round2_start_state_digest=round2_start_digest,
    )


def _result_row(
    config_id: str,
    policy: AdaptationMode,
    learning_rate: float,
    steps: int,
    round_name: str,
    metrics: Mapping[str, float],
    adaptation: AdaptationResult | None,
    inference_time: float,
    cumulative_adaptation_time: float,
    *,
    total_parameters: int,
) -> dict[str, object]:
    return {
        "config_id": config_id,
        "freeze_policy": policy,
        "learning_rate": learning_rate,
        "steps": steps,
        "round": round_name,
        **{name: metrics[name] for name in METRIC_FIELDS},
        "adaptation_time": 0.0 if adaptation is None else adaptation.adaptation_time,
        "inference_time": inference_time,
        "cumulative_adaptation_time": cumulative_adaptation_time,
        "trainable_parameters": (
            0 if adaptation is None else adaptation.trainable_parameters
        ),
        "frozen_parameters": (
            total_parameters if adaptation is None else adaptation.frozen_parameters
        ),
        "total_parameters": total_parameters,
        "trainable_module_names": (
            "" if adaptation is None else ";".join(adaptation.trainable_module_names)
        ),
        "optimizer": "none" if adaptation is None else "Adam",
        "lambda_consistency": LAMBDA_CONSISTENCY,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
    }


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rank_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Rank deterministically by R2 Dice, IoU, pore error, then config ID."""
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row["round2_dice"]),
            -float(row["round2_iou"]),
            float(row["round2_absolute_pore_fraction_error"]),
            str(row["config_id"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _summary_row(
    rows: Sequence[Mapping[str, object]], baseline_total_time: float
) -> dict[str, object]:
    by_round = {str(row["round"]): row for row in rows}
    source = by_round["source"]
    round1 = by_round["round1"]
    round2 = by_round["round2"]
    reasons = []
    if float(round2["dice"]) < float(round1["dice"]):
        reasons.append("round2_dice_below_round1")
    if (
        abs(
            float(round2["pore_fraction_prediction"])
            - float(round1["pore_fraction_prediction"])
        )
        > 0.10
    ):
        reasons.append("pore_fraction_changed_over_0.10")
    if float(source["precision"]) - float(round2["precision"]) > 0.15:
        reasons.append("source_to_round2_precision_drop_over_0.15")
    total_adaptation_time = float(round1["adaptation_time"]) + float(
        round2["adaptation_time"]
    )
    if (
        baseline_total_time > 0
        and total_adaptation_time >= 2.0 * baseline_total_time
        and float(round2["dice"]) - float(source["dice"]) < 0.005
    ):
        reasons.append("under_0.005_dice_gain_at_2x_baseline_time")
    result: dict[str, object] = {
        "rank": 0,
        "config_id": round1["config_id"],
        "freeze_policy": round1["freeze_policy"],
        "learning_rate": round1["learning_rate"],
        "steps": round1["steps"],
        "source_dice": source["dice"],
        "delta_dice_source_to_round1": float(round1["dice"]) - float(source["dice"]),
        "delta_dice_round1_to_round2": float(round2["dice"]) - float(round1["dice"]),
        "delta_dice_source_to_round2": float(round2["dice"]) - float(source["dice"]),
        "round1_adaptation_time": round1["adaptation_time"],
        "round2_adaptation_time": round2["adaptation_time"],
        "total_adaptation_time": total_adaptation_time,
        "round1_inference_time": round1["inference_time"],
        "round2_inference_time": round2["inference_time"],
        "trainable_parameters": round1["trainable_parameters"],
        "frozen_parameters": round1["frozen_parameters"],
        "total_parameters": round1["total_parameters"],
        "trainable_module_names": round1["trainable_module_names"],
        "unstable_or_suspicious": bool(reasons),
        "suspicion_reasons": ";".join(reasons),
    }
    for prefix, row in (("round1", round1), ("round2", round2)):
        for metric in (
            "dice",
            "iou",
            "precision",
            "recall",
            "accuracy",
            "pore_fraction_prediction",
            "absolute_pore_fraction_error",
        ):
            result[f"{prefix}_{metric}"] = row[metric]
    return result


def _xy(value: float, low: float, high: float, top: int, bottom: int) -> int:
    if high <= low:
        return (top + bottom) // 2
    return round(bottom - (value - low) / (high - low) * (bottom - top))


def _line_plot(
    path: Path,
    title: str,
    x_labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    y_label: str,
) -> None:
    width, height = 1200, 720
    left, right, top, bottom = 90, 850, 65, 640
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    values = [value for _, line in series for value in line]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.12, 0.002)
    low -= padding
    high += padding
    draw.text((left, 20), title, fill="black")
    draw.text((12, top), y_label, fill="black")
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    for tick in range(6):
        value = low + tick * (high - low) / 5
        y = _xy(value, low, high, top, bottom)
        draw.line((left - 5, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((20, y - 7), f"{value:.3f}", fill="black")
    x_positions = np.linspace(left, right, len(x_labels)).round().astype(int)
    for x, label in zip(x_positions, x_labels, strict=True):
        draw.line((x, bottom, x, bottom + 5), fill="black", width=1)
        draw.text((x - 18, bottom + 10), label, fill="black")
    colors = (
        (0, 92, 230),
        (230, 82, 0),
        (0, 150, 80),
        (165, 60, 190),
        (205, 160, 0),
        (0, 165, 175),
        (110, 70, 40),
        (230, 80, 145),
        (90, 90, 90),
    )
    legend_x, legend_y = 885, 75
    for index, (label, line) in enumerate(series):
        color = colors[index % len(colors)]
        points = [
            (int(x), _xy(value, low, high, top, bottom))
            for x, value in zip(x_positions, line, strict=True)
        ]
        draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        y = legend_y + index * 28
        draw.line((legend_x, y + 6, legend_x + 24, y + 6), fill=color, width=3)
        draw.text((legend_x + 32, y), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _scatter_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    width, height = 1000, 720
    left, right, top, bottom = 90, 920, 60, 640
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    precision = [float(row["round2_precision"]) for row in rows]
    recall = [float(row["round2_recall"]) for row in rows]
    x_low, x_high = min(recall) - 0.01, max(recall) + 0.01
    y_low, y_high = min(precision) - 0.01, max(precision) + 0.01
    draw.text((left, 20), "Round-2 precision vs recall", fill="black")
    draw.text((left, bottom + 35), "Recall", fill="black")
    draw.text((12, top), "Precision", fill="black")
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    colors = {
        "last_decoder_head": (0, 92, 230),
        "last2_decoder_head": (230, 82, 0),
        "full_decoder": (0, 150, 80),
    }
    for row in rows:
        x = round(
            left
            + (float(row["round2_recall"]) - x_low) / (x_high - x_low) * (right - left)
        )
        y = _xy(float(row["round2_precision"]), y_low, y_high, top, bottom)
        color = colors[str(row["freeze_policy"])]
        radius = {25: 4, 50: 6, 100: 8}[int(row["steps"])]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    for index, (policy, color) in enumerate(colors.items()):
        y = 70 + 25 * index
        draw.ellipse((760, y, 772, y + 12), fill=color)
        draw.text((780, y), policy, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _bar_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: str(row["config_id"]))
    width, height = 1400, 720
    left, right, top, bottom = 90, 1340, 60, 620
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    high = max(float(row["round2_absolute_pore_fraction_error"]) for row in ordered)
    high = max(high * 1.1, 0.01)
    draw.text((left, 20), "Round-2 absolute pore-fraction error", fill="black")
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    colors = {
        "last_decoder_head": (0, 92, 230),
        "last2_decoder_head": (230, 82, 0),
        "full_decoder": (0, 150, 80),
    }
    bar_width = max(3, (right - left) // (len(ordered) * 2))
    positions = np.linspace(left + 15, right - 15, len(ordered)).round().astype(int)
    for x, row in zip(positions, ordered, strict=True):
        value = float(row["round2_absolute_pore_fraction_error"])
        y = _xy(value, 0.0, high, top, bottom)
        draw.rectangle(
            (x - bar_width, y, x + bar_width, bottom),
            fill=colors[str(row["freeze_policy"])],
        )
        draw.text((x - 10, bottom + 8), str(row["steps"]), fill="black")
    draw.text(
        (left, bottom + 38),
        "Bars ordered by policy/lr/steps; labels are steps",
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_plots(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    by_key = {
        (str(row["freeze_policy"]), float(row["learning_rate"]), int(row["steps"])): row
        for row in rows
    }
    step_series = []
    for policy in FREEZE_POLICIES:
        for learning_rate in LEARNING_RATES:
            step_series.append(
                (
                    f"{policy}, lr={learning_rate:g}",
                    [
                        float(by_key[(policy, learning_rate, steps)]["round2_dice"])
                        for steps in STEP_COUNTS
                    ],
                )
            )
    _line_plot(
        output_dir / "round2_dice_vs_steps.png",
        "Round-2 Dice vs optimization steps",
        [str(value) for value in STEP_COUNTS],
        step_series,
        "Dice",
    )
    lr_series = []
    for policy in FREEZE_POLICIES:
        for steps in STEP_COUNTS:
            lr_series.append(
                (
                    f"{policy}, steps={steps}",
                    [
                        float(by_key[(policy, learning_rate, steps)]["round2_dice"])
                        for learning_rate in LEARNING_RATES
                    ],
                )
            )
    _line_plot(
        output_dir / "round2_dice_vs_lr.png",
        "Round-2 Dice vs learning rate",
        [f"{value:g}" for value in LEARNING_RATES],
        lr_series,
        "Dice",
    )
    _scatter_plot(output_dir / "precision_recall_round2.png", rows)
    _bar_plot(output_dir / "pore_error_round2.png", rows)


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


def _save_best_artifacts(
    output_dir: Path,
    image: np.ndarray,
    overlay1: np.ndarray,
    overlay2: np.ndarray,
    ground_truth: np.ndarray,
    source_probability: np.ndarray,
    round1_probability: np.ndarray,
    round2_probability: np.ndarray,
) -> None:
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    probabilities = {
        "source": source_probability,
        "round1": round1_probability,
        "round2": round2_probability,
    }
    predictions = {}
    for name, probability in probabilities.items():
        prediction = probability >= THRESHOLD
        predictions[name] = prediction
        save_probability_map(best_dir / f"{name}_probability.npy", probability)
        Image.fromarray(np.where(prediction, 255, 0).astype(np.uint8)).save(
            best_dir / f"{name}_segmentation.png"
        )
    save_errors(best_dir / "round1_errors.png", ground_truth, predictions["round1"])
    save_errors(best_dir / "round2_errors.png", ground_truth, predictions["round2"])
    panels = (
        _gray_preview(image),
        _mask_preview(predictions["source"]),
        overlay1[..., :3],
        _mask_preview(predictions["round1"]),
        overlay2[..., :3],
        _mask_preview(predictions["round2"]),
        _mask_preview(ground_truth),
        _error_preview(ground_truth, predictions["round2"]),
    )
    titles = (
        "Original",
        "Source",
        "R1 scribbles",
        "R1 prediction",
        "R2 scribbles",
        "R2 prediction",
        "Ground truth",
        "R2 errors (FP red, FN blue)",
    )
    scale = min(1.0, 500 / image.shape[0])
    panel_size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    header = 34
    canvas = Image.new(
        "RGB", (panel_size[0] * len(panels), panel_size[1] + header), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        left = index * panel_size[0]
        resized = Image.fromarray(panel).resize(panel_size, Image.Resampling.NEAREST)
        canvas.paste(resized, (left, header))
        draw.text((left + 5, 10), title, fill="black")
    canvas.save(best_dir / "best_overview.png")


def _configure_determinism(device: str) -> dict[str, object]:
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.manual_seed_all(RANDOM_STATE)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "random_state": RANDOM_STATE,
        "torch_deterministic_algorithms": "enabled with warn_only=True",
        "cuda_cudnn_benchmark": False if cuda else "not applicable",
        "cuda_cudnn_deterministic": True if cuda else "not applicable",
        "cuda_reproducibility_note": (
            "CUDA kernels can retain platform/version-dependent numerical variation."
            if cuda
            else "CPU execution used; optimizer and tile sequence are deterministic."
        ),
    }


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    determinism = _configure_determinism(args.device)
    checkpoint_digest_before = file_digest(args.checkpoint)
    image = load_grayscale_image(args.current)
    overlay1 = load_overlay(args.round1_scribbles)
    overlay2 = load_overlay(args.round2_scribbles)
    scribbles1 = extract_overlay_scribbles(overlay1, expected_shape=image.shape)
    scribbles2 = extract_overlay_scribbles(overlay2, expected_shape=image.shape)
    scribbles2 = validate_cumulative_scribbles(scribbles1, scribbles2)
    frozen = load_frozen_fibnet(args.checkpoint, device=args.device)
    if frozen.feature_mode != "stack_relief" or frozen.model_arch != "resunet":
        raise ValueError("Ablation requires production resunet/stack_relief.")
    if frozen.image_size != MANUSCRIPT_TILE_SIZE:
        raise ValueError(
            f"Checkpoint tile size {frozen.image_size} is not {MANUSCRIPT_TILE_SIZE}."
        )
    source_probability, source_inference_time = _predict_native(
        frozen.model,
        args.current,
        args.previous,
        args.next,
        frozen.feature_mode,
        args.device,
    )
    source_state = {
        name: value.detach().cpu().clone()
        for name, value in frozen.model.state_dict().items()
    }
    source_state_digest = state_dict_digest(source_state)
    round1_tiles = prepare_scribble_training_tiles(
        frozen,
        args.previous,
        args.current,
        args.next,
        scribbles1,
        compute_source_probability=False,
    )
    round2_tiles = prepare_scribble_training_tiles(
        frozen,
        args.previous,
        args.current,
        args.next,
        scribbles2,
        compute_source_probability=False,
    )
    total_parameters = sum(parameter.numel() for parameter in frozen.model.parameters())
    policy_details = {}
    for policy in FREEZE_POLICIES:
        details = describe_freeze_policy(frozen.model, policy)
        policy_details[policy] = {
            "trainable_module_names": list(details.trainable_module_names),
            "trainable_parameters": details.trainable_parameters,
            "frozen_parameters": details.frozen_parameters,
            "total_parameters": details.total_parameters,
        }

    def predict(model: torch.nn.Module) -> tuple[np.ndarray, float]:
        return _predict_native(
            model,
            args.current,
            args.previous,
            args.next,
            frozen.feature_mode,
            args.device,
        )

    result_rows: list[dict[str, object]] = []
    best_probabilities: tuple[np.ndarray, np.ndarray] | None = None
    best_key: tuple[float, float, float, str] | None = None
    run_digests = {}
    grid = configurations()
    for index, (policy, learning_rate, steps) in enumerate(grid, start=1):
        config_id = _config_id(policy, learning_rate, steps)
        sequential = run_sequential_configuration(
            frozen,
            source_state,
            round1_tiles,
            round2_tiles,
            policy,
            learning_rate,
            steps,
            predict,
        )
        if sequential.round1_state_digest != sequential.round2_start_state_digest:
            raise RuntimeError("Round 2 did not start from round-1 weights.")
        run_digests[config_id] = {
            "round1_end": sequential.round1_state_digest,
            "round2_start": sequential.round2_start_state_digest,
        }
        # Ground truth enters only after both predictions for this configuration.
        ground_truth = load_ground_truth(args.mask, image.shape)
        source_metrics = segmentation_metrics(
            ground_truth, source_probability >= THRESHOLD
        )
        round1_metrics = segmentation_metrics(
            ground_truth, sequential.round1.probability >= THRESHOLD
        )
        round2_metrics = segmentation_metrics(
            ground_truth, sequential.round2.probability >= THRESHOLD
        )
        del ground_truth
        rows = (
            _result_row(
                config_id,
                policy,
                learning_rate,
                steps,
                "source",
                source_metrics,
                None,
                source_inference_time,
                0.0,
                total_parameters=total_parameters,
            ),
            _result_row(
                config_id,
                policy,
                learning_rate,
                steps,
                "round1",
                round1_metrics,
                sequential.round1.adaptation,
                sequential.round1.inference_time,
                sequential.round1.adaptation.adaptation_time,
                total_parameters=total_parameters,
            ),
            _result_row(
                config_id,
                policy,
                learning_rate,
                steps,
                "round2",
                round2_metrics,
                sequential.round2.adaptation,
                sequential.round2.inference_time,
                sequential.round1.adaptation.adaptation_time
                + sequential.round2.adaptation.adaptation_time,
                total_parameters=total_parameters,
            ),
        )
        result_rows.extend(rows)
        candidate_key = (
            -round2_metrics["dice"],
            -round2_metrics["iou"],
            round2_metrics["absolute_pore_fraction_error"],
            config_id,
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_probabilities = (
                sequential.round1.probability.copy(),
                sequential.round2.probability.copy(),
            )
        _write_csv(args.output_dir / "results.partial.csv", RESULT_FIELDS, result_rows)
        print(
            f"[{index:02d}/{len(grid)}] {config_id}: "
            f"R1 Dice={round1_metrics['dice']:.5f}, "
            f"R2 Dice={round2_metrics['dice']:.5f}, "
            f"adapt={rows[1]['adaptation_time'] + rows[2]['adaptation_time']:.2f}s",
            flush=True,
        )
        if frozen.device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_id = _config_id("last_decoder_head", 1e-4, 50)
    baseline_rows = [row for row in result_rows if row["config_id"] == baseline_id]
    baseline_total_time = sum(
        float(row["adaptation_time"])
        for row in baseline_rows
        if row["round"] != "source"
    )
    summary_rows = []
    for policy, learning_rate, steps in grid:
        config_id = _config_id(policy, learning_rate, steps)
        rows = [row for row in result_rows if row["config_id"] == config_id]
        summary_rows.append(_summary_row(rows, baseline_total_time))
    ranked = rank_summary_rows(summary_rows)
    _write_csv(args.output_dir / "results.csv", RESULT_FIELDS, result_rows)
    _write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, ranked)
    _write_csv(args.output_dir / "ranking.csv", SUMMARY_FIELDS, ranked)
    save_plots(args.output_dir, ranked)

    best = ranked[0]
    best_id = str(best["config_id"])
    if best_probabilities is None:
        raise RuntimeError("The ablation grid produced no predictions.")
    best_round1, best_round2 = best_probabilities
    ground_truth = load_ground_truth(args.mask, image.shape)
    _save_best_artifacts(
        args.output_dir,
        image,
        overlay1,
        overlay2,
        ground_truth,
        source_probability,
        best_round1,
        best_round2,
    )
    checkpoint_digest_after = file_digest(args.checkpoint)
    if checkpoint_digest_after != checkpoint_digest_before:
        raise RuntimeError("Source checkpoint file changed during the ablation.")
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_before": checkpoint_digest_before,
        "checkpoint_sha256_after": checkpoint_digest_after,
        "source_state_sha256": source_state_digest,
        "source_checkpoint_unchanged": True,
        "native_image_shape": list(image.shape),
        "feature_mode": frozen.feature_mode,
        "tile_size": MANUSCRIPT_TILE_SIZE,
        "overlap": MANUSCRIPT_OVERLAP,
        "overlap_weighting": "production triangular edge-distance weights",
        "threshold": THRESHOLD,
        "optimizer": "Adam, re-created independently for every round",
        "lambda_consistency": LAMBDA_CONSISTENCY,
        "scribble_loss": "BCEWithLogits on labeled pixels only",
        "unlabeled_pixels": "ignored (lambda_consistency=0)",
        "round2_initialization": "round-1 adapted weights; no source reset",
        "tile_sequence": (
            "numpy.default_rng(42) shuffled full passes; reset to the same seed "
            "for every round and configuration"
        ),
        "round1_training_tiles": len(round1_tiles),
        "round2_training_tiles": len(round2_tiles),
        "round1_labeled_pixels": int(np.count_nonzero(scribbles1)),
        "round2_labeled_pixels": int(np.count_nonzero(scribbles2)),
        "ground_truth_used_for": "metrics/artifacts after prediction only",
        "freeze_policies": policy_details,
        "run_state_digests": run_digests,
        "best_config_id": best_id,
        "ranking": "round2 Dice desc, round2 IoU desc, pore error asc",
        "suspicion_rules": [
            "round2 Dice below round1",
            "absolute pore-fraction change between rounds over 0.10",
            "precision drop over 0.15 from source to round2",
            "under 0.005 source-to-round2 Dice gain at >=2x baseline time",
        ],
        **determinism,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    (args.output_dir / "results.partial.csv").unlink(missing_ok=True)
    del best_probabilities, source_state, round1_tiles, round2_tiles, ground_truth
    gc.collect()
    return ranked


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
        default=Path("outputs/adaptation_ablation_60"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ranked = run(args)
    print("\nTop 10 configurations:")
    for row in ranked[:10]:
        print(
            f"#{row['rank']:02d} {row['config_id']} "
            f"R2 Dice={float(row['round2_dice']):.5f} "
            f"IoU={float(row['round2_iou']):.5f} "
            f"precision={float(row['round2_precision']):.5f} "
            f"recall={float(row['round2_recall']):.5f} "
            f"pore_error={float(row['round2_absolute_pore_fraction_error']):.5f}"
        )


if __name__ == "__main__":
    main()
