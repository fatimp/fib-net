"""Benchmark replay and cumulative strategies for sequential slice adaptation."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fibnet.interactive import (
    ScribbleSliceTrainingTiles,
    adapt_from_multislice_scribbles,
    load_frozen_fibnet,
)
from fibnet.probability_io import load_probability_map, save_probability_map
from scripts.benchmark_multislice_adaptation import (
    FREEZE_POLICY,
    LAMBDA_CONSISTENCY,
    LEARNING_RATE,
    RANDOM_STATE,
    THRESHOLD,
    EvaluationInput,
    StackContext,
    _add_scribble_purity,
    _clone_state,
    _load_scribbles_and_tiles,
    _predict_native,
    _save_history,
    _write_csv,
    discover_evaluation_inputs,
    discover_stack_contexts,
    file_digest,
    resolve_scribble_inputs,
    state_dict_digest,
)
from scripts.benchmark_rf_scribbles import load_ground_truth, segmentation_metrics

STEPS_PER_STAGE = 25
ORDER_DEPENDENCE_THRESHOLD = 0.01

PER_STAGE_FIELDS = (
    "strategy",
    "stage_order",
    "stage_number",
    "stage_label",
    "stage_training_slices",
    "evaluation_slice",
    "seen_unseen",
    "dice",
    "iou",
    "precision",
    "recall",
    "accuracy",
    "gt_pore_fraction",
    "predicted_pore_fraction",
    "absolute_pore_fraction_error",
    "stage_adaptation_time",
    "cumulative_adaptation_time",
    "inference_time",
)
SUMMARY_FIELDS = (
    "strategy",
    "stage_order",
    "mean_final_dice",
    "mean_unseen_final_dice",
    "max_forgetting",
    "mean_forgetting",
    "pore_fraction_error",
    "total_adaptation_time",
)
FORGETTING_FIELDS = (
    "strategy",
    "stage_order",
    "evaluation_slice",
    "best_previous_stage",
    "best_previous_dice",
    "final_stage",
    "final_dice",
    "forgetting",
)
WIDE_FIELDS = (
    "strategy",
    "stage_order",
    "evaluation_slice",
    "after46",
    "after60",
    "after86",
)


@dataclass(frozen=True)
class StagePlan:
    """One 25-step stage expressed only in terms of adaptation slice IDs."""

    label: str
    slice_ids: tuple[str, ...]
    slice_weights: tuple[float, ...]


@dataclass(frozen=True)
class StrategyPlan:
    name: str
    order: str
    stages: tuple[StagePlan, ...]


@dataclass(frozen=True)
class StageSnapshot:
    strategy: str
    order: str
    stage_number: int
    stage_label: str
    training_slice_ids: tuple[str, ...]
    stage_adaptation_time: float
    cumulative_adaptation_time: float
    probability_dir: Path
    inference_times: Mapping[str, float]


def build_strategy_plans() -> tuple[StrategyPlan, ...]:
    """Return the locked forward, reverse, cumulative, and replay experiments."""
    return (
        StrategyPlan(
            "sequential_current_only",
            "46->60->86",
            (
                StagePlan("after46", ("46",), (1.0,)),
                StagePlan("after60", ("60",), (1.0,)),
                StagePlan("after86", ("86",), (1.0,)),
            ),
        ),
        StrategyPlan(
            "sequential_current_only_reverse",
            "86->60->46",
            (
                StagePlan("after86", ("86",), (1.0,)),
                StagePlan("after60", ("60",), (1.0,)),
                StagePlan("after46", ("46",), (1.0,)),
            ),
        ),
        StrategyPlan(
            "cumulative_sequential",
            "46->60->86",
            (
                StagePlan("after46", ("46",), (1.0,)),
                StagePlan("after60", ("46", "60"), (1.0, 1.0)),
                StagePlan("after86", ("46", "60", "86"), (1.0, 1.0, 1.0)),
            ),
        ),
        StrategyPlan(
            "replay_50",
            "46->60->86",
            (
                StagePlan("after46", ("46",), (1.0,)),
                StagePlan("after60", ("46", "60"), (1.0, 1.0)),
                StagePlan("after86", ("46", "60", "86"), (1.0, 1.0, 2.0)),
            ),
        ),
        StrategyPlan(
            "cumulative_new_x2",
            "46->60->86",
            (
                StagePlan("after46", ("46",), (1.0,)),
                StagePlan("after60", ("46", "60"), (1.0, 2.0)),
                StagePlan("after86", ("46", "60", "86"), (1.0, 1.0, 2.0)),
            ),
        ),
    )


def _stage_output_dir(
    output_dir: Path, strategy: str, stage_number: int, label: str
) -> Path:
    return output_dir / "strategies" / strategy / f"stage_{stage_number}_{label}"


def _predict_stage(
    output_dir: Path,
    strategy: str,
    stage_number: int,
    stage_label: str,
    model: torch.nn.Module,
    contexts: Sequence[StackContext],
    feature_mode: str,
    device: str,
) -> tuple[Path, dict[str, float]]:
    stage_dir = _stage_output_dir(output_dir, strategy, stage_number, stage_label)
    probability_dir = stage_dir / "probabilities"
    segmentation_dir = stage_dir / "segmentations"
    timings = {}
    for index, context in enumerate(contexts, start=1):
        probability, elapsed = _predict_native(model, context, feature_mode, device)
        save_probability_map(probability_dir / f"{context.slice_id}.npy", probability)
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.where(probability >= THRESHOLD, 255, 0).astype(np.uint8)
        ).save(segmentation_dir / f"{context.slice_id}.png")
        timings[context.slice_id] = elapsed
        print(
            f"  {strategy} {stage_label}: inference {index}/{len(contexts)} "
            f"slice={context.slice_id} time={elapsed:.2f}s",
            flush=True,
        )
        del probability
    return probability_dir, timings


def run_staged_strategy(
    frozen,
    source_state: Mapping[str, torch.Tensor],
    strategy: StrategyPlan,
    groups_by_slice: Mapping[str, ScribbleSliceTrainingTiles],
    evaluation_contexts: Sequence[StackContext],
    output_dir: Path,
    device: str,
) -> tuple[list[StageSnapshot], list[dict[str, object]]]:
    """Run a staged strategy without accepting any ground-truth path or pixels."""
    frozen.model.load_state_dict(source_state)
    snapshots = []
    transitions = []
    cumulative_time = 0.0
    trained_slice_ids: list[str] = []
    previous_end = state_dict_digest(frozen.model.state_dict())
    for stage_number, stage in enumerate(strategy.stages, start=1):
        start_digest = state_dict_digest(frozen.model.state_dict())
        if start_digest != previous_end:
            raise RuntimeError(f"{strategy.name} reset weights between stages.")
        groups = tuple(groups_by_slice[slice_id] for slice_id in stage.slice_ids)
        adaptation = adapt_from_multislice_scribbles(
            frozen,
            groups,
            FREEZE_POLICY,
            LAMBDA_CONSISTENCY,
            steps=STEPS_PER_STAGE,
            learning_rate=LEARNING_RATE,
            sampling_strategy="joint_slice_balanced",
            slice_weights=stage.slice_weights,
            random_state=RANDOM_STATE,
        )
        cumulative_time += adaptation.adaptation_time
        end_digest = state_dict_digest(frozen.model.state_dict())
        previous_end = end_digest
        for slice_id in stage.slice_ids:
            if slice_id not in trained_slice_ids:
                trained_slice_ids.append(slice_id)
        stage_dir = _stage_output_dir(
            output_dir, strategy.name, stage_number, stage.label
        )
        _save_history(stage_dir / "history.csv", (adaptation,))
        probability_dir, inference_times = _predict_stage(
            output_dir,
            strategy.name,
            stage_number,
            stage.label,
            frozen.model,
            evaluation_contexts,
            frozen.feature_mode,
            device,
        )
        counts = Counter(step.slice_id for step in adaptation.history)
        transitions.append(
            {
                "stage_number": stage_number,
                "stage_label": stage.label,
                "slice_ids": list(stage.slice_ids),
                "slice_weights": list(stage.slice_weights),
                "sample_counts": dict(counts),
                "start_state_sha256": start_digest,
                "end_state_sha256": end_digest,
            }
        )
        snapshots.append(
            StageSnapshot(
                strategy=strategy.name,
                order=strategy.order,
                stage_number=stage_number,
                stage_label=stage.label,
                training_slice_ids=tuple(trained_slice_ids),
                stage_adaptation_time=adaptation.adaptation_time,
                cumulative_adaptation_time=cumulative_time,
                probability_dir=probability_dir,
                inference_times=inference_times,
            )
        )
    return snapshots, transitions


def run_joint_baseline(
    frozen,
    source_state: Mapping[str, torch.Tensor],
    groups: Sequence[ScribbleSliceTrainingTiles],
    evaluation_contexts: Sequence[StackContext],
    output_dir: Path,
    device: str,
) -> tuple[StageSnapshot, dict[str, object]]:
    """Run the locked 75-step tile-uniform joint baseline as one session."""
    strategy = "joint_pixel_uniform_75"
    frozen.model.load_state_dict(source_state)
    start_digest = state_dict_digest(frozen.model.state_dict())
    adaptation = adapt_from_multislice_scribbles(
        frozen,
        groups,
        FREEZE_POLICY,
        LAMBDA_CONSISTENCY,
        steps=75,
        learning_rate=LEARNING_RATE,
        sampling_strategy="joint_pixel_uniform",
        random_state=RANDOM_STATE,
    )
    end_digest = state_dict_digest(frozen.model.state_dict())
    stage_dir = _stage_output_dir(output_dir, strategy, 3, "after86")
    _save_history(stage_dir / "history.csv", (adaptation,))
    probability_dir, inference_times = _predict_stage(
        output_dir,
        strategy,
        3,
        "after86",
        frozen.model,
        evaluation_contexts,
        frozen.feature_mode,
        device,
    )
    snapshot = StageSnapshot(
        strategy=strategy,
        order="joint",
        stage_number=3,
        stage_label="after86",
        training_slice_ids=("46", "60", "86"),
        stage_adaptation_time=adaptation.adaptation_time,
        cumulative_adaptation_time=adaptation.adaptation_time,
        probability_dir=probability_dir,
        inference_times=inference_times,
    )
    audit = {
        "stage_number": 3,
        "stage_label": "after86",
        "slice_ids": ["46", "60", "86"],
        "slice_weights": "tile-uniform",
        "sample_counts": dict(Counter(step.slice_id for step in adaptation.history)),
        "start_state_sha256": start_digest,
        "end_state_sha256": end_digest,
    }
    return snapshot, audit


def _load_ground_truth_after_predictions(
    evaluations: Sequence[EvaluationInput],
) -> dict[str, np.ndarray]:
    """Load evaluation masks at the explicit post-prediction GT boundary."""
    ground_truth = {}
    for evaluation in evaluations:
        with Image.open(evaluation.context.image_path) as image:
            shape = (image.height, image.width)
        ground_truth[evaluation.context.slice_id] = load_ground_truth(
            evaluation.mask_path, shape
        )
    return ground_truth


def evaluate_snapshots(
    snapshots: Sequence[StageSnapshot],
    ground_truth: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for snapshot in snapshots:
        trained = set(snapshot.training_slice_ids)
        for slice_id, mask in ground_truth.items():
            probability = load_probability_map(
                snapshot.probability_dir / f"{slice_id}.npy"
            )
            metrics = segmentation_metrics(mask, probability >= THRESHOLD)
            rows.append(
                {
                    "strategy": snapshot.strategy,
                    "stage_order": snapshot.order,
                    "stage_number": snapshot.stage_number,
                    "stage_label": snapshot.stage_label,
                    "stage_training_slices": "+".join(snapshot.training_slice_ids),
                    "evaluation_slice": slice_id,
                    "seen_unseen": "seen" if slice_id in trained else "unseen",
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "accuracy": metrics["accuracy"],
                    "gt_pore_fraction": metrics["pore_fraction_gt"],
                    "predicted_pore_fraction": metrics["pore_fraction_prediction"],
                    "absolute_pore_fraction_error": metrics[
                        "absolute_pore_fraction_error"
                    ],
                    "stage_adaptation_time": snapshot.stage_adaptation_time,
                    "cumulative_adaptation_time": snapshot.cumulative_adaptation_time,
                    "inference_time": snapshot.inference_times[slice_id],
                }
            )
            del probability
    return rows


def build_wide_dice_rows(
    result_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    keys = sorted(
        {
            (
                str(row["strategy"]),
                str(row["stage_order"]),
                str(row["evaluation_slice"]),
            )
            for row in result_rows
        },
        key=lambda item: (item[0], int(item[2]) if item[2].isdigit() else item[2]),
    )
    for strategy, order, slice_id in keys:
        matching = [
            row
            for row in result_rows
            if row["strategy"] == strategy and row["evaluation_slice"] == slice_id
        ]
        by_label = {str(row["stage_label"]): row["dice"] for row in matching}
        rows.append(
            {
                "strategy": strategy,
                "stage_order": order,
                "evaluation_slice": slice_id,
                "after46": by_label.get("after46", ""),
                "after60": by_label.get("after60", ""),
                "after86": by_label.get("after86", ""),
            }
        )
    return rows


def compute_forgetting_rows(
    result_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    forgetting_rows = []
    strategy_names = sorted({str(row["strategy"]) for row in result_rows})
    for strategy in strategy_names:
        strategy_rows = [row for row in result_rows if row["strategy"] == strategy]
        order = str(strategy_rows[0]["stage_order"])
        slice_ids = sorted(
            {str(row["evaluation_slice"]) for row in strategy_rows},
            key=lambda value: int(value) if value.isdigit() else value,
        )
        for slice_id in slice_ids:
            trajectory = sorted(
                (
                    row
                    for row in strategy_rows
                    if str(row["evaluation_slice"]) == slice_id
                ),
                key=lambda row: int(row["stage_number"]),
            )
            final = trajectory[-1]
            previous = trajectory[:-1]
            if previous:
                best = max(previous, key=lambda row: float(row["dice"]))
                best_stage: object = best["stage_label"]
                best_dice: object = float(best["dice"])
                forgetting: object = best_dice - float(final["dice"])
            else:
                best_stage = ""
                best_dice = ""
                forgetting = ""
            forgetting_rows.append(
                {
                    "strategy": strategy,
                    "stage_order": order,
                    "evaluation_slice": slice_id,
                    "best_previous_stage": best_stage,
                    "best_previous_dice": best_dice,
                    "final_stage": final["stage_label"],
                    "final_dice": final["dice"],
                    "forgetting": forgetting,
                }
            )
    return forgetting_rows


def build_summary_rows(
    result_rows: Sequence[Mapping[str, object]],
    forgetting_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    summaries = []
    strategy_names = sorted({str(row["strategy"]) for row in result_rows})
    for strategy in strategy_names:
        strategy_rows = [row for row in result_rows if row["strategy"] == strategy]
        final_stage_number = max(int(row["stage_number"]) for row in strategy_rows)
        final_rows = [
            row
            for row in strategy_rows
            if int(row["stage_number"]) == final_stage_number
        ]
        unseen = [row for row in final_rows if row["seen_unseen"] == "unseen"]
        forgetting_values = [
            float(row["forgetting"])
            for row in forgetting_rows
            if row["strategy"] == strategy and row["forgetting"] != ""
        ]
        summaries.append(
            {
                "strategy": strategy,
                "stage_order": final_rows[0]["stage_order"],
                "mean_final_dice": float(
                    np.mean([float(row["dice"]) for row in final_rows])
                ),
                "mean_unseen_final_dice": float(
                    np.mean([float(row["dice"]) for row in unseen])
                ),
                "max_forgetting": (max(forgetting_values) if forgetting_values else ""),
                "mean_forgetting": (
                    float(np.mean(forgetting_values)) if forgetting_values else ""
                ),
                "pore_fraction_error": float(
                    np.mean(
                        [
                            float(row["absolute_pore_fraction_error"])
                            for row in final_rows
                        ]
                    )
                ),
                "total_adaptation_time": float(
                    final_rows[0]["cumulative_adaptation_time"]
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            -float(row["mean_unseen_final_dice"]),
            -float(row["mean_final_dice"]),
            float(row["pore_fraction_error"]),
        ),
    )


def build_order_sensitivity_rows(
    result_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    forward_name = "sequential_current_only"
    reverse_name = "sequential_current_only_reverse"
    final_by_strategy = {}
    for strategy in (forward_name, reverse_name):
        matching = [row for row in result_rows if row["strategy"] == strategy]
        final_stage = max(int(row["stage_number"]) for row in matching)
        final_by_strategy[strategy] = {
            str(row["evaluation_slice"]): row
            for row in matching
            if int(row["stage_number"]) == final_stage
        }
    rows = []
    for slice_id in sorted(
        final_by_strategy[forward_name],
        key=lambda value: int(value) if value.isdigit() else value,
    ):
        forward = float(final_by_strategy[forward_name][slice_id]["dice"])
        reverse = float(final_by_strategy[reverse_name][slice_id]["dice"])
        difference = reverse - forward
        rows.append(
            {
                "evaluation_slice": slice_id,
                "forward_final_dice": forward,
                "reverse_final_dice": reverse,
                "reverse_minus_forward": difference,
                "absolute_difference": abs(difference),
                "substantial_at_0p01": abs(difference) >= ORDER_DEPENDENCE_THRESHOLD,
            }
        )
    return rows


def run(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_before = file_digest(args.checkpoint)
    contexts = discover_stack_contexts(args.target_dir)
    evaluations = discover_evaluation_inputs(contexts, args.ideal_dir)
    evaluation_contexts = tuple(item.context for item in evaluations)
    scribble_inputs = resolve_scribble_inputs(contexts, args.scribble_dir)
    print(
        "evaluation slices: "
        + ", ".join(context.slice_id for context in evaluation_contexts),
        flush=True,
    )

    frozen = load_frozen_fibnet(args.checkpoint, device=args.device)
    if frozen.feature_mode != "stack_relief" or frozen.model_arch != "resunet":
        raise ValueError(
            "Benchmark requires the production resunet/stack_relief checkpoint; "
            f"received {frozen.model_arch}/{frozen.feature_mode}."
        )
    if frozen.image_size != 384:
        raise ValueError(f"Checkpoint tile size {frozen.image_size} is not 384.")
    source_state = _clone_state(frozen.model)
    source_state_sha256 = state_dict_digest(source_state)
    scribbles, tile_groups, scribble_stats = _load_scribbles_and_tiles(
        frozen, scribble_inputs
    )
    groups_by_slice = {group.slice_id: group for group in tile_groups}

    all_snapshots = []
    transition_audit = {}
    joint_snapshot, joint_audit = run_joint_baseline(
        frozen,
        source_state,
        tile_groups,
        evaluation_contexts,
        args.output_dir,
        args.device,
    )
    all_snapshots.append(joint_snapshot)
    transition_audit[joint_snapshot.strategy] = [joint_audit]

    for strategy in build_strategy_plans():
        snapshots, transitions = run_staged_strategy(
            frozen,
            source_state,
            strategy,
            groups_by_slice,
            evaluation_contexts,
            args.output_dir,
            args.device,
        )
        all_snapshots.extend(snapshots)
        transition_audit[strategy.name] = transitions
        if frozen.device.type == "cuda":
            torch.cuda.empty_cache()

    # Explicit GT boundary: all model updates and predictions are now complete.
    ground_truth = _load_ground_truth_after_predictions(evaluations)
    _add_scribble_purity(scribble_stats, scribbles, ground_truth)
    result_rows = evaluate_snapshots(all_snapshots, ground_truth)
    wide_rows = build_wide_dice_rows(result_rows)
    forgetting_rows = compute_forgetting_rows(result_rows)
    summary_rows = build_summary_rows(result_rows, forgetting_rows)
    order_rows = build_order_sensitivity_rows(result_rows)

    _write_csv(args.output_dir / "results_per_stage.csv", PER_STAGE_FIELDS, result_rows)
    _write_csv(args.output_dir / "dice_by_stage.csv", WIDE_FIELDS, wide_rows)
    _write_csv(args.output_dir / "forgetting.csv", FORGETTING_FIELDS, forgetting_rows)
    _write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(
        args.output_dir / "order_sensitivity.csv",
        (
            "evaluation_slice",
            "forward_final_dice",
            "reverse_final_dice",
            "reverse_minus_forward",
            "absolute_difference",
            "substantial_at_0p01",
        ),
        order_rows,
    )
    _write_csv(
        args.output_dir / "scribble_stats.csv",
        tuple(scribble_stats[0]),
        scribble_stats,
    )

    checkpoint_after = file_digest(args.checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("Source checkpoint changed during the benchmark.")
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "source_checkpoint_unchanged": True,
        "source_state_sha256": source_state_sha256,
        "feature_mode": frozen.feature_mode,
        "model_arch": frozen.model_arch,
        "freeze_policy": FREEZE_POLICY,
        "encoder": "frozen",
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam, reset at each sequential stage",
        "steps_per_stage": STEPS_PER_STAGE,
        "lambda_consistency": LAMBDA_CONSISTENCY,
        "threshold": THRESHOLD,
        "tile_size": 384,
        "overlap": 96,
        "ground_truth_used_for": "evaluation only, after all adaptation and inference",
        "forgetting_definition": "best Dice among prior adapted stages minus final Dice",
        "joint_baseline_note": "one 75-step session; only its final value is placed in after86",
        "order_dependence_threshold": ORDER_DEPENDENCE_THRESHOLD,
        "evaluation_slices": [context.slice_id for context in evaluation_contexts],
        "transition_audit": transition_audit,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    del source_state, tile_groups, scribbles, ground_truth
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
        "--output-dir",
        type=Path,
        default=Path("outputs/sequential_strategy_adaptation"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _results, summary = run(args)
    print("\nFinal ranking:")
    for row in summary:
        forgetting = row["max_forgetting"]
        print(
            f"{row['strategy']}: final={float(row['mean_final_dice']):.5f} "
            f"unseen={float(row['mean_unseen_final_dice']):.5f} "
            f"max_forgetting={forgetting if forgetting == '' else f'{float(forgetting):.5f}'} "
            f"pore_error={float(row['pore_fraction_error']):.5f}"
        )
    print(f"outputs={args.output_dir}")


if __name__ == "__main__":
    main()
