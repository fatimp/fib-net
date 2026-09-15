"""Benchmark source-weight and confidence-gated functional anchoring."""

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

from fibnet.inference import iter_tiled_inputs
from fibnet.interactive import (
    FunctionalAnchorTile,
    ScribbleSliceTrainingTiles,
    adapt_from_scribbles_with_anchors,
    load_frozen_fibnet,
)
from scripts.benchmark_multislice_adaptation import (
    FREEZE_POLICY,
    LAMBDA_CONSISTENCY,
    LEARNING_RATE,
    RANDOM_STATE,
    THRESHOLD,
    _add_scribble_purity,
    _clone_state,
    _load_scribbles_and_tiles,
    _save_history,
    _write_csv,
    discover_evaluation_inputs,
    discover_stack_contexts,
    file_digest,
    resolve_scribble_inputs,
    state_dict_digest,
)
from scripts.benchmark_sequential_strategies import (
    PER_STAGE_FIELDS,
    SUMMARY_FIELDS,
    WIDE_FIELDS,
    StageSnapshot,
    _load_ground_truth_after_predictions,
    _predict_stage,
    build_summary_rows,
    build_wide_dice_rows,
    compute_forgetting_rows,
    evaluate_snapshots,
)

TRAINING_ORDER = ("46", "60", "86")
REQUESTED_REPRESENTATIVE_SLICES = (
    "10",
    "20",
    "30",
    "40",
    "50",
    "70",
    "80",
    "90",
    "100",
    "110",
    "120",
)
WEIGHT_LAMBDAS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
FUNCTIONAL_LAMBDAS = (0.01, 0.1, 1.0)
CONFIDENCE_LOW = 0.05
CONFIDENCE_HIGH = 0.95
STEPS_PER_STAGE = 25

REGULARIZED_SUMMARY_FIELDS = (
    *SUMMARY_FIELDS,
    "dice47",
    "dice70",
    "lambda_weight",
    "lambda_anchor",
)
LOSS_SCALE_FIELDS = (
    "strategy",
    "stage_number",
    "stage_label",
    "scribble_slice",
    "lambda_weight",
    "lambda_anchor",
    "initial_scribble_loss",
    "final_scribble_loss",
    "mean_scribble_loss",
    "final_raw_weight_anchor_loss",
    "mean_raw_weight_anchor_loss",
    "mean_weighted_weight_anchor_loss",
    "final_raw_functional_anchor_loss",
    "mean_raw_functional_anchor_loss",
    "mean_weighted_functional_anchor_loss",
    "mean_confident_pixels",
    "mean_confident_fraction",
)


@dataclass(frozen=True)
class RegularizationExperiment:
    name: str
    lambda_weight: float
    lambda_anchor: float


def _lambda_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def candidate_experiments() -> tuple[RegularizationExperiment, ...]:
    return (
        RegularizationExperiment("sequential_current_only", 0.0, 0.0),
        *(
            RegularizationExperiment(f"weight_anchor_{_lambda_slug(value)}", value, 0.0)
            for value in WEIGHT_LAMBDAS[1:]
        ),
        *(
            RegularizationExperiment(
                f"functional_anchor_{_lambda_slug(value)}", 0.0, value
            )
            for value in FUNCTIONAL_LAMBDAS
        ),
    )


def select_representative_slice_ids(
    available_slice_ids: Sequence[str],
    excluded: Sequence[str] = TRAINING_ORDER,
) -> tuple[str, ...]:
    """Select the requested evenly spaced z-grid without training slices."""
    available = set(available_slice_ids)
    excluded_set = set(excluded)
    selected = tuple(
        slice_id
        for slice_id in REQUESTED_REPRESENTATIVE_SLICES
        if slice_id in available and slice_id not in excluded_set
    )
    if len(selected) < 3:
        numeric = sorted(
            int(value)
            for value in available
            if value.isdigit() and value not in excluded_set
        )
        if len(numeric) < 3:
            raise ValueError("At least three non-training stack slices are required.")
        indices = np.linspace(0, len(numeric) - 1, min(11, len(numeric)))
        selected = tuple(str(numeric[round(index)]) for index in indices)
    if set(selected) & excluded_set:
        raise RuntimeError("Auxiliary representative grid contains a scribble slice.")
    return selected


def _axis_tile_count(length: int, tile_size: int = 384, overlap: int = 96) -> int:
    if length <= tile_size:
        return 1
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return len(positions)


def context_tile_count(context) -> int:
    with Image.open(context.image_path) as image:
        width, height = image.size
    return _axis_tile_count(width) * _axis_tile_count(height)


def build_auxiliary_schedule(
    representative_slice_ids: Sequence[str],
    tile_counts: Mapping[str, int],
    *,
    steps: int,
    random_state: int,
) -> tuple[tuple[str, int], ...]:
    """Balance z first, then cycle uniformly through spatial tiles per slice."""
    if not representative_slice_ids:
        raise ValueError("At least one representative slice is required.")
    if any(tile_counts.get(slice_id, 0) < 1 for slice_id in representative_slice_ids):
        raise ValueError("Every representative slice must contain at least one tile.")
    generator = np.random.default_rng(random_state)
    slice_order = []
    while len(slice_order) < steps:
        slice_order.extend(
            representative_slice_ids[int(index)]
            for index in generator.permutation(len(representative_slice_ids))
        )
    tile_cycles: dict[str, list[int]] = {
        slice_id: [] for slice_id in representative_slice_ids
    }
    schedule = []
    for slice_id in slice_order[:steps]:
        if not tile_cycles[slice_id]:
            tile_cycles[slice_id].extend(
                int(index) for index in generator.permutation(tile_counts[slice_id])
            )
        schedule.append((slice_id, tile_cycles[slice_id].pop(0)))
    return tuple(schedule)


def prepare_functional_anchor_sequences(
    source_teacher,
    contexts: Mapping[str, object],
    representative_slice_ids: Sequence[str],
) -> tuple[tuple[FunctionalAnchorTile, ...], ...]:
    """Cache only the 75 no-GT auxiliary tiles required by three stages."""
    tile_counts = {
        slice_id: context_tile_count(contexts[slice_id])
        for slice_id in representative_slice_ids
    }
    schedules = tuple(
        build_auxiliary_schedule(
            representative_slice_ids,
            tile_counts,
            steps=STEPS_PER_STAGE,
            random_state=RANDOM_STATE + 1000 * stage_number,
        )
        for stage_number in range(1, 4)
    )
    needed: dict[str, set[int]] = {
        slice_id: set() for slice_id in representative_slice_ids
    }
    for schedule in schedules:
        for slice_id, tile_index in schedule:
            needed[slice_id].add(tile_index)

    cached = {}
    source_teacher.model.eval()
    with torch.no_grad():
        for slice_id in representative_slice_ids:
            context = contexts[slice_id]
            for tile_index, tile in enumerate(
                iter_tiled_inputs(
                    context.image_path,
                    384,
                    96,
                    source_teacher.feature_mode,
                    previous_path=context.previous_path,
                    next_path=context.next_path,
                )
            ):
                if tile_index not in needed[slice_id]:
                    continue
                source_probability = torch.sigmoid(
                    source_teacher.model(tile.tensor.to(source_teacher.device))[0, 0]
                ).cpu()
                confidence = (source_probability <= CONFIDENCE_LOW) | (
                    source_probability >= CONFIDENCE_HIGH
                )
                if tile.crop_height < 384 or tile.crop_width < 384:
                    valid = torch.zeros((384, 384), dtype=torch.bool)
                    valid[: tile.crop_height, : tile.crop_width] = True
                    confidence &= valid
                if not torch.any(confidence):
                    raise ValueError(
                        f"Source teacher has no confident pixels on {slice_id} "
                        f"tile {tile_index}."
                    )
                cached[(slice_id, tile_index)] = FunctionalAnchorTile(
                    tensor=tile.tensor.cpu(),
                    source_probability=source_probability,
                    confidence_mask=confidence,
                    slice_id=slice_id,
                    tile_index=tile_index,
                    x=tile.x,
                    y=tile.y,
                )
    missing = {key for schedule in schedules for key in schedule if key not in cached}
    if missing:
        raise RuntimeError(f"Failed to prepare auxiliary tiles: {sorted(missing)}")
    sequences = tuple(tuple(cached[key] for key in schedule) for schedule in schedules)
    counts = Counter(slice_id for schedule in schedules for slice_id, _ in schedule)
    print(
        "auxiliary teacher grid: "
        + ", ".join(f"{key}={counts[key]}" for key in representative_slice_ids),
        flush=True,
    )
    return sequences


def _loss_scale_row(
    experiment: RegularizationExperiment,
    stage_number: int,
    scribble_slice: str,
    adaptation,
) -> dict[str, object]:
    history = adaptation.history
    return {
        "strategy": experiment.name,
        "stage_number": stage_number,
        "stage_label": f"after{scribble_slice}",
        "scribble_slice": scribble_slice,
        "lambda_weight": experiment.lambda_weight,
        "lambda_anchor": experiment.lambda_anchor,
        "initial_scribble_loss": history[0].scribble_loss,
        "final_scribble_loss": history[-1].scribble_loss,
        "mean_scribble_loss": float(np.mean([step.scribble_loss for step in history])),
        "final_raw_weight_anchor_loss": history[-1].weight_anchor_loss,
        "mean_raw_weight_anchor_loss": float(
            np.mean([step.weight_anchor_loss for step in history])
        ),
        "mean_weighted_weight_anchor_loss": float(
            np.mean([step.weighted_weight_anchor_loss for step in history])
        ),
        "final_raw_functional_anchor_loss": history[-1].functional_anchor_loss,
        "mean_raw_functional_anchor_loss": float(
            np.mean([step.functional_anchor_loss for step in history])
        ),
        "mean_weighted_functional_anchor_loss": float(
            np.mean([step.weighted_functional_anchor_loss for step in history])
        ),
        "mean_confident_pixels": float(
            np.mean([step.functional_confident_pixels for step in history])
        ),
        "mean_confident_fraction": float(
            np.mean([step.functional_confident_fraction for step in history])
        ),
    }


def run_experiment(
    frozen,
    source_state: Mapping[str, torch.Tensor],
    experiment: RegularizationExperiment,
    groups_by_slice: Mapping[str, ScribbleSliceTrainingTiles],
    auxiliary_sequences: Sequence[Sequence[FunctionalAnchorTile]],
    evaluation_contexts: Sequence[object],
    output_dir: Path,
    device: str,
    training_order: Sequence[str] = TRAINING_ORDER,
) -> tuple[list[StageSnapshot], list[dict[str, object]], list[dict[str, object]]]:
    """Run current-only stages without accepting a GT path or mask."""
    frozen.model.load_state_dict(source_state)
    snapshots = []
    loss_scales = []
    transitions = []
    cumulative_time = 0.0
    previous_end = state_dict_digest(frozen.model.state_dict())
    trained = []
    order_label = "->".join(training_order)
    for stage_number, slice_id in enumerate(training_order, start=1):
        start_digest = state_dict_digest(frozen.model.state_dict())
        if start_digest != previous_end:
            raise RuntimeError(f"{experiment.name} reset weights between stages.")
        auxiliary = (
            auxiliary_sequences[stage_number - 1]
            if experiment.lambda_anchor > 0.0
            else ()
        )
        adaptation = adapt_from_scribbles_with_anchors(
            frozen,
            groups_by_slice[slice_id].tiles,
            FREEZE_POLICY,
            steps=STEPS_PER_STAGE,
            learning_rate=LEARNING_RATE,
            source_state=source_state,
            lambda_weight=experiment.lambda_weight,
            log_weight_anchor=experiment.lambda_weight >= 0.0
            and experiment.lambda_anchor == 0.0,
            auxiliary_tiles=auxiliary,
            lambda_functional=experiment.lambda_anchor,
            scribble_slice_id=slice_id,
            random_state=RANDOM_STATE,
        )
        cumulative_time += adaptation.adaptation_time
        end_digest = state_dict_digest(frozen.model.state_dict())
        previous_end = end_digest
        trained.append(slice_id)
        stage_label = f"after{slice_id}"
        strategy_dir = output_dir / "strategies" / experiment.name
        stage_dir = strategy_dir / f"stage_{stage_number}_{stage_label}"
        _save_history(stage_dir / "history.csv", (adaptation,))
        probability_dir, inference_times = _predict_stage(
            output_dir,
            experiment.name,
            stage_number,
            stage_label,
            frozen.model,
            evaluation_contexts,
            frozen.feature_mode,
            device,
        )
        snapshots.append(
            StageSnapshot(
                strategy=experiment.name,
                order=order_label,
                stage_number=stage_number,
                stage_label=stage_label,
                training_slice_ids=tuple(trained),
                stage_adaptation_time=adaptation.adaptation_time,
                cumulative_adaptation_time=cumulative_time,
                probability_dir=probability_dir,
                inference_times=inference_times,
            )
        )
        loss_scales.append(
            _loss_scale_row(experiment, stage_number, slice_id, adaptation)
        )
        transitions.append(
            {
                "stage_number": stage_number,
                "stage_label": stage_label,
                "scribble_slice": slice_id,
                "auxiliary_slice_counts": dict(
                    Counter(
                        step.functional_anchor_slice_id
                        for step in adaptation.history
                        if step.functional_anchor_slice_id is not None
                    )
                ),
                "start_state_sha256": start_digest,
                "end_state_sha256": end_digest,
            }
        )
    return snapshots, loss_scales, transitions


def _augment_summaries(
    summaries: list[dict[str, object]],
    results: Sequence[Mapping[str, object]],
    experiments: Sequence[RegularizationExperiment],
) -> None:
    experiments_by_name = {item.name: item for item in experiments}
    for summary in summaries:
        strategy = str(summary["strategy"])
        final = [
            row
            for row in results
            if row["strategy"] == strategy and int(row["stage_number"]) == 3
        ]
        by_slice = {str(row["evaluation_slice"]): row for row in final}
        experiment = experiments_by_name[strategy]
        summary["dice47"] = by_slice["47"]["dice"]
        summary["dice70"] = by_slice["70"]["dice"]
        summary["lambda_weight"] = experiment.lambda_weight
        summary["lambda_anchor"] = experiment.lambda_anchor


def _experiments_from_args(
    args: argparse.Namespace,
) -> tuple[RegularizationExperiment, ...]:
    if args.experiment_set == "candidates":
        return candidate_experiments()
    if args.experiment_set == "order_check":
        if args.order_lambda_anchor <= 0.0:
            raise ValueError("--order-lambda-anchor must be positive.")
        return (
            RegularizationExperiment(
                f"functional_anchor_{_lambda_slug(args.order_lambda_anchor)}_reverse",
                0.0,
                args.order_lambda_anchor,
            ),
        )
    if args.combined_lambda_weight is None or args.combined_lambda_anchor is None:
        raise ValueError(
            "Combined runs require --combined-lambda-weight and "
            "--combined-lambda-anchor."
        )
    if args.combined_lambda_weight <= 0.0 or args.combined_lambda_anchor <= 0.0:
        raise ValueError("Combined anchor coefficients must be positive.")
    return (
        RegularizationExperiment(
            (
                f"combined_weight_{_lambda_slug(args.combined_lambda_weight)}"
                f"_functional_{_lambda_slug(args.combined_lambda_anchor)}"
            ),
            args.combined_lambda_weight,
            args.combined_lambda_anchor,
        ),
    )


def run(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    experiments = _experiments_from_args(args)
    run_dir = args.output_dir / args.experiment_set
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_before = file_digest(args.checkpoint)
    contexts = discover_stack_contexts(args.target_dir)
    evaluations = discover_evaluation_inputs(contexts, args.ideal_dir)
    evaluation_contexts = tuple(item.context for item in evaluations)
    scribble_inputs = resolve_scribble_inputs(contexts, args.scribble_dir)
    representative_ids = select_representative_slice_ids(tuple(contexts))
    print(f"representative slices: {', '.join(representative_ids)}", flush=True)

    frozen = load_frozen_fibnet(args.checkpoint, device=args.device)
    if frozen.feature_mode != "stack_relief" or frozen.model_arch != "resunet":
        raise ValueError("Benchmark requires production resunet/stack_relief.")
    if frozen.image_size != 384:
        raise ValueError(f"Checkpoint tile size {frozen.image_size} is not 384.")
    source_state = _clone_state(frozen.model)
    source_state_sha256 = state_dict_digest(source_state)
    scribbles, tile_groups, scribble_stats = _load_scribbles_and_tiles(
        frozen, scribble_inputs
    )
    groups_by_slice = {group.slice_id: group for group in tile_groups}
    auxiliary_sequences: tuple[tuple[FunctionalAnchorTile, ...], ...] = ()
    if any(experiment.lambda_anchor > 0.0 for experiment in experiments):
        auxiliary_sequences = prepare_functional_anchor_sequences(
            frozen, contexts, representative_ids
        )

    snapshots = []
    loss_scales = []
    transition_audit = {}
    training_order = (
        tuple(reversed(TRAINING_ORDER))
        if args.experiment_set == "order_check"
        else TRAINING_ORDER
    )
    for experiment in experiments:
        experiment_snapshots, experiment_scales, transitions = run_experiment(
            frozen,
            source_state,
            experiment,
            groups_by_slice,
            auxiliary_sequences,
            evaluation_contexts,
            run_dir,
            args.device,
            training_order,
        )
        snapshots.extend(experiment_snapshots)
        loss_scales.extend(experiment_scales)
        transition_audit[experiment.name] = transitions
        if frozen.device.type == "cuda":
            torch.cuda.empty_cache()

    # GT is first loaded here, after every requested adaptation and inference.
    ground_truth = _load_ground_truth_after_predictions(evaluations)
    _add_scribble_purity(scribble_stats, scribbles, ground_truth)
    result_rows = evaluate_snapshots(snapshots, ground_truth)
    forgetting_rows = compute_forgetting_rows(result_rows)
    summary_rows = build_summary_rows(result_rows, forgetting_rows)
    _augment_summaries(summary_rows, result_rows, experiments)
    wide_rows = build_wide_dice_rows(result_rows)

    _write_csv(run_dir / "results_per_stage.csv", PER_STAGE_FIELDS, result_rows)
    _write_csv(run_dir / "dice_by_stage.csv", WIDE_FIELDS, wide_rows)
    _write_csv(
        run_dir / "forgetting.csv",
        (
            "strategy",
            "stage_order",
            "evaluation_slice",
            "best_previous_stage",
            "best_previous_dice",
            "final_stage",
            "final_dice",
            "forgetting",
        ),
        forgetting_rows,
    )
    _write_csv(run_dir / "summary.csv", REGULARIZED_SUMMARY_FIELDS, summary_rows)
    _write_csv(run_dir / "loss_scales.csv", LOSS_SCALE_FIELDS, loss_scales)
    _write_csv(run_dir / "scribble_stats.csv", tuple(scribble_stats[0]), scribble_stats)

    checkpoint_after = file_digest(args.checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("Source checkpoint changed during the benchmark.")
    metadata = {
        "experiment_set": args.experiment_set,
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
        "optimizer": "Adam, reset at each stage",
        "steps_per_stage": STEPS_PER_STAGE,
        "lambda_consistency": LAMBDA_CONSISTENCY,
        "threshold": THRESHOLD,
        "training_order": list(training_order),
        "representative_slices": list(representative_ids),
        "excluded_auxiliary_slices": list(TRAINING_ORDER),
        "confidence_gate": (
            f"p_source <= {CONFIDENCE_LOW} or p_source >= {CONFIDENCE_HIGH}"
        ),
        "functional_anchor": "Bernoulli KL(source || adapted)",
        "weight_anchor": "mean squared displacement over trainable parameters from source checkpoint",
        "ground_truth_used_for": "evaluation only after all adaptation/inference",
        "experiments": [experiment.__dict__ for experiment in experiments],
        "transition_audit": transition_audit,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    del source_state, tile_groups, auxiliary_sequences, ground_truth, scribbles
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
        default=Path("outputs/regularized_sequential_adaptation"),
    )
    parser.add_argument(
        "--experiment-set",
        choices=("candidates", "combined", "order_check"),
        default="candidates",
    )
    parser.add_argument("--combined-lambda-weight", type=float, default=None)
    parser.add_argument("--combined-lambda-anchor", type=float, default=None)
    parser.add_argument("--order-lambda-anchor", type=float, default=0.01)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _results, summary = run(args)
    print("\nFinal ranking:")
    for row in summary:
        print(
            f"{row['strategy']}: final={float(row['mean_final_dice']):.5f} "
            f"unseen={float(row['mean_unseen_final_dice']):.5f} "
            f"D47={float(row['dice47']):.5f} D70={float(row['dice70']):.5f} "
            f"forget={float(row['max_forgetting']):.5f} "
            f"pore_error={float(row['pore_fraction_error']):.5f}"
        )
    print(f"outputs={args.output_dir / args.experiment_set}")


if __name__ == "__main__":
    main()
