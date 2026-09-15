"""Benchmark confidence and disagreement gates for functional anchoring."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

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
from scripts.benchmark_regularized_sequential import (
    REQUESTED_REPRESENTATIVE_SLICES,
    STEPS_PER_STAGE,
    prepare_functional_anchor_sequences,
    select_representative_slice_ids,
)
from scripts.benchmark_sequential_strategies import (
    FORGETTING_FIELDS,
    PER_STAGE_FIELDS,
    WIDE_FIELDS,
    StageSnapshot,
    _load_ground_truth_after_predictions,
    _predict_stage,
    build_summary_rows,
    build_wide_dice_rows,
    compute_forgetting_rows,
    evaluate_snapshots,
)

FORWARD_ORDER = ("46", "60", "86")
REVERSE_ORDER = tuple(reversed(FORWARD_ORDER))
CONFIDENCE_GATES = ((0.05, 0.95), (0.01, 0.99), (0.005, 0.995))
DISAGREEMENT_DELTAS: tuple[float | None, ...] = (None, 0.1, 0.2)
LAMBDA_ANCHOR = 0.1

BASELINE_REFERENCE = {
    "dice47": 0.74723,
    "dice70": 0.75811,
    "mean_unseen_final_dice": 0.75267,
    "max_forgetting": 0.02554,
    "pore_fraction_error": 0.01959,
}
FUNCTIONAL_0P1_REFERENCE = {"dice47": 0.7425749, "dice70": 0.7637043}

LOSS_SCALE_FIELDS = (
    "strategy",
    "stage_order",
    "stage_number",
    "stage_label",
    "scribble_slice",
    "confidence_low",
    "confidence_high",
    "disagreement_delta",
    "lambda_anchor",
    "fraction_source_confident",
    "fraction_anchor_active",
    "fraction_anchor_active_among_confident",
    "mean_kl_before_weighting",
    "weighted_anchor_loss",
    "scribble_loss",
    "ratio_anchor_scribble",
    "initial_scribble_loss",
    "final_scribble_loss",
    "stage_adaptation_time",
)
ORDER_FIELDS = (
    "strategy",
    "evaluation_slice",
    "forward_order",
    "reverse_order",
    "forward_final_dice",
    "reverse_final_dice",
    "reverse_minus_forward",
    "absolute_order_gap",
)
SUMMARY_FIELDS = (
    "strategy",
    "stage_order",
    "confidence_low",
    "confidence_high",
    "disagreement_delta",
    "lambda_anchor",
    "mean_final_dice",
    "mean_unseen_final_dice",
    "dice47",
    "dice70",
    "max_forgetting",
    "mean_forgetting",
    "pore_fraction_error",
    "order_gap47",
    "order_gap70",
    "max_unseen_order_gap",
    "mean_unseen_order_gap",
    "fraction_source_confident",
    "fraction_anchor_active",
    "fraction_anchor_active_among_confident",
    "mean_kl_before_weighting",
    "weighted_anchor_loss",
    "scribble_loss",
    "ratio_anchor_scribble",
    "total_adaptation_time",
    "reverse_adaptation_time",
)


@dataclass(frozen=True)
class FunctionalGatingExperiment:
    name: str
    confidence_low: float
    confidence_high: float
    disagreement_delta: float | None


def _slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_experiments() -> tuple[FunctionalGatingExperiment, ...]:
    """Return the locked 3x3 confidence/disagreement grid."""
    experiments = []
    for low, high in CONFIDENCE_GATES:
        for delta in DISAGREEMENT_DELTAS:
            delta_slug = "none" if delta is None else _slug(delta)
            experiments.append(
                FunctionalGatingExperiment(
                    name=(f"confidence_{_slug(low)}_{_slug(high)}_delta_{delta_slug}"),
                    confidence_low=low,
                    confidence_high=high,
                    disagreement_delta=delta,
                )
            )
    return tuple(experiments)


def apply_confidence_gate(
    auxiliary_sequences: Sequence[Sequence[FunctionalAnchorTile]],
    experiment: FunctionalGatingExperiment,
) -> tuple[tuple[FunctionalAnchorTile, ...], ...]:
    """Apply a tighter source-only confidence gate to cached teacher tiles."""
    gated_sequences = []
    for sequence in auxiliary_sequences:
        gated = []
        for tile in sequence:
            probability = tile.source_probability
            confidence = tile.confidence_mask & (
                (probability <= experiment.confidence_low)
                | (probability >= experiment.confidence_high)
            )
            gated.append(replace(tile, confidence_mask=confidence))
        gated_sequences.append(tuple(gated))
    return tuple(gated_sequences)


def _loss_scale_row(
    experiment: FunctionalGatingExperiment,
    stage_order: str,
    stage_number: int,
    scribble_slice: str,
    adaptation,
) -> dict[str, object]:
    history = adaptation.history
    confident = sum(step.functional_confident_pixels for step in history)
    active = sum(step.functional_active_pixels for step in history)
    mean_scribble = float(np.mean([step.scribble_loss for step in history]))
    mean_weighted_anchor = float(
        np.mean([step.weighted_functional_anchor_loss for step in history])
    )
    return {
        "strategy": experiment.name,
        "stage_order": stage_order,
        "stage_number": stage_number,
        "stage_label": f"after{scribble_slice}",
        "scribble_slice": scribble_slice,
        "confidence_low": experiment.confidence_low,
        "confidence_high": experiment.confidence_high,
        "disagreement_delta": (
            "none"
            if experiment.disagreement_delta is None
            else experiment.disagreement_delta
        ),
        "lambda_anchor": LAMBDA_ANCHOR,
        "fraction_source_confident": float(
            np.mean([step.functional_confident_fraction for step in history])
        ),
        "fraction_anchor_active": float(
            np.mean([step.functional_active_fraction for step in history])
        ),
        "fraction_anchor_active_among_confident": (
            active / confident if confident else 0.0
        ),
        "mean_kl_before_weighting": float(
            np.mean([step.functional_anchor_loss for step in history])
        ),
        "weighted_anchor_loss": mean_weighted_anchor,
        "scribble_loss": mean_scribble,
        "ratio_anchor_scribble": (
            mean_weighted_anchor / mean_scribble if mean_scribble else 0.0
        ),
        "initial_scribble_loss": history[0].scribble_loss,
        "final_scribble_loss": history[-1].scribble_loss,
        "stage_adaptation_time": adaptation.adaptation_time,
    }


def run_order(
    frozen,
    source_state: Mapping[str, torch.Tensor],
    experiment: FunctionalGatingExperiment,
    groups_by_slice: Mapping[str, ScribbleSliceTrainingTiles],
    auxiliary_sequences: Sequence[Sequence[FunctionalAnchorTile]],
    evaluation_contexts: Sequence[object],
    output_dir: Path,
    device: str,
    training_order: Sequence[str],
    *,
    infer_every_stage: bool,
) -> tuple[list[StageSnapshot], list[dict[str, object]], list[dict[str, object]]]:
    """Adapt in one order without accepting any GT path or mask."""
    frozen.model.load_state_dict(source_state)
    snapshots = []
    loss_scales = []
    transitions = []
    cumulative_time = 0.0
    trained: list[str] = []
    previous_end = state_dict_digest(frozen.model.state_dict())
    stage_order = "->".join(training_order)
    for stage_number, slice_id in enumerate(training_order, start=1):
        start_digest = state_dict_digest(frozen.model.state_dict())
        if start_digest != previous_end:
            raise RuntimeError(f"{experiment.name} reset weights between stages.")
        adaptation = adapt_from_scribbles_with_anchors(
            frozen,
            groups_by_slice[slice_id].tiles,
            FREEZE_POLICY,
            steps=STEPS_PER_STAGE,
            learning_rate=LEARNING_RATE,
            auxiliary_tiles=auxiliary_sequences[stage_number - 1],
            lambda_functional=LAMBDA_ANCHOR,
            functional_disagreement_delta=experiment.disagreement_delta,
            scribble_slice_id=slice_id,
            random_state=RANDOM_STATE,
        )
        cumulative_time += adaptation.adaptation_time
        end_digest = state_dict_digest(frozen.model.state_dict())
        previous_end = end_digest
        trained.append(slice_id)
        stage_label = f"after{slice_id}"
        stage_dir = (
            output_dir
            / "strategies"
            / experiment.name
            / f"stage_{stage_number}_{stage_label}"
        )
        _save_history(stage_dir / "history.csv", (adaptation,))
        loss_scales.append(
            _loss_scale_row(experiment, stage_order, stage_number, slice_id, adaptation)
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
        if infer_every_stage or stage_number == len(training_order):
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
                    order=stage_order,
                    stage_number=stage_number,
                    stage_label=stage_label,
                    training_slice_ids=tuple(trained),
                    stage_adaptation_time=adaptation.adaptation_time,
                    cumulative_adaptation_time=cumulative_time,
                    probability_dir=probability_dir,
                    inference_times=inference_times,
                )
            )
    return snapshots, loss_scales, transitions


def _build_order_rows(
    forward_rows: Sequence[Mapping[str, object]],
    reverse_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    strategies = sorted({str(row["strategy"]) for row in forward_rows})
    for strategy in strategies:
        forward_final = {
            str(row["evaluation_slice"]): row
            for row in forward_rows
            if row["strategy"] == strategy and int(row["stage_number"]) == 3
        }
        reverse_final = {
            str(row["evaluation_slice"]): row
            for row in reverse_rows
            if row["strategy"] == strategy and int(row["stage_number"]) == 3
        }
        for slice_id in sorted(forward_final, key=int):
            forward_dice = float(forward_final[slice_id]["dice"])
            reverse_dice = float(reverse_final[slice_id]["dice"])
            difference = reverse_dice - forward_dice
            rows.append(
                {
                    "strategy": strategy,
                    "evaluation_slice": slice_id,
                    "forward_order": "->".join(FORWARD_ORDER),
                    "reverse_order": "->".join(REVERSE_ORDER),
                    "forward_final_dice": forward_dice,
                    "reverse_final_dice": reverse_dice,
                    "reverse_minus_forward": difference,
                    "absolute_order_gap": abs(difference),
                }
            )
    return rows


def _mean_loss_values(
    loss_rows: Sequence[Mapping[str, object]], strategy: str, stage_order: str
) -> dict[str, float]:
    selected = [
        row
        for row in loss_rows
        if row["strategy"] == strategy and row["stage_order"] == stage_order
    ]
    fields = (
        "fraction_source_confident",
        "fraction_anchor_active",
        "fraction_anchor_active_among_confident",
        "mean_kl_before_weighting",
        "weighted_anchor_loss",
        "scribble_loss",
    )
    values = {
        field: float(np.mean([float(row[field]) for row in selected]))
        for field in fields
    }
    values["ratio_anchor_scribble"] = (
        values["weighted_anchor_loss"] / values["scribble_loss"]
        if values["scribble_loss"]
        else 0.0
    )
    return values


def _augment_summaries(
    summaries: list[dict[str, object]],
    forward_rows: Sequence[Mapping[str, object]],
    order_rows: Sequence[Mapping[str, object]],
    loss_rows: Sequence[Mapping[str, object]],
    experiments: Sequence[FunctionalGatingExperiment],
) -> None:
    by_name = {experiment.name: experiment for experiment in experiments}
    forward_order = "->".join(FORWARD_ORDER)
    reverse_order = "->".join(REVERSE_ORDER)
    for summary in summaries:
        strategy = str(summary["strategy"])
        experiment = by_name[strategy]
        final = {
            str(row["evaluation_slice"]): row
            for row in forward_rows
            if row["strategy"] == strategy and int(row["stage_number"]) == 3
        }
        gaps = {
            str(row["evaluation_slice"]): float(row["absolute_order_gap"])
            for row in order_rows
            if row["strategy"] == strategy
        }
        loss_values = _mean_loss_values(loss_rows, strategy, forward_order)
        reverse_loss_rows = [
            row
            for row in loss_rows
            if row["strategy"] == strategy and row["stage_order"] == reverse_order
        ]
        summary.update(
            {
                "confidence_low": experiment.confidence_low,
                "confidence_high": experiment.confidence_high,
                "disagreement_delta": (
                    "none"
                    if experiment.disagreement_delta is None
                    else experiment.disagreement_delta
                ),
                "lambda_anchor": LAMBDA_ANCHOR,
                "dice47": final["47"]["dice"],
                "dice70": final["70"]["dice"],
                "order_gap47": gaps["47"],
                "order_gap70": gaps["70"],
                "max_unseen_order_gap": max(gaps["47"], gaps["70"]),
                "mean_unseen_order_gap": float(np.mean([gaps["47"], gaps["70"]])),
                "reverse_adaptation_time": float(
                    sum(
                        float(row["stage_adaptation_time"]) for row in reverse_loss_rows
                    )
                ),
                **loss_values,
            }
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _audit_existing_artifacts(output_dir: Path) -> dict[str, object]:
    forward_histories = tuple(
        (output_dir / "forward" / "strategies").rglob("history.csv")
    )
    reverse_histories = tuple(
        (output_dir / "reverse" / "strategies").rglob("history.csv")
    )
    history_lengths = {
        str(path.relative_to(output_dir)): len(_read_csv(path))
        for path in (*forward_histories, *reverse_histories)
    }
    counts = {
        "forward_metric_rows": len(_read_csv(output_dir / "results_per_stage.csv")),
        "reverse_metric_rows": len(_read_csv(output_dir / "reverse_final.csv")),
        "loss_scale_rows": len(_read_csv(output_dir / "loss_scales.csv")),
        "forward_histories": len(forward_histories),
        "reverse_histories": len(reverse_histories),
        "forward_probability_maps": len(
            tuple((output_dir / "forward").rglob("probabilities/*.npy"))
        ),
        "reverse_probability_maps": len(
            tuple((output_dir / "reverse").rglob("probabilities/*.npy"))
        ),
    }
    expected = {
        "forward_metric_rows": 135,
        "reverse_metric_rows": 45,
        "loss_scale_rows": 54,
        "forward_histories": 27,
        "reverse_histories": 27,
        "forward_probability_maps": 135,
        "reverse_probability_maps": 45,
    }
    if counts != expected or set(history_lengths.values()) != {STEPS_PER_STAGE}:
        raise RuntimeError(
            f"Incomplete functional-gating artifacts: counts={counts}, "
            f"history lengths={set(history_lengths.values())}."
        )
    return {
        "counts": counts,
        "expected_counts": expected,
        "all_histories_have_25_steps": True,
    }


def finalize_existing(
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Finish aggregation from a complete prediction run without adaptation."""
    artifact_audit = _audit_existing_artifacts(args.output_dir)
    experiments = build_experiments()
    forward_rows = _read_csv(args.output_dir / "results_per_stage.csv")
    forgetting_rows = _read_csv(args.output_dir / "forgetting.csv")
    order_rows = _read_csv(args.output_dir / "order_sensitivity.csv")
    loss_scales = _read_csv(args.output_dir / "loss_scales.csv")
    summary_rows = build_summary_rows(forward_rows, forgetting_rows)
    _augment_summaries(summary_rows, forward_rows, order_rows, loss_scales, experiments)
    summary_rows.sort(
        key=lambda row: (
            float(row["max_unseen_order_gap"]),
            -float(row["mean_unseen_final_dice"]),
        )
    )
    _write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)

    checkpoint_before = file_digest(args.checkpoint)
    contexts = discover_stack_contexts(args.target_dir)
    evaluations = discover_evaluation_inputs(contexts, args.ideal_dir)
    scribble_inputs = resolve_scribble_inputs(contexts, args.scribble_dir)
    representative_ids = select_representative_slice_ids(tuple(contexts))
    frozen = load_frozen_fibnet(args.checkpoint, device=args.device)
    source_state = _clone_state(frozen.model)
    source_state_sha256 = state_dict_digest(source_state)
    scribbles, tile_groups, scribble_stats = _load_scribbles_and_tiles(
        frozen, scribble_inputs
    )
    ground_truth = _load_ground_truth_after_predictions(evaluations)
    _add_scribble_purity(scribble_stats, scribbles, ground_truth)
    _write_csv(
        args.output_dir / "scribble_stats.csv",
        tuple(scribble_stats[0]),
        scribble_stats,
    )
    checkpoint_after = file_digest(args.checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("Source checkpoint changed during finalization.")
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
        "optimizer": "Adam, reset at each stage",
        "steps_per_stage": STEPS_PER_STAGE,
        "lambda_consistency": LAMBDA_CONSISTENCY,
        "lambda_anchor": LAMBDA_ANCHOR,
        "threshold": THRESHOLD,
        "forward_order": list(FORWARD_ORDER),
        "reverse_order": list(REVERSE_ORDER),
        "reverse_inference": "final stage only, for order-gap measurement",
        "representative_slices": list(representative_ids),
        "requested_representative_slices": list(REQUESTED_REPRESENTATIVE_SLICES),
        "excluded_auxiliary_slices": list(FORWARD_ORDER),
        "functional_anchor": "Bernoulli KL(source || adapted)",
        "disagreement_gate": (
            "source-confident AND abs(p_adapted.detach() - p_source) >= delta; "
            "none means all source-confident pixels"
        ),
        "fraction_denominators": {
            "fraction_source_confident": "all tile pixels",
            "fraction_anchor_active": "all tile pixels",
            "fraction_anchor_active_among_confident": "source-confident pixels",
        },
        "ground_truth_used_for": "evaluation only after all adaptation/inference",
        "forgetting_definition": "best prior-stage Dice minus final Dice",
        "order_gap_definition": "absolute reverse-final minus forward-final Dice",
        "experiments": [experiment.__dict__ for experiment in experiments],
        "baseline_reference": BASELINE_REFERENCE,
        "functional_anchor_0p1_reference": FUNCTIONAL_0P1_REFERENCE,
        "artifact_audit": artifact_audit,
        "transition_continuity": (
            "validated online by state-digest assertions before every stage; "
            "per-transition hashes were not retained after a summary-schema error"
        ),
        "finalized_from_existing_complete_predictions": True,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    del source_state, tile_groups, ground_truth, scribbles
    gc.collect()
    return forward_rows, summary_rows


def run(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments()
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
    base_auxiliary = prepare_functional_anchor_sequences(
        frozen, contexts, representative_ids
    )

    forward_snapshots = []
    reverse_snapshots = []
    loss_scales = []
    transition_audit = {}
    for index, experiment in enumerate(experiments, start=1):
        print(f"experiment {index}/{len(experiments)}: {experiment.name}", flush=True)
        auxiliary = apply_confidence_gate(base_auxiliary, experiment)
        forward, scales, transitions = run_order(
            frozen,
            source_state,
            experiment,
            groups_by_slice,
            auxiliary,
            evaluation_contexts,
            args.output_dir / "forward",
            args.device,
            FORWARD_ORDER,
            infer_every_stage=True,
        )
        forward_snapshots.extend(forward)
        loss_scales.extend(scales)
        reverse, scales, reverse_transitions = run_order(
            frozen,
            source_state,
            experiment,
            groups_by_slice,
            auxiliary,
            evaluation_contexts,
            args.output_dir / "reverse",
            args.device,
            REVERSE_ORDER,
            infer_every_stage=False,
        )
        reverse_snapshots.extend(reverse)
        loss_scales.extend(scales)
        transition_audit[experiment.name] = {
            "forward": transitions,
            "reverse": reverse_transitions,
        }
        if frozen.device.type == "cuda":
            torch.cuda.empty_cache()

    # Hard GT boundary: every model update and prediction above is complete.
    ground_truth = _load_ground_truth_after_predictions(evaluations)
    _add_scribble_purity(scribble_stats, scribbles, ground_truth)
    forward_rows = evaluate_snapshots(forward_snapshots, ground_truth)
    reverse_rows = evaluate_snapshots(reverse_snapshots, ground_truth)
    forgetting_rows = compute_forgetting_rows(forward_rows)
    summary_rows = build_summary_rows(forward_rows, forgetting_rows)
    order_rows = _build_order_rows(forward_rows, reverse_rows)
    _augment_summaries(summary_rows, forward_rows, order_rows, loss_scales, experiments)
    summary_rows.sort(
        key=lambda row: (
            float(row["max_unseen_order_gap"]),
            -float(row["mean_unseen_final_dice"]),
        )
    )

    _write_csv(
        args.output_dir / "results_per_stage.csv", PER_STAGE_FIELDS, forward_rows
    )
    _write_csv(args.output_dir / "reverse_final.csv", PER_STAGE_FIELDS, reverse_rows)
    _write_csv(
        args.output_dir / "dice_by_stage.csv",
        WIDE_FIELDS,
        build_wide_dice_rows(forward_rows),
    )
    _write_csv(args.output_dir / "forgetting.csv", FORGETTING_FIELDS, forgetting_rows)
    _write_csv(args.output_dir / "order_sensitivity.csv", ORDER_FIELDS, order_rows)
    _write_csv(args.output_dir / "loss_scales.csv", LOSS_SCALE_FIELDS, loss_scales)
    _write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
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
        "optimizer": "Adam, reset at each stage",
        "steps_per_stage": STEPS_PER_STAGE,
        "lambda_consistency": LAMBDA_CONSISTENCY,
        "lambda_anchor": LAMBDA_ANCHOR,
        "threshold": THRESHOLD,
        "forward_order": list(FORWARD_ORDER),
        "reverse_order": list(REVERSE_ORDER),
        "reverse_inference": "final stage only, for order-gap measurement",
        "representative_slices": list(representative_ids),
        "requested_representative_slices": list(REQUESTED_REPRESENTATIVE_SLICES),
        "excluded_auxiliary_slices": list(FORWARD_ORDER),
        "functional_anchor": "Bernoulli KL(source || adapted)",
        "disagreement_gate": (
            "source-confident AND abs(p_adapted.detach() - p_source) >= delta; "
            "none means all source-confident pixels"
        ),
        "fraction_denominators": {
            "fraction_source_confident": "all tile pixels",
            "fraction_anchor_active": "all tile pixels",
            "fraction_anchor_active_among_confident": "source-confident pixels",
        },
        "ground_truth_used_for": "evaluation only after all adaptation/inference",
        "forgetting_definition": "best prior-stage Dice minus final Dice",
        "order_gap_definition": "absolute reverse-final minus forward-final Dice",
        "experiments": [experiment.__dict__ for experiment in experiments],
        "baseline_reference": BASELINE_REFERENCE,
        "functional_anchor_0p1_reference": FUNCTIONAL_0P1_REFERENCE,
        "transition_audit": transition_audit,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    del source_state, tile_groups, base_auxiliary, ground_truth, scribbles
    gc.collect()
    return forward_rows, summary_rows


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
        default=Path("outputs/functional_gating_adaptation"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Rebuild summary/metadata from an already complete prediction run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.finalize_only:
        _results, summary = finalize_existing(args)
    else:
        _results, summary = run(args)
    print("\nFinal ranking (smallest unseen order gap first):")
    for row in summary:
        print(
            f"{row['strategy']}: unseen={float(row['mean_unseen_final_dice']):.5f} "
            f"D47={float(row['dice47']):.5f} D70={float(row['dice70']):.5f} "
            f"forget={float(row['max_forgetting']):.5f} "
            f"order_gap={float(row['max_unseen_order_gap']):.5f} "
            f"pore_error={float(row['pore_fraction_error']):.5f}"
        )
    print(f"outputs={args.output_dir}")


if __name__ == "__main__":
    main()
