from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

from scripts.benchmark_sequential_strategies import (
    StagePlan,
    build_order_sensitivity_rows,
    build_strategy_plans,
    build_summary_rows,
    build_wide_dice_rows,
    compute_forgetting_rows,
    run_staged_strategy,
)


def _row(
    strategy: str,
    order: str,
    stage: int,
    label: str,
    slice_id: str,
    dice: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "stage_order": order,
        "stage_number": stage,
        "stage_label": label,
        "evaluation_slice": slice_id,
        "seen_unseen": "unseen",
        "dice": dice,
        "absolute_pore_fraction_error": 0.02,
        "cumulative_adaptation_time": float(stage),
    }


def test_strategy_plans_encode_requested_replay_and_new_slice_weights() -> None:
    plans = {plan.name: plan for plan in build_strategy_plans()}

    assert plans["sequential_current_only"].order == "46->60->86"
    assert plans["sequential_current_only_reverse"].order == "86->60->46"
    assert plans["cumulative_sequential"].stages[-1] == StagePlan(
        "after86", ("46", "60", "86"), (1.0, 1.0, 1.0)
    )
    assert plans["replay_50"].stages[1].slice_weights == (1.0, 1.0)
    assert plans["replay_50"].stages[2].slice_weights == (1.0, 1.0, 2.0)
    assert plans["cumulative_new_x2"].stages[1].slice_weights == (1.0, 2.0)
    assert plans["cumulative_new_x2"].stages[2].slice_weights == (1.0, 1.0, 2.0)
    assert "mask" not in {field.name for field in fields(StagePlan)}
    assert "ground_truth" not in inspect.signature(run_staged_strategy).parameters


def test_wide_dice_table_maps_reverse_stages_by_actual_completed_slice() -> None:
    rows = [
        _row("reverse", "86->60->46", 1, "after86", "47", 0.50),
        _row("reverse", "86->60->46", 2, "after60", "47", 0.60),
        _row("reverse", "86->60->46", 3, "after46", "47", 0.70),
    ]

    wide = build_wide_dice_rows(rows)

    assert wide == [
        {
            "strategy": "reverse",
            "stage_order": "86->60->46",
            "evaluation_slice": "47",
            "after46": 0.70,
            "after60": 0.60,
            "after86": 0.50,
        }
    ]


def test_forgetting_uses_best_prior_stage_and_summary_uses_final_stage() -> None:
    rows = [
        _row("strategy", "46->60->86", 1, "after46", "47", 0.80),
        _row("strategy", "46->60->86", 2, "after60", "47", 0.75),
        _row("strategy", "46->60->86", 3, "after86", "47", 0.70),
        _row("strategy", "46->60->86", 1, "after46", "70", 0.60),
        _row("strategy", "46->60->86", 2, "after60", "70", 0.70),
        _row("strategy", "46->60->86", 3, "after86", "70", 0.65),
    ]

    forgetting = compute_forgetting_rows(rows)
    summary = build_summary_rows(rows, forgetting)

    assert [row["best_previous_stage"] for row in forgetting] == [
        "after46",
        "after60",
    ]
    assert [row["forgetting"] for row in forgetting] == pytest.approx([0.10, 0.05])
    assert summary[0]["mean_final_dice"] == pytest.approx(0.675)
    assert summary[0]["mean_unseen_final_dice"] == pytest.approx(0.675)
    assert summary[0]["max_forgetting"] == pytest.approx(0.10)
    assert summary[0]["mean_forgetting"] == pytest.approx(0.075)


def test_order_sensitivity_compares_final_chronological_stage() -> None:
    rows = [
        _row("sequential_current_only", "46->60->86", 3, "after86", "47", 0.70),
        _row("sequential_current_only", "46->60->86", 3, "after86", "70", 0.80),
        _row(
            "sequential_current_only_reverse",
            "86->60->46",
            3,
            "after46",
            "47",
            0.72,
        ),
        _row(
            "sequential_current_only_reverse",
            "86->60->46",
            3,
            "after46",
            "70",
            0.79,
        ),
    ]

    sensitivity = build_order_sensitivity_rows(rows)

    assert sensitivity[0]["reverse_minus_forward"] == pytest.approx(0.02)
    assert sensitivity[0]["substantial_at_0p01"] is True
    assert sensitivity[1]["absolute_difference"] == pytest.approx(0.01)
