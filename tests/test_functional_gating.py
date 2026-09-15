from __future__ import annotations

import inspect

import torch

from fibnet.interactive import FunctionalAnchorTile
from scripts.benchmark_functional_gating import (
    CONFIDENCE_GATES,
    DISAGREEMENT_DELTAS,
    LAMBDA_ANCHOR,
    apply_confidence_gate,
    build_experiments,
    run_order,
)


def test_functional_gating_grid_matches_requested_values() -> None:
    experiments = build_experiments()

    assert len(experiments) == 9
    assert {(item.confidence_low, item.confidence_high) for item in experiments} == set(
        CONFIDENCE_GATES
    )
    assert {item.disagreement_delta for item in experiments} == set(DISAGREEMENT_DELTAS)
    assert LAMBDA_ANCHOR == 0.1


def test_tighter_confidence_gate_uses_only_source_probabilities() -> None:
    probability = torch.tensor([[0.001, 0.007, 0.02, 0.98, 0.993, 0.999]])
    tile = FunctionalAnchorTile(
        tensor=torch.zeros((1, 1, 1, 6)),
        source_probability=probability,
        confidence_mask=torch.tensor([[True, True, False, True, True, False]]),
        slice_id="10",
        tile_index=0,
        x=0,
        y=0,
    )
    experiment = next(
        item
        for item in build_experiments()
        if item.confidence_low == 0.005 and item.disagreement_delta is None
    )

    ((gated,),) = apply_confidence_gate(((tile,),), experiment)

    torch.testing.assert_close(
        gated.confidence_mask,
        torch.tensor([[True, False, False, False, False, False]]),
    )


def test_functional_gating_runner_has_no_gt_input() -> None:
    parameters = inspect.signature(run_order).parameters

    assert "ground_truth" not in parameters
    assert "mask" not in parameters
    assert "ideal" not in parameters
