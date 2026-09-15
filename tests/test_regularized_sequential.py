from __future__ import annotations

import inspect

import numpy as np

from scripts.benchmark_regularized_sequential import (
    FUNCTIONAL_LAMBDAS,
    REQUESTED_REPRESENTATIVE_SLICES,
    TRAINING_ORDER,
    WEIGHT_LAMBDAS,
    build_auxiliary_schedule,
    candidate_experiments,
    run_experiment,
    select_representative_slice_ids,
)


def test_candidate_grid_matches_requested_coefficients() -> None:
    experiments = candidate_experiments()
    weight_values = tuple(
        experiment.lambda_weight
        for experiment in experiments
        if experiment.lambda_anchor == 0.0
    )
    functional_values = tuple(
        experiment.lambda_anchor
        for experiment in experiments
        if experiment.lambda_anchor > 0.0
    )

    assert weight_values == WEIGHT_LAMBDAS
    assert functional_values == FUNCTIONAL_LAMBDAS


def test_representative_grid_is_uniform_and_excludes_scribble_slices() -> None:
    available = tuple(str(index) for index in range(137))

    selected = select_representative_slice_ids(available)

    assert selected == REQUESTED_REPRESENTATIVE_SLICES
    assert not set(selected) & set(TRAINING_ORDER)


def test_auxiliary_schedule_balances_z_before_spatial_tiles() -> None:
    slice_ids = ("10", "20", "30", "40")
    tile_counts = {slice_id: index + 2 for index, slice_id in enumerate(slice_ids)}

    schedule = build_auxiliary_schedule(
        slice_ids, tile_counts, steps=25, random_state=42
    )

    counts = {slice_id: 0 for slice_id in slice_ids}
    for slice_id, tile_index in schedule:
        counts[slice_id] += 1
        assert 0 <= tile_index < tile_counts[slice_id]
    assert max(counts.values()) - min(counts.values()) <= 1
    np.testing.assert_array_equal(sorted(counts.values()), [6, 6, 6, 7])


def test_regularized_adaptation_runner_has_no_gt_input() -> None:
    parameters = inspect.signature(run_experiment).parameters

    assert "ground_truth" not in parameters
    assert "mask" not in parameters
    assert "ideal" not in parameters
