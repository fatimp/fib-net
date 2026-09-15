from __future__ import annotations

import numpy as np
import pytest

from fibnet.interactive import INTERACTIVE_ADAPTATION_BASELINE
from fibnet.interactive.io import PORE, SOLID
from scripts.benchmark_iterative_scribble_refinement import (
    validate_cumulative_scribbles,
)


def test_interactive_adaptation_baseline_is_locked() -> None:
    config = INTERACTIVE_ADAPTATION_BASELINE

    assert config.mode == "last_decoder_head"
    assert config.lambda_consistency == 0.0
    assert config.optimizer == "Adam"
    assert config.learning_rate == 1e-4
    assert config.steps == 50
    assert config.threshold == 0.5


def test_cumulative_round2_preserves_round1_and_adds_labels() -> None:
    round1 = np.zeros((8, 9), dtype=np.uint8)
    round1[1, 1] = PORE
    round1[6, 7] = SOLID
    round2 = round1.copy()
    round2[2, 3] = SOLID
    round2[5, 4] = PORE

    validated = validate_cumulative_scribbles(round1, round2)

    np.testing.assert_array_equal(validated, round2)


@pytest.mark.parametrize("change", ["removed", "relabeled", "no_new_labels"])
def test_invalid_cumulative_round2_is_rejected(change: str) -> None:
    round1 = np.zeros((6, 7), dtype=np.uint8)
    round1[1, 1] = PORE
    round1[4, 5] = SOLID
    round2 = round1.copy()
    if change == "removed":
        round2[1, 1] = 0
        round2[2, 2] = PORE
    elif change == "relabeled":
        round2[1, 1] = SOLID
        round2[2, 2] = PORE

    with pytest.raises(ValueError, match="cumulative|no new"):
        validate_cumulative_scribbles(round1, round2)
