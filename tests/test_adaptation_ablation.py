from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fibnet.interactive.adaptation import ScribbleTrainingTile
from fibnet.interactive.encoder import FrozenFibNet
from fibnet.interactive.io import PORE, SOLID
from fibnet.model import ResUNet
from scripts.benchmark_adaptation_ablation import (
    METRIC_FIELDS,
    RESULT_FIELDS,
    SUMMARY_FIELDS,
    _result_row,
    configurations,
    file_digest,
    rank_summary_rows,
    run_sequential_configuration,
    state_dict_digest,
)


def _tile(input_tensor: torch.Tensor, pore: tuple[int, int]) -> ScribbleTrainingTile:
    labels = torch.zeros((16, 16), dtype=torch.uint8)
    labels[pore] = PORE
    labels[14 - pore[0], 14 - pore[1]] = SOLID
    return ScribbleTrainingTile(
        tensor=input_tensor,
        labels=labels,
        valid_mask=torch.ones((16, 16), dtype=torch.bool),
        source_probability=torch.zeros((16, 16)),
        x=0,
        y=0,
    )


def test_ablation_grid_contains_exactly_27_configurations() -> None:
    grid = configurations()
    assert len(grid) == 27
    assert len(set(grid)) == 27
    assert {policy for policy, _, _ in grid} == {
        "last_decoder_head",
        "last2_decoder_head",
        "full_decoder",
    }


def test_round2_starts_from_round1_and_checkpoint_is_unchanged(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    model = ResUNet(in_channels=6, features=(4, 8, 16, 32)).eval()
    checkpoint = tmp_path / "source.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    checkpoint_before = file_digest(checkpoint)
    frozen = FrozenFibNet(
        model=model,
        checkpoint=checkpoint,
        feature_mode="stack_relief",
        model_arch="resunet",
        image_size=16,
        device=torch.device("cpu"),
    )
    source_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    source_digest = state_dict_digest(source_state)
    input_tensor = torch.rand(1, 6, 16, 16)

    result = run_sequential_configuration(
        frozen,
        source_state,
        (_tile(input_tensor, (2, 3)),),
        (_tile(input_tensor, (5, 6)),),
        "head_only",
        1e-3,
        1,
        lambda _model: (np.zeros((16, 16), dtype=np.float32), 0.0),
    )

    assert result.round1_state_digest == result.round2_start_state_digest
    assert state_dict_digest(source_state) == source_digest
    assert state_dict_digest(model.state_dict()) != source_digest
    assert file_digest(checkpoint) == checkpoint_before


def test_results_schema_contains_metrics_and_adaptation_fields() -> None:
    metrics = {name: 0.5 for name in METRIC_FIELDS}
    row = _result_row(
        "config",
        "last_decoder_head",
        1e-4,
        50,
        "source",
        metrics,
        None,
        1.0,
        0.0,
        total_parameters=123,
    )

    assert tuple(row) == RESULT_FIELDS
    assert "round2_dice" in SUMMARY_FIELDS
    assert "round2_iou" in SUMMARY_FIELDS
    assert "round2_absolute_pore_fraction_error" in SUMMARY_FIELDS


def test_ranking_is_deterministic_and_uses_requested_tiebreaks() -> None:
    rows = [
        {
            "config_id": "b",
            "round2_dice": 0.8,
            "round2_iou": 0.7,
            "round2_absolute_pore_fraction_error": 0.01,
        },
        {
            "config_id": "a",
            "round2_dice": 0.8,
            "round2_iou": 0.7,
            "round2_absolute_pore_fraction_error": 0.01,
        },
        {
            "config_id": "better_iou",
            "round2_dice": 0.8,
            "round2_iou": 0.71,
            "round2_absolute_pore_fraction_error": 0.02,
        },
        {
            "config_id": "best_dice",
            "round2_dice": 0.81,
            "round2_iou": 0.6,
            "round2_absolute_pore_fraction_error": 0.1,
        },
    ]

    first = rank_summary_rows(rows)
    second = rank_summary_rows(list(reversed(rows)))

    assert [row["config_id"] for row in first] == [
        "best_dice",
        "better_iou",
        "a",
        "b",
    ]
    assert first == second
