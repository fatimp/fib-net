from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fibnet.interactive import (
    PORE,
    SOLID,
    ScribbleSliceTrainingTiles,
    ScribbleTrainingTile,
    adapt_from_multislice_scribbles,
    build_multislice_tile_schedule,
)
from fibnet.interactive.encoder import FrozenFibNet
from fibnet.model import ResUNet
from scripts.benchmark_multislice_adaptation import (
    ScribbleInput,
    classify_seen,
    discover_evaluation_inputs,
    discover_stack_contexts,
    file_digest,
    rank_multislice,
    resolve_scribble_inputs,
    run_sequential_adaptation,
)


def _tile(seed: int) -> ScribbleTrainingTile:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.zeros((16, 16), dtype=torch.uint8)
    labels[2 + seed % 3, 3] = PORE
    labels[11, 10 + seed % 3] = SOLID
    return ScribbleTrainingTile(
        tensor=torch.rand((1, 6, 16, 16), generator=generator),
        labels=labels,
        valid_mask=torch.ones((16, 16), dtype=torch.bool),
        source_probability=torch.zeros((16, 16)),
        x=seed,
        y=0,
    )


def _groups(counts: tuple[int, ...]) -> tuple[ScribbleSliceTrainingTiles, ...]:
    return tuple(
        ScribbleSliceTrainingTiles(
            slice_id=str(40 + group_index),
            tiles=tuple(_tile(10 * group_index + index) for index in range(count)),
        )
        for group_index, count in enumerate(counts)
    )


def _frozen(tmp_path: Path) -> FrozenFibNet:
    return FrozenFibNet(
        model=ResUNet(in_channels=6, features=(4, 8, 16, 32)).eval(),
        checkpoint=tmp_path / "source.pt",
        feature_mode="stack_relief",
        model_arch="resunet",
        image_size=16,
        device=torch.device("cpu"),
    )


def test_multislice_paths_associate_image_context_and_scribbles(tmp_path: Path) -> None:
    target_dir = tmp_path / "target"
    ideal_dir = tmp_path / "ideal"
    scribble_dir = tmp_path / "scribbles"
    target_dir.mkdir()
    ideal_dir.mkdir()
    scribble_dir.mkdir()
    for slice_id in (45, 46, 47, 59, 60, 61, 85, 86, 87):
        Image.new("L", (8, 6), slice_id).save(target_dir / f"{slice_id}.tiff")
    for name in ("46.tif", "47.tiff", "60.tif", "86.tiff"):
        Image.new("L", (8, 6), 255).save(ideal_dir / name)
    for slice_id in (46, 60, 86):
        Image.new("RGBA", (8, 6), (0, 0, 0, 0)).save(
            scribble_dir / f"{slice_id}_scribbles.png"
        )

    contexts = discover_stack_contexts(target_dir)
    evaluations = discover_evaluation_inputs(contexts, ideal_dir)
    scribbles = resolve_scribble_inputs(contexts, scribble_dir)

    assert [item.context.slice_id for item in evaluations] == ["46", "47", "60", "86"]
    assert [item.context.previous_path.stem for item in scribbles] == ["45", "59", "85"]
    assert [item.context.next_path.stem for item in scribbles] == ["47", "61", "87"]
    assert [item.scribble_path.stem for item in scribbles] == [
        "46_scribbles",
        "60_scribbles",
        "86_scribbles",
    ]
    assert "mask" not in {field.name for field in fields(ScribbleInput)}


def test_slice_balanced_schedule_really_balances_slices() -> None:
    groups = _groups((1, 5, 2))
    balanced = build_multislice_tile_schedule(
        groups, 30, "joint_slice_balanced", random_state=7
    )
    uniform = build_multislice_tile_schedule(
        groups, 8, "joint_pixel_uniform", random_state=7
    )

    balanced_counts = np.bincount([item[0] for item in balanced], minlength=3)
    uniform_counts = np.bincount([item[0] for item in uniform], minlength=3)
    np.testing.assert_array_equal(balanced_counts, [10, 10, 10])
    np.testing.assert_array_equal(uniform_counts, [1, 5, 2])


def test_weighted_slice_schedule_uses_stage_weights_not_labeled_pixels() -> None:
    groups = _groups((1, 9, 3))
    two_slice = build_multislice_tile_schedule(
        groups[:2],
        25,
        "joint_slice_balanced",
        slice_weights=(1, 2),
        random_state=42,
    )
    three_slice = build_multislice_tile_schedule(
        groups,
        25,
        "joint_slice_balanced",
        slice_weights=(1, 1, 2),
        random_state=42,
    )

    np.testing.assert_array_equal(
        np.bincount([item[0] for item in two_slice], minlength=2), [8, 17]
    )
    np.testing.assert_array_equal(
        np.bincount([item[0] for item in three_slice], minlength=3), [6, 6, 13]
    )


def test_joint_adaptation_updates_one_shared_model(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    groups = _groups((1, 1))
    model_id = id(frozen.model)
    before = {
        name: value.detach().clone()
        for name, value in frozen.model.state_dict().items()
    }

    result = adapt_from_multislice_scribbles(
        frozen,
        groups,
        "full_decoder",
        0.0,
        steps=2,
        learning_rate=3e-5,
        sampling_strategy="joint_slice_balanced",
        random_state=42,
    )

    assert id(frozen.model) == model_id
    assert {step.slice_id for step in result.history} == {"40", "41"}
    assert any(
        not torch.equal(before[name], value)
        for name, value in frozen.model.state_dict().items()
        if name.startswith(("up_blocks.", "head."))
    )


def test_sequential_adaptation_continues_previous_weights_and_not_checkpoint(
    tmp_path: Path,
) -> None:
    frozen = _frozen(tmp_path)
    frozen.checkpoint.write_bytes(b"immutable-source-checkpoint")
    checkpoint_before = file_digest(frozen.checkpoint)
    source_digest = {
        name: value.detach().clone()
        for name, value in frozen.model.state_dict().items()
    }

    histories, transitions = run_sequential_adaptation(
        frozen, _groups((1, 1)), steps_per_slice=1
    )

    assert len(histories) == 2
    assert transitions[0]["end_state_sha256"] == transitions[1]["start_state_sha256"]
    assert transitions[0]["start_state_sha256"] != transitions[0]["end_state_sha256"]
    assert transitions[1]["start_state_sha256"] != transitions[1]["end_state_sha256"]
    assert any(
        not torch.equal(source_digest[name], value)
        for name, value in frozen.model.state_dict().items()
    )
    assert file_digest(frozen.checkpoint) == checkpoint_before


def test_seen_classification_and_multislice_ranking_prioritize_unseen() -> None:
    assert classify_seen("60", ("60",)) == "seen"
    assert classify_seen("46", ("60",)) == "unseen"
    assert classify_seen("60", ()) == "unseen"
    rows = [
        {
            "regime": "higher_all",
            "number_training_slices": 3,
            "mean_unseen_dice": 0.70,
            "mean_all_dice": 0.90,
            "mean_unseen_pore_fraction_error": 0.01,
        },
        {
            "regime": "higher_unseen",
            "number_training_slices": 3,
            "mean_unseen_dice": 0.71,
            "mean_all_dice": 0.80,
            "mean_unseen_pore_fraction_error": 0.02,
        },
        {
            "regime": "single",
            "number_training_slices": 1,
            "mean_unseen_dice": 0.99,
            "mean_all_dice": 0.99,
            "mean_unseen_pore_fraction_error": 0.0,
        },
    ]

    assert rank_multislice(rows) == ["higher_unseen", "higher_all"]
