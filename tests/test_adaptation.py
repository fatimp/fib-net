from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fibnet.interactive.adaptation import (
    FunctionalAnchorTile,
    ScribbleTrainingTile,
    adapt_from_scribbles,
    adapt_from_scribbles_with_anchors,
    confidence_gated_bernoulli_kl,
    configure_trainable_parameters,
    describe_freeze_policy,
    sparse_adaptation_loss,
    weight_anchor_parameter_mse,
)
from fibnet.interactive.encoder import FrozenFibNet
from fibnet.interactive.io import PORE, SOLID
from fibnet.model import ResUNet


def test_sparse_loss_uses_scribbles_and_consistency_uses_only_unlabeled() -> None:
    logits = torch.tensor([[0.5, -0.4], [0.8, -1.2]], requires_grad=True)
    labels = torch.tensor([[PORE, 0], [0, SOLID]])
    valid = torch.ones((2, 2), dtype=torch.bool)
    teacher = torch.tensor([[0.0, 0.2], [0.7, 1.0]])
    changed_at_scribbles = teacher.clone()
    changed_at_scribbles[0, 0] = 1.0
    changed_at_scribbles[1, 1] = 0.0

    total, scribble, consistency = sparse_adaptation_loss(
        logits,
        labels,
        valid,
        teacher,
        lambda_consistency=3.0,
    )
    _, scribble_again, consistency_again = sparse_adaptation_loss(
        logits,
        labels,
        valid,
        changed_at_scribbles,
        lambda_consistency=3.0,
    )

    assert torch.equal(scribble, scribble_again)
    assert torch.equal(consistency, consistency_again)
    torch.testing.assert_close(total, scribble + 3.0 * consistency)


def test_functional_anchor_allows_an_empty_active_gate() -> None:
    logits = torch.tensor([[0.5, -0.4], [0.1, -0.2]], requires_grad=True)
    teacher = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    active = torch.zeros((2, 2), dtype=torch.bool)

    loss = confidence_gated_bernoulli_kl(logits, teacher, active)
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_trainable_parameter_modes_are_exact() -> None:
    model = ResUNet(in_channels=6, features=(4, 8, 16, 32))

    configure_trainable_parameters(model, "head_only")
    head_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert head_names
    assert all(name.startswith("head.") for name in head_names)

    configure_trainable_parameters(model, "last_decoder_head")
    decoder_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("up_blocks.3.") for name in decoder_names)
    assert any(name.startswith("head.") for name in decoder_names)
    assert all(name.startswith(("up_blocks.3.", "head.")) for name in decoder_names)

    configure_trainable_parameters(model, "last2_decoder_head")
    last_two_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("up_blocks.2.") for name in last_two_names)
    assert any(name.startswith("up_blocks.3.") for name in last_two_names)
    assert all(
        name.startswith(("up_blocks.2.", "up_blocks.3.", "head."))
        for name in last_two_names
    )

    details = describe_freeze_policy(model, "full_decoder")
    full_decoder_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert details.trainable_module_names == (
        "up_blocks.0",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
        "head",
    )
    assert all(name.startswith(("up_blocks.", "head.")) for name in full_decoder_names)
    assert not any(
        name.startswith(("stem.", "down_blocks.", "bottleneck."))
        for name in full_decoder_names
    )
    assert details.trainable_parameters == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert details.frozen_parameters + details.trainable_parameters == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_head_only_step_does_not_change_frozen_weights(tmp_path: Path) -> None:
    model = ResUNet(in_channels=6, features=(4, 8, 16, 32)).eval()
    frozen = FrozenFibNet(
        model=model,
        checkpoint=tmp_path / "unused.pt",
        feature_mode="stack_relief",
        model_arch="resunet",
        image_size=16,
        device=torch.device("cpu"),
    )
    input_tensor = torch.rand(1, 6, 16, 16)
    with torch.no_grad():
        source_probability = torch.sigmoid(model(input_tensor)[0, 0])
    labels = torch.zeros((16, 16), dtype=torch.uint8)
    labels[3, 3] = PORE
    labels[12, 12] = SOLID
    tile = ScribbleTrainingTile(
        tensor=input_tensor,
        labels=labels,
        valid_mask=torch.ones((16, 16), dtype=torch.bool),
        source_probability=source_probability,
        x=0,
        y=0,
    )
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    adapt_from_scribbles(
        frozen,
        (tile,),
        "head_only",
        0.1,
        steps=1,
        learning_rate=1e-3,
    )

    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("head.")
    )
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("head.")
    )


def test_anchor_losses_have_interpretable_zero_at_source() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
    weight_loss = weight_anchor_parameter_mse(
        (("weight", parameter),), {"weight": torch.tensor([0.0, 1.0])}
    )
    probability = torch.tensor([[0.01, 0.5], [0.99, 0.8]])
    logits = torch.logit(probability)
    confidence = (probability <= 0.05) | (probability >= 0.95)
    functional_loss = confidence_gated_bernoulli_kl(logits, probability, confidence)

    torch.testing.assert_close(weight_loss, torch.tensor(2.5))
    torch.testing.assert_close(functional_loss, torch.tensor(0.0), atol=1e-7, rtol=0)


def test_regularized_adaptation_logs_raw_and_weighted_anchor_scales(
    tmp_path: Path,
) -> None:
    model = ResUNet(in_channels=6, features=(4, 8, 16, 32)).eval()
    frozen = FrozenFibNet(
        model=model,
        checkpoint=tmp_path / "unused.pt",
        feature_mode="stack_relief",
        model_arch="resunet",
        image_size=16,
        device=torch.device("cpu"),
    )
    source_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    input_tensor = torch.rand(1, 6, 16, 16)
    with torch.no_grad():
        source_probability = torch.sigmoid(model(input_tensor)[0, 0])
    labels = torch.zeros((16, 16), dtype=torch.uint8)
    labels[3, 3] = PORE
    labels[12, 12] = SOLID
    scribble_tile = ScribbleTrainingTile(
        tensor=input_tensor,
        labels=labels,
        valid_mask=torch.ones((16, 16), dtype=torch.bool),
        source_probability=torch.zeros((16, 16)),
        x=0,
        y=0,
    )
    confidence = torch.ones((16, 16), dtype=torch.bool)
    auxiliary = FunctionalAnchorTile(
        tensor=input_tensor,
        source_probability=source_probability,
        confidence_mask=confidence,
        slice_id="10",
        tile_index=0,
        x=0,
        y=0,
    )

    result = adapt_from_scribbles_with_anchors(
        frozen,
        (scribble_tile,),
        "full_decoder",
        steps=2,
        learning_rate=3e-5,
        source_state=source_state,
        lambda_weight=1e-2,
        log_weight_anchor=True,
        auxiliary_tiles=(auxiliary, auxiliary),
        lambda_functional=0.1,
    )

    assert result.history[0].weight_anchor_loss == 0.0
    assert result.history[1].weight_anchor_loss > 0.0
    assert result.history[1].weighted_weight_anchor_loss == pytest.approx(
        1e-2 * result.history[1].weight_anchor_loss
    )
    assert result.history[1].weighted_functional_anchor_loss == pytest.approx(
        0.1 * result.history[1].functional_anchor_loss
    )
    assert result.history[0].functional_anchor_slice_id == "10"
    assert result.history[0].functional_confident_fraction == 1.0
    assert result.history[0].functional_active_fraction == 1.0
