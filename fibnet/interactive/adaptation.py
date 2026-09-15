"""Short scribble-guided adaptation of a frozen FIB-NET source prior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as functional

from ..inference import iter_tiled_inputs
from .encoder import MANUSCRIPT_OVERLAP, MANUSCRIPT_TILE_SIZE, FrozenFibNet
from .io import PORE, UNLABELED, validate_scribbles

AdaptationMode = Literal[
    "head_only",
    "last_decoder_head",
    "last2_decoder_head",
    "full_decoder",
]
JointSamplingStrategy = Literal["joint_pixel_uniform", "joint_slice_balanced"]
JOINT_SAMPLING_STRATEGIES: tuple[JointSamplingStrategy, ...] = (
    "joint_pixel_uniform",
    "joint_slice_balanced",
)
ADAPTATION_MODES: tuple[AdaptationMode, ...] = (
    "head_only",
    "last_decoder_head",
    "last2_decoder_head",
    "full_decoder",
)


@dataclass(frozen=True)
class InteractiveAdaptationConfig:
    """Locked baseline for interactive scribble refinement."""

    mode: AdaptationMode = "last_decoder_head"
    lambda_consistency: float = 0.0
    optimizer: str = "Adam"
    learning_rate: float = 1e-4
    steps: int = 50
    threshold: float = 0.5


INTERACTIVE_ADAPTATION_BASELINE = InteractiveAdaptationConfig()


@dataclass(frozen=True)
class ScribbleTrainingTile:
    """One production-preprocessed tile plus sparse labels and source teacher."""

    tensor: torch.Tensor
    labels: torch.Tensor
    valid_mask: torch.Tensor
    source_probability: torch.Tensor
    x: int
    y: int


@dataclass(frozen=True)
class ScribbleSliceTrainingTiles:
    """Production-preprocessed scribble tiles belonging to one stack slice."""

    slice_id: str
    tiles: tuple[ScribbleTrainingTile, ...]


@dataclass(frozen=True)
class FunctionalAnchorTile:
    """Unlabeled target-stack tile with a cached frozen-source teacher."""

    tensor: torch.Tensor
    source_probability: torch.Tensor
    confidence_mask: torch.Tensor
    slice_id: str
    tile_index: int
    x: int
    y: int


@dataclass(frozen=True)
class AdaptationStep:
    step: int
    tile_index: int
    labeled_pixels: int
    total_loss: float
    scribble_loss: float
    consistency_loss: float
    slice_id: str | None = None
    weight_anchor_loss: float = 0.0
    weighted_weight_anchor_loss: float = 0.0
    functional_anchor_loss: float = 0.0
    weighted_functional_anchor_loss: float = 0.0
    functional_anchor_slice_id: str | None = None
    functional_confident_pixels: int = 0
    functional_confident_fraction: float = 0.0
    functional_active_pixels: int = 0
    functional_active_fraction: float = 0.0


@dataclass(frozen=True)
class AdaptationResult:
    mode: AdaptationMode
    lambda_consistency: float
    steps: int
    learning_rate: float
    trainable_parameters: int
    frozen_parameters: int
    trainable_module_names: tuple[str, ...]
    adaptation_time: float
    history: tuple[AdaptationStep, ...]
    lambda_weight: float = 0.0
    lambda_functional: float = 0.0


@dataclass(frozen=True)
class FreezePolicyInfo:
    """Auditable parameter selection for one adaptation freeze policy."""

    policy: AdaptationMode
    trainable_module_names: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int
    total_parameters: int


def prepare_scribble_training_tiles(
    frozen: FrozenFibNet,
    previous_path: str | Path,
    current_path: str | Path,
    next_path: str | Path,
    scribbles: np.ndarray,
    *,
    tile_size: int = MANUSCRIPT_TILE_SIZE,
    overlap: int = MANUSCRIPT_OVERLAP,
    compute_source_probability: bool = True,
) -> tuple[ScribbleTrainingTile, ...]:
    """Prepare labeled native tiles and frozen source probabilities without GT."""
    previous_path = Path(previous_path)
    current_path = Path(current_path)
    next_path = Path(next_path)
    with Image.open(current_path) as image:
        native_shape = (image.height, image.width)
    labels = validate_scribbles(scribbles, expected_shape=native_shape)
    frozen.model.eval()
    training_tiles = []
    with torch.no_grad():
        for tile in iter_tiled_inputs(
            current_path,
            tile_size,
            overlap,
            frozen.feature_mode,
            previous_path=previous_path,
            next_path=next_path,
        ):
            local_labels = np.full((tile_size, tile_size), UNLABELED, dtype=np.uint8)
            local_labels[: tile.crop_height, : tile.crop_width] = labels[
                tile.y : tile.y + tile.crop_height,
                tile.x : tile.x + tile.crop_width,
            ]
            if not np.any(local_labels != UNLABELED):
                continue
            valid = np.zeros((tile_size, tile_size), dtype=bool)
            valid[: tile.crop_height, : tile.crop_width] = True
            if compute_source_probability:
                source_logits = frozen.model(tile.tensor.to(frozen.device))
                source_probability = torch.sigmoid(source_logits[0, 0]).cpu()
            else:
                # Lambda=0 ablations do not consume the teacher tensor. Keeping a
                # correctly shaped placeholder avoids unnecessary source forwards.
                source_probability = torch.zeros(
                    (tile_size, tile_size), dtype=tile.tensor.dtype
                )
            training_tiles.append(
                ScribbleTrainingTile(
                    tensor=tile.tensor.cpu(),
                    labels=torch.from_numpy(local_labels),
                    valid_mask=torch.from_numpy(valid),
                    source_probability=source_probability,
                    x=tile.x,
                    y=tile.y,
                )
            )
    if not training_tiles:
        raise ValueError("No production inference tile contains a scribble pixel.")
    return tuple(training_tiles)


def configure_trainable_parameters(
    model: nn.Module, mode: AdaptationMode
) -> tuple[nn.Parameter, ...]:
    """Freeze the model except for the requested conservative adaptation block."""
    if mode not in ADAPTATION_MODES:
        raise ValueError(
            f"Unknown adaptation mode {mode!r}; expected {ADAPTATION_MODES}."
        )
    if not hasattr(model, "head") or not hasattr(model, "up_blocks"):
        raise ValueError(
            "The selected FIB-NET model has no expected decoder/head blocks."
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules: list[nn.Module] = [model.head]
    decoder_count = {
        "head_only": 0,
        "last_decoder_head": 1,
        "last2_decoder_head": 2,
        "full_decoder": len(model.up_blocks),
    }[mode]
    if decoder_count > len(model.up_blocks):
        raise ValueError(
            f"Adaptation mode {mode!r} requires {decoder_count} decoder blocks, "
            f"but the model has {len(model.up_blocks)}."
        )
    if decoder_count:
        modules[:0] = list(model.up_blocks[-decoder_count:])
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeError("Adaptation configuration selected no trainable parameters.")
    return trainable


def describe_freeze_policy(model: nn.Module, mode: AdaptationMode) -> FreezePolicyInfo:
    """Configure and describe the exact trainable/frozen parameter split."""
    configure_trainable_parameters(model, mode)
    decoder_count = {
        "head_only": 0,
        "last_decoder_head": 1,
        "last2_decoder_head": 2,
        "full_decoder": len(model.up_blocks),
    }[mode]
    decoder_names = tuple(
        f"up_blocks.{index}"
        for index in range(len(model.up_blocks) - decoder_count, len(model.up_blocks))
    )
    module_names = (*decoder_names, "head")
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return FreezePolicyInfo(
        policy=mode,
        trainable_module_names=module_names,
        trainable_parameters=trainable_parameters,
        frozen_parameters=total_parameters - trainable_parameters,
        total_parameters=total_parameters,
    )


def sparse_adaptation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    source_probability: torch.Tensor,
    *,
    lambda_consistency: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, scribble BCE, and unlabeled source-consistency MSE."""
    if lambda_consistency < 0:
        raise ValueError("lambda_consistency must be non-negative.")
    logits = logits.squeeze()
    if not (
        logits.shape == labels.shape == valid_mask.shape == source_probability.shape
    ):
        raise ValueError("Logits, labels, valid mask, and source prior must align.")
    scribbled = valid_mask & (labels != UNLABELED)
    if not torch.any(scribbled):
        raise ValueError("The adaptation tile must contain at least one scribble.")
    targets = (labels == PORE).to(dtype=logits.dtype)
    scribble_loss = functional.binary_cross_entropy_with_logits(
        logits[scribbled], targets[scribbled]
    )
    unlabeled = valid_mask & (labels == UNLABELED)
    if torch.any(unlabeled):
        consistency_loss = functional.mse_loss(
            torch.sigmoid(logits[unlabeled]),
            source_probability[unlabeled].to(dtype=logits.dtype),
        )
    else:
        consistency_loss = logits.sum() * 0.0
    total_loss = scribble_loss + lambda_consistency * consistency_loss
    return total_loss, scribble_loss, consistency_loss


def weight_anchor_parameter_mse(
    trainable_parameters: Sequence[tuple[str, nn.Parameter]],
    source_parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Mean squared displacement from source over all trainable parameters."""
    if not trainable_parameters:
        raise ValueError("At least one trainable parameter is required.")
    squared_sum: torch.Tensor | None = None
    element_count = 0
    for name, parameter in trainable_parameters:
        if name not in source_parameters:
            raise KeyError(f"Source state is missing trainable parameter {name!r}.")
        source = source_parameters[name]
        if source.shape != parameter.shape:
            raise ValueError(
                f"Source parameter {name!r} has shape {source.shape}; "
                f"expected {parameter.shape}."
            )
        difference = parameter - source.to(
            device=parameter.device, dtype=parameter.dtype
        )
        value = difference.square().sum()
        squared_sum = value if squared_sum is None else squared_sum + value
        element_count += parameter.numel()
    assert squared_sum is not None
    return squared_sum / element_count


def confidence_gated_bernoulli_kl(
    logits: torch.Tensor,
    source_probability: torch.Tensor,
    confidence_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean Bernoulli KL(source || adapted) over source-confident pixels."""
    logits = logits.squeeze()
    if not (logits.shape == source_probability.shape == confidence_mask.shape):
        raise ValueError("Functional-anchor logits, teacher, and mask must align.")
    if not torch.any(confidence_mask):
        return logits.sum() * 0.0
    teacher = source_probability[confidence_mask].to(dtype=logits.dtype)
    if torch.any((teacher < 0.0) | (teacher > 1.0)):
        raise ValueError("Source probabilities must lie in [0, 1].")
    epsilon = torch.finfo(logits.dtype).eps
    teacher = teacher.clamp(epsilon, 1.0 - epsilon)
    selected_logits = logits[confidence_mask]
    return (
        teacher * (teacher.log() - functional.logsigmoid(selected_logits))
        + (1.0 - teacher)
        * ((1.0 - teacher).log() - functional.logsigmoid(-selected_logits))
    ).mean()


def adapt_from_scribbles(
    frozen: FrozenFibNet,
    training_tiles: tuple[ScribbleTrainingTile, ...],
    mode: AdaptationMode,
    lambda_consistency: float,
    *,
    steps: int = 50,
    learning_rate: float = 1e-4,
    random_state: int = 42,
) -> AdaptationResult:
    """Run a short deterministic adaptation from the model's current weights."""
    return adapt_from_multislice_scribbles(
        frozen,
        (ScribbleSliceTrainingTiles(slice_id="single", tiles=training_tiles),),
        mode,
        lambda_consistency,
        steps=steps,
        learning_rate=learning_rate,
        sampling_strategy="joint_pixel_uniform",
        random_state=random_state,
    )


def build_multislice_tile_schedule(
    slice_tiles: Sequence[ScribbleSliceTrainingTiles],
    steps: int,
    sampling_strategy: JointSamplingStrategy,
    *,
    slice_weights: Sequence[float] | None = None,
    random_state: int = 42,
) -> tuple[tuple[int, int], ...]:
    """Return deterministic ``(slice, tile)`` indices for joint adaptation.

    Pixel-uniform scheduling shuffles the union of all scribble-containing tiles.
    Slice-balanced scheduling cycles through shuffled slice permutations and then
    draws from a shuffled tile cycle within the selected slice. Consequently,
    slice counts differ by at most one while tiles remain exchangeable inside a
    slice.
    """
    if steps < 1:
        raise ValueError("steps must be positive.")
    if sampling_strategy not in JOINT_SAMPLING_STRATEGIES:
        raise ValueError(
            f"Unknown joint sampling strategy {sampling_strategy!r}; "
            f"expected {JOINT_SAMPLING_STRATEGIES}."
        )
    if not slice_tiles:
        raise ValueError("At least one scribble slice is required.")
    if any(not item.tiles for item in slice_tiles):
        raise ValueError("Every scribble slice must contain at least one tile.")
    slice_ids = [item.slice_id for item in slice_tiles]
    if len(set(slice_ids)) != len(slice_ids):
        raise ValueError("Scribble slice identifiers must be unique.")
    weights = None
    if slice_weights is not None:
        weights = np.asarray(slice_weights, dtype=np.float64)
        if weights.shape != (len(slice_tiles),):
            raise ValueError("slice_weights must contain one value per scribble slice.")
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("slice_weights must contain finite positive values.")
        if sampling_strategy != "joint_slice_balanced":
            raise ValueError(
                "slice_weights are supported only by joint_slice_balanced sampling."
            )

    generator = np.random.default_rng(random_state)
    schedule: list[tuple[int, int]] = []
    if sampling_strategy == "joint_pixel_uniform":
        flattened = [
            (slice_index, tile_index)
            for slice_index, item in enumerate(slice_tiles)
            for tile_index in range(len(item.tiles))
        ]
        while len(schedule) < steps:
            order = generator.permutation(len(flattened))
            take = min(steps - len(schedule), len(flattened))
            schedule.extend(flattened[int(index)] for index in order[:take])
        return tuple(schedule)

    tile_cycles: list[list[int]] = [[] for _ in slice_tiles]
    slice_order_values: list[int] = []
    if weights is None:
        while len(slice_order_values) < steps:
            slice_order_values.extend(
                int(index) for index in generator.permutation(len(slice_tiles))
            )
        slice_order_values = slice_order_values[:steps]
    else:
        probabilities = weights / weights.sum()
        counts = np.zeros(len(slice_tiles), dtype=np.int64)
        tie_order = generator.permutation(len(slice_tiles))
        tie_rank = np.empty(len(slice_tiles), dtype=np.int64)
        tie_rank[tie_order] = np.arange(len(slice_tiles))
        for step in range(steps):
            deficits = probabilities * (step + 1) - counts
            best = np.flatnonzero(np.isclose(deficits, deficits.max()))
            slice_index = int(best[np.argmin(tie_rank[best])])
            slice_order_values.append(slice_index)
            counts[slice_index] += 1

    for slice_index in slice_order_values:
        if not tile_cycles[slice_index]:
            tile_cycles[slice_index].extend(
                int(index)
                for index in generator.permutation(len(slice_tiles[slice_index].tiles))
            )
        tile_index = tile_cycles[slice_index].pop(0)
        schedule.append((slice_index, tile_index))
    return tuple(schedule)


def adapt_from_multislice_scribbles(
    frozen: FrozenFibNet,
    slice_tiles: Sequence[ScribbleSliceTrainingTiles],
    mode: AdaptationMode,
    lambda_consistency: float,
    *,
    steps: int,
    learning_rate: float,
    sampling_strategy: JointSamplingStrategy,
    slice_weights: Sequence[float] | None = None,
    random_state: int = 42,
) -> AdaptationResult:
    """Adapt one shared model jointly from sparse labels on several slices."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    schedule = build_multislice_tile_schedule(
        slice_tiles,
        steps,
        sampling_strategy,
        slice_weights=slice_weights,
        random_state=random_state,
    )
    policy = describe_freeze_policy(frozen.model, mode)
    trainable = tuple(
        parameter for parameter in frozen.model.parameters() if parameter.requires_grad
    )
    frozen.model.eval()
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    history = []
    started = perf_counter()
    for step, (slice_index, tile_index) in enumerate(schedule, start=1):
        slice_item = slice_tiles[slice_index]
        tile = slice_item.tiles[tile_index]
        input_tensor = tile.tensor.to(frozen.device)
        labels = tile.labels.to(frozen.device)
        valid_mask = tile.valid_mask.to(frozen.device)
        source_probability = tile.source_probability.to(frozen.device)
        optimizer.zero_grad(set_to_none=True)
        logits = frozen.model(input_tensor)[0, 0]
        total_loss, scribble_loss, consistency_loss = sparse_adaptation_loss(
            logits,
            labels,
            valid_mask,
            source_probability,
            lambda_consistency=lambda_consistency,
        )
        total_loss.backward()
        optimizer.step()
        history.append(
            AdaptationStep(
                step=step,
                tile_index=tile_index,
                labeled_pixels=int(torch.count_nonzero(labels).item()),
                total_loss=float(total_loss.detach().cpu()),
                scribble_loss=float(scribble_loss.detach().cpu()),
                consistency_loss=float(consistency_loss.detach().cpu()),
                slice_id=slice_item.slice_id,
            )
        )
    frozen.model.eval()
    return AdaptationResult(
        mode=mode,
        lambda_consistency=lambda_consistency,
        steps=steps,
        learning_rate=learning_rate,
        trainable_parameters=policy.trainable_parameters,
        frozen_parameters=policy.frozen_parameters,
        trainable_module_names=policy.trainable_module_names,
        adaptation_time=perf_counter() - started,
        history=tuple(history),
    )


def adapt_from_scribbles_with_anchors(
    frozen: FrozenFibNet,
    training_tiles: tuple[ScribbleTrainingTile, ...],
    mode: AdaptationMode,
    *,
    steps: int,
    learning_rate: float,
    source_state: Mapping[str, torch.Tensor] | None = None,
    lambda_weight: float = 0.0,
    log_weight_anchor: bool = False,
    auxiliary_tiles: Sequence[FunctionalAnchorTile] = (),
    lambda_functional: float = 0.0,
    functional_disagreement_delta: float | None = None,
    scribble_slice_id: str = "current",
    random_state: int = 42,
) -> AdaptationResult:
    """Adapt one scribble slice with source-weight and/or functional anchors."""
    if steps < 1:
        raise ValueError("steps must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if lambda_weight < 0.0 or lambda_functional < 0.0:
        raise ValueError("Anchor coefficients must be non-negative.")
    if functional_disagreement_delta is not None and not (
        0.0 <= functional_disagreement_delta < 1.0
    ):
        raise ValueError("functional_disagreement_delta must lie in [0, 1).")
    if not training_tiles:
        raise ValueError("At least one scribble training tile is required.")
    if (lambda_weight > 0.0 or log_weight_anchor) and source_state is None:
        raise ValueError("source_state is required for weight anchoring.")
    if lambda_functional > 0.0 and len(auxiliary_tiles) < steps:
        raise ValueError("Functional anchoring requires one auxiliary tile per step.")

    policy = describe_freeze_policy(frozen.model, mode)
    named_trainable = tuple(
        (name, parameter)
        for name, parameter in frozen.model.named_parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.Adam(
        tuple(parameter for _, parameter in named_trainable), lr=learning_rate
    )
    tile_schedule = build_multislice_tile_schedule(
        (ScribbleSliceTrainingTiles(slice_id=scribble_slice_id, tiles=training_tiles),),
        steps,
        "joint_pixel_uniform",
        random_state=random_state,
    )
    history = []
    frozen.model.eval()
    started = perf_counter()
    for step, (_, tile_index) in enumerate(tile_schedule, start=1):
        tile = training_tiles[tile_index]
        labels = tile.labels.to(frozen.device)
        valid_mask = tile.valid_mask.to(frozen.device)
        optimizer.zero_grad(set_to_none=True)
        logits = frozen.model(tile.tensor.to(frozen.device))[0, 0]
        _total, scribble_loss, consistency_loss = sparse_adaptation_loss(
            logits,
            labels,
            valid_mask,
            tile.source_probability.to(frozen.device),
            lambda_consistency=0.0,
        )

        zero = scribble_loss.new_zeros(())
        weight_loss = zero
        if source_state is not None and (lambda_weight > 0.0 or log_weight_anchor):
            if lambda_weight > 0.0:
                weight_loss = weight_anchor_parameter_mse(named_trainable, source_state)
            else:
                with torch.no_grad():
                    weight_loss = weight_anchor_parameter_mse(
                        named_trainable, source_state
                    )

        functional_loss = zero
        anchor_slice_id = None
        confident_pixels = 0
        confident_fraction = 0.0
        active_pixels = 0
        active_fraction = 0.0
        if lambda_functional > 0.0:
            auxiliary = auxiliary_tiles[step - 1]
            anchor_logits = frozen.model(auxiliary.tensor.to(frozen.device))[0, 0]
            confidence_mask = auxiliary.confidence_mask.to(frozen.device)
            active_mask = confidence_mask
            if functional_disagreement_delta is not None:
                teacher = auxiliary.source_probability.to(frozen.device)
                disagreement = (torch.sigmoid(anchor_logits).detach() - teacher).abs()
                active_mask = active_mask & (
                    disagreement >= functional_disagreement_delta
                )
            functional_loss = confidence_gated_bernoulli_kl(
                anchor_logits,
                auxiliary.source_probability.to(frozen.device),
                active_mask,
            )
            anchor_slice_id = auxiliary.slice_id
            confident_pixels = int(torch.count_nonzero(confidence_mask).item())
            confident_fraction = confident_pixels / confidence_mask.numel()
            active_pixels = int(torch.count_nonzero(active_mask).item())
            active_fraction = active_pixels / active_mask.numel()

        weighted_weight = lambda_weight * weight_loss
        weighted_functional = lambda_functional * functional_loss
        total_loss = scribble_loss + weighted_weight + weighted_functional
        total_loss.backward()
        optimizer.step()
        history.append(
            AdaptationStep(
                step=step,
                tile_index=tile_index,
                labeled_pixels=int(
                    torch.count_nonzero(valid_mask & (labels != UNLABELED)).item()
                ),
                total_loss=float(total_loss.detach().cpu()),
                scribble_loss=float(scribble_loss.detach().cpu()),
                consistency_loss=float(consistency_loss.detach().cpu()),
                slice_id=scribble_slice_id,
                weight_anchor_loss=float(weight_loss.detach().cpu()),
                weighted_weight_anchor_loss=float(weighted_weight.detach().cpu()),
                functional_anchor_loss=float(functional_loss.detach().cpu()),
                weighted_functional_anchor_loss=float(
                    weighted_functional.detach().cpu()
                ),
                functional_anchor_slice_id=anchor_slice_id,
                functional_confident_pixels=confident_pixels,
                functional_confident_fraction=confident_fraction,
                functional_active_pixels=active_pixels,
                functional_active_fraction=active_fraction,
            )
        )
    frozen.model.eval()
    return AdaptationResult(
        mode=mode,
        lambda_consistency=0.0,
        steps=steps,
        learning_rate=learning_rate,
        trainable_parameters=policy.trainable_parameters,
        frozen_parameters=policy.frozen_parameters,
        trainable_module_names=policy.trainable_module_names,
        adaptation_time=perf_counter() - started,
        history=tuple(history),
        lambda_weight=lambda_weight,
        lambda_functional=lambda_functional,
    )
