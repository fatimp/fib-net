"""Native-resolution tiled deep features from a frozen FIB-NET model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch import nn

from ..image_features import feature_channels
from ..inference import TiledInput, iter_tiled_inputs
from ..model import build_model
from .features import FeatureSet

ENCODER_FEATURE_MODE = "encoder"
MANUSCRIPT_TILE_SIZE = 384
MANUSCRIPT_OVERLAP = 96


@dataclass(frozen=True)
class FrozenFibNet:
    """A loaded checkpoint and the preprocessing metadata needed for inference."""

    model: nn.Module
    checkpoint: Path
    feature_mode: str
    model_arch: str
    image_size: int
    device: torch.device


@dataclass(frozen=True)
class TiledDeepFeatureResult:
    """Overlap-weighted native-resolution shallow and decoder feature maps."""

    shallow_encoder: FeatureSet
    decoder: FeatureSet
    tile_count: int
    tile_size: int
    overlap: int
    extraction_time: float


def load_frozen_fibnet(
    checkpoint_path: str | Path, *, device: str = "cpu"
) -> FrozenFibNet:
    """Load a FIB-NET checkpoint in eval mode with all gradients disabled."""
    path = Path(checkpoint_path)
    torch_device = torch.device(device)
    checkpoint = torch.load(path, map_location=torch_device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    image_size = int(checkpoint_args.get("image_size", 256))
    feature_mode = str(checkpoint_args.get("feature_mode", "grayscale"))
    model_arch = str(checkpoint_args.get("model_arch", "unet"))
    model = build_model(
        model_arch,
        in_channels=feature_channels(feature_mode),
    ).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return FrozenFibNet(
        model=model,
        checkpoint=path,
        feature_mode=feature_mode,
        model_arch=model_arch,
        image_size=image_size,
        device=torch_device,
    )


def _accumulate_weighted_feature_tile(
    feature_sum: np.ndarray,
    feature_tile: np.ndarray,
    tile: TiledInput,
) -> None:
    expected_shape = (tile.crop_height, tile.crop_width, feature_sum.shape[-1])
    if feature_tile.shape != expected_shape:
        raise ValueError(
            f"Feature tile has shape {feature_tile.shape}; expected {expected_shape}."
        )
    feature_sum[
        tile.y : tile.y + tile.crop_height,
        tile.x : tile.x + tile.crop_width,
    ] += feature_tile * tile.weight[..., None]


def _feature_names(prefix: str, channels: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index:02d}" for index in range(1, channels + 1))


def extract_tiled_deep_features(
    frozen: FrozenFibNet,
    previous_path: str | Path,
    current_path: str | Path,
    next_path: str | Path,
    *,
    tile_size: int = MANUSCRIPT_TILE_SIZE,
    overlap: int = MANUSCRIPT_OVERLAP,
) -> TiledDeepFeatureResult:
    """Extract native features with production tile preprocessing and blending.

    ``shallow_encoder`` is the output of the full-resolution stem. ``decoder``
    is the output of the final decoder block immediately before the head. No
    labels, ground truth, or dimensionality reduction are accepted by this API.
    """
    previous_path = Path(previous_path)
    current_path = Path(current_path)
    next_path = Path(next_path)
    with Image.open(current_path) as image:
        width, height = image.size

    if not hasattr(frozen.model, "stem") or not hasattr(frozen.model, "up_blocks"):
        raise ValueError(
            "The selected FIB-NET model does not expose encoder/decoder blocks."
        )
    if len(frozen.model.up_blocks) == 0:
        raise ValueError("The selected FIB-NET model has no decoder blocks.")

    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured[name] = output

        return hook

    handles = (
        frozen.model.stem.register_forward_hook(capture("shallow_encoder")),
        frozen.model.up_blocks[-1].register_forward_hook(capture("decoder")),
    )
    shallow_sum: np.ndarray | None = None
    decoder_sum: np.ndarray | None = None
    weight_sum = np.zeros((height, width), dtype=np.float32)
    tile_count = 0
    started = perf_counter()
    try:
        with torch.no_grad():
            for tile in iter_tiled_inputs(
                current_path,
                tile_size,
                overlap,
                frozen.feature_mode,
                previous_path=previous_path,
                next_path=next_path,
            ):
                captured.clear()
                frozen.model(tile.tensor.to(frozen.device))
                if set(captured) != {"shallow_encoder", "decoder"}:
                    raise RuntimeError(
                        "Failed to capture the requested FIB-NET features."
                    )
                shallow_tile = (
                    captured["shallow_encoder"][
                        0, :, : tile.crop_height, : tile.crop_width
                    ]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                decoder_tile = (
                    captured["decoder"][0, :, : tile.crop_height, : tile.crop_width]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                if shallow_sum is None:
                    shallow_sum = np.zeros(
                        (height, width, shallow_tile.shape[-1]), dtype=np.float32
                    )
                    decoder_sum = np.zeros(
                        (height, width, decoder_tile.shape[-1]), dtype=np.float32
                    )
                assert decoder_sum is not None
                _accumulate_weighted_feature_tile(shallow_sum, shallow_tile, tile)
                _accumulate_weighted_feature_tile(decoder_sum, decoder_tile, tile)
                weight_sum[
                    tile.y : tile.y + tile.crop_height,
                    tile.x : tile.x + tile.crop_width,
                ] += tile.weight
                tile_count += 1
    finally:
        for handle in handles:
            handle.remove()

    if shallow_sum is None or decoder_sum is None or tile_count == 0:
        raise RuntimeError("Tiled deep-feature extraction produced no tiles.")
    if np.any(weight_sum <= 0.0):
        raise RuntimeError("Tiled deep-feature extraction left uncovered pixels.")
    shallow_sum /= weight_sum[..., None]
    decoder_sum /= weight_sum[..., None]
    if not np.isfinite(shallow_sum).all() or not np.isfinite(decoder_sum).all():
        raise ValueError("Tiled deep-feature extraction produced non-finite values.")
    return TiledDeepFeatureResult(
        shallow_encoder=FeatureSet(
            shallow_sum,
            _feature_names("shallow_encoder", shallow_sum.shape[-1]),
        ),
        decoder=FeatureSet(
            decoder_sum,
            _feature_names("decoder", decoder_sum.shape[-1]),
        ),
        tile_count=tile_count,
        tile_size=tile_size,
        overlap=overlap,
        extraction_time=perf_counter() - started,
    )
