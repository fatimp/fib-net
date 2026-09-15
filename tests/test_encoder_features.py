from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fibnet.inference import TiledInput
from fibnet.interactive.encoder import (
    FrozenFibNet,
    _accumulate_weighted_feature_tile,
    extract_tiled_deep_features,
)
from fibnet.model import ResUNet


def _save_context_images(directory: Path, shape: tuple[int, int]) -> tuple[Path, ...]:
    paths = []
    base = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    for index, offset in enumerate((0, 7, 14), start=1):
        path = directory / f"{index}.tiff"
        Image.fromarray(base + offset).save(path)
        paths.append(path)
    return tuple(paths)


def test_weighted_feature_accumulation_uses_tile_weight_and_position() -> None:
    feature_sum = np.zeros((5, 6, 2), dtype=np.float32)
    values = np.full((2, 3, 2), 4.0, dtype=np.float32)
    weight = np.full((2, 3), 0.25, dtype=np.float32)
    tile = TiledInput(torch.empty(1), 2, 1, 3, 2, weight)

    _accumulate_weighted_feature_tile(feature_sum, values, tile)

    np.testing.assert_array_equal(feature_sum[1:3, 2:5], 1.0)
    assert np.count_nonzero(feature_sum) == 12


def test_native_tiled_extraction_returns_stem_and_final_decoder_without_pca(
    tmp_path: Path,
) -> None:
    previous, current, next_path = _save_context_images(tmp_path, (18, 22))
    model = ResUNet(in_channels=6, features=(4, 8, 16, 32)).eval()
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    frozen = FrozenFibNet(
        model=model,
        checkpoint=tmp_path / "unused.pt",
        feature_mode="stack_relief",
        model_arch="resunet",
        image_size=16,
        device=torch.device("cpu"),
    )

    result = extract_tiled_deep_features(
        frozen,
        previous,
        current,
        next_path,
        tile_size=16,
        overlap=4,
    )

    assert result.tile_count == 4
    assert result.shallow_encoder.feature_tensor.shape == (18, 22, 4)
    assert result.decoder.feature_tensor.shape == (18, 22, 4)
    assert result.shallow_encoder.feature_tensor.dtype == np.float32
    assert result.decoder.feature_tensor.dtype == np.float32
    assert np.isfinite(result.shallow_encoder.feature_tensor).all()
    assert np.isfinite(result.decoder.feature_tensor).all()
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
