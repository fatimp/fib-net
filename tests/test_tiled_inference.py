import numpy as np
from PIL import Image

from fibnet.inference import (
    _axis_positions,
    _tile_weight,
    accumulate_weighted_tile,
    iter_tiled_inputs,
    normalize_weighted_probability,
)

HEIGHT = 963
WIDTH = 2022
TILE_SIZE = 384
OVERLAP = 96


def reconstruct(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stride = TILE_SIZE - OVERLAP
    xs = _axis_positions(WIDTH, TILE_SIZE, stride)
    ys = _axis_positions(HEIGHT, TILE_SIZE, stride)
    tile_weight = _tile_weight(TILE_SIZE)
    probability_sum = np.zeros_like(field, dtype=np.float32)
    weight_sum = np.zeros_like(field, dtype=np.float32)

    for y in ys:
        for x in xs:
            crop_height = min(TILE_SIZE, HEIGHT - y)
            crop_width = min(TILE_SIZE, WIDTH - x)
            probability = field[y : y + crop_height, x : x + crop_width]
            weight = tile_weight[:crop_height, :crop_width]
            accumulate_weighted_tile(
                probability_sum, weight_sum, probability, weight, y, x
            )

    return normalize_weighted_probability(probability_sum, weight_sum), weight_sum


def test_constant_probability_is_preserved_for_realistic_geometry() -> None:
    field = np.full((HEIGHT, WIDTH), 0.7, dtype=np.float32)

    reconstructed, weight_sum = reconstruct(field)

    assert np.all(weight_sum > 0.0)
    np.testing.assert_allclose(reconstructed, field, rtol=1e-6, atol=1e-6)


def test_spatial_field_is_preserved_at_borders_overlaps_and_low_weights() -> None:
    y = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, WIDTH, dtype=np.float32)[None, :]
    field = (0.1 + 0.35 * x + 0.45 * y + 0.05 * x * y).astype(np.float32)

    reconstructed, weight_sum = reconstruct(field)

    stride = TILE_SIZE - OVERLAP
    sample_points = np.array(
        [
            (0, 0),
            (0, WIDTH - 1),
            (HEIGHT - 1, 0),
            (HEIGHT - 1, WIDTH - 1),
            (HEIGHT // 2, WIDTH // 2),
            (stride, stride),
            (TILE_SIZE - 1, TILE_SIZE - 1),
        ]
    )
    assert np.any(weight_sum < 1.0)
    np.testing.assert_allclose(reconstructed, field, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        reconstructed[sample_points[:, 0], sample_points[:, 1]],
        field[sample_points[:, 0], sample_points[:, 1]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_axis_positions_rejects_uncovered_gaps() -> None:
    positions = _axis_positions(100, 32, 24)

    assert positions[0] == 0
    assert positions[-1] == 68


def test_tiled_inputs_reuse_production_stack_relief_preprocessing(tmp_path) -> None:
    paths = []
    for index, value in enumerate((40, 80, 120)):
        path = tmp_path / f"{index}.tiff"
        Image.fromarray(np.full((18, 22), value, dtype=np.uint8)).save(path)
        paths.append(path)

    tiles = list(
        iter_tiled_inputs(
            paths[1],
            tile_size=16,
            overlap=4,
            feature_mode="stack_relief",
            previous_path=paths[0],
            next_path=paths[2],
        )
    )

    assert len(tiles) == 4
    assert {(tile.x, tile.y) for tile in tiles} == {(0, 0), (6, 0), (0, 2), (6, 2)}
    assert all(tuple(tile.tensor.shape) == (1, 6, 16, 16) for tile in tiles)
    assert {(tile.crop_width, tile.crop_height) for tile in tiles} == {(16, 16)}
