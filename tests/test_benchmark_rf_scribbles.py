from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from skimage.measure import label as connected_components

from fibnet.interactive import PORE, SOLID, UNLABELED
from scripts.benchmark_rf_scribbles import generate_scribbles, load_ground_truth


@pytest.fixture
def pore_ground_truth() -> np.ndarray:
    pore = np.zeros((180, 200), dtype=bool)
    pore[25:160, 30:120] = True
    return pore


def make_scribbles(pore: np.ndarray, seed: int = 17) -> np.ndarray:
    return generate_scribbles(
        pore,
        stroke_count=3,
        stroke_length=18,
        brush_width=5,
        seed=seed,
    )


def test_scribbles_stay_inside_their_ground_truth_classes(
    pore_ground_truth: np.ndarray,
) -> None:
    scribbles = make_scribbles(pore_ground_truth)

    assert np.all(pore_ground_truth[scribbles == PORE])
    assert np.all(~pore_ground_truth[scribbles == SOLID])


def test_scribbles_are_reproducible_for_the_same_seed(
    pore_ground_truth: np.ndarray,
) -> None:
    first = make_scribbles(pore_ground_truth, seed=5)
    second = make_scribbles(pore_ground_truth, seed=5)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_usually_produce_different_scribbles(
    pore_ground_truth: np.ndarray,
) -> None:
    first = make_scribbles(pore_ground_truth, seed=5)
    second = make_scribbles(pore_ground_truth, seed=6)

    assert not np.array_equal(first, second)


def test_generation_leaves_unlabeled_pixels(pore_ground_truth: np.ndarray) -> None:
    scribbles = make_scribbles(pore_ground_truth)

    assert np.any(scribbles == UNLABELED)
    assert np.mean(scribbles != UNLABELED) < 0.25


def test_single_strokes_are_spatially_connected(
    pore_ground_truth: np.ndarray,
) -> None:
    scribbles = generate_scribbles(
        pore_ground_truth,
        stroke_count=1,
        stroke_length=25,
        brush_width=3,
        seed=9,
    )

    assert connected_components(scribbles == PORE, connectivity=2).max() == 1
    assert connected_components(scribbles == SOLID, connectivity=2).max() == 1


def test_too_small_object_has_clear_error() -> None:
    pore = np.zeros((15, 15), dtype=bool)
    pore[7, 7] = True

    with pytest.raises(ValueError, match=r"pore stroke.*class is too small"):
        generate_scribbles(
            pore,
            stroke_count=1,
            stroke_length=5,
            brush_width=5,
        )


def test_ground_truth_loader_uses_exact_zero_as_pore(tmp_path: Path) -> None:
    mask = np.array([[0, 1], [127, 255]], dtype=np.uint8)
    path = tmp_path / "manual.tiff"
    Image.fromarray(mask).save(path)

    pore = load_ground_truth(path, mask.shape)

    np.testing.assert_array_equal(pore, [[True, False], [False, False]])
