from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fibnet.interactive import (
    PORE,
    SOLID,
    UNLABELED,
    build_training_set,
    extract_multiscale_features,
    predict_pore_probability,
    segment_image,
)
from fibnet.interactive.io import save_outputs, validate_scribbles


def test_multiscale_feature_extraction() -> None:
    y, x = np.mgrid[:24, :32]
    image = (x + 2 * y).astype(np.float32)

    features = extract_multiscale_features(image)

    assert features.shape[:2] == image.shape
    assert features.shape[2] == 20
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_unlabeled_pixels_are_not_in_training_set() -> None:
    features = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    scribbles = np.full((3, 4), UNLABELED, dtype=np.uint8)
    scribbles[0, 1] = PORE
    scribbles[2, 3] = SOLID

    training_features, labels = build_training_set(features, scribbles)

    np.testing.assert_array_equal(training_features, features[[0, 2], [1, 3]])
    np.testing.assert_array_equal(labels, [PORE, SOLID])


def test_pore_probability_uses_classes_mapping() -> None:
    class ReversedClassifier:
        classes_ = np.array([SOLID, PORE])

        def predict_proba(self, samples: np.ndarray) -> np.ndarray:
            return np.tile([0.15, 0.85], (samples.shape[0], 1))

    features = np.zeros((3, 5, 2), dtype=np.float32)

    probability = predict_pore_probability(ReversedClassifier(), features)

    assert probability.shape == features.shape[:2]
    np.testing.assert_allclose(probability, 0.85)


def test_two_phase_image_is_segmented_with_high_quality(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    shape = (64, 72)
    truth = np.zeros(shape, dtype=bool)
    truth[:, :34] = True
    image = np.where(truth, 0.18, 0.82)
    image += rng.normal(0.0, 0.025, shape)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    scribbles = np.zeros(shape, dtype=np.uint8)
    scribbles[8:57:12, 8:29:10] = PORE
    scribbles[8:57:12, 43:68:10] = SOLID

    result = segment_image(
        image,
        scribbles,
        rf_options={"n_estimators": 50, "n_jobs": 1},
    )
    paths = save_outputs(tmp_path, result.probability, result.segmentation)

    assert result.probability.shape == shape
    assert result.probability.dtype == np.float32
    assert result.segmentation.shape == shape
    assert np.mean(result.segmentation == truth) > 0.98
    stored_probability = np.load(paths.probability, allow_pickle=False)
    assert stored_probability.dtype == np.float32
    assert np.all((stored_probability >= 0.0) & (stored_probability <= 1.0))
    stored_segmentation = np.asarray(Image.open(paths.segmentation))
    assert set(np.unique(stored_segmentation)) <= {0, 255}


def test_invalid_scribble_values_are_rejected() -> None:
    scribbles = np.array([[PORE, SOLID], [UNLABELED, 3]], dtype=np.uint8)

    with pytest.raises(ValueError, match=r"Unknown scribble label value.*3"):
        validate_scribbles(scribbles)


@pytest.mark.parametrize(
    ("scribbles", "message"),
    [
        (np.array([[PORE, UNLABELED]]), "solid scribble"),
        (np.array([[SOLID, UNLABELED]]), "pore scribble"),
    ],
)
def test_both_scribble_classes_are_required(
    scribbles: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_scribbles(scribbles)


def test_constant_image_is_handled_without_nonfinite_output() -> None:
    image = np.full((20, 20), 0.5, dtype=np.float32)
    scribbles = np.zeros_like(image, dtype=np.uint8)
    scribbles[2, 2] = PORE
    scribbles[-3, -3] = SOLID

    result = segment_image(
        image,
        scribbles,
        rf_options={"n_estimators": 10, "n_jobs": 1},
    )

    assert result.probability.shape == image.shape
    assert np.isfinite(result.probability).all()
    assert np.all((result.probability >= 0.0) & (result.probability <= 1.0))
