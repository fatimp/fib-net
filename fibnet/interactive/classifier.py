"""Random Forest training and prediction from sparse semantic scribbles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .features import DEFAULT_FEATURE_EXTRACTOR, FeatureExtractor
from .io import PORE, UNLABELED, validate_inputs, validate_scribbles

RANDOM_FOREST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 200,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}


class ProbabilityClassifier(Protocol):
    """Minimal classifier interface needed for dense pore prediction."""

    classes_: np.ndarray

    def predict_proba(self, samples: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class SegmentationResult:
    """Dense result plus trained classifier and feature-tensor metadata."""

    probability: np.ndarray
    segmentation: np.ndarray
    classifier: RandomForestClassifier
    feature_shape: tuple[int, int, int]


def _validate_features(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features)
    if array.ndim != 3 or array.shape[-1] == 0:
        raise ValueError(
            "Expected features with shape (height, width, feature_count), "
            f"received {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Feature values must be finite numeric data.")
    return array


def build_training_set(
    features: np.ndarray, scribbles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select only pore and solid scribbled pixels for classifier training."""
    feature_array = _validate_features(features)
    labels = validate_scribbles(scribbles, expected_shape=feature_array.shape[:2])
    labeled = labels != UNLABELED
    return feature_array[labeled], labels[labeled]


def train_classifier(
    features: np.ndarray,
    scribbles: np.ndarray,
    *,
    rf_options: Mapping[str, Any] | None = None,
) -> RandomForestClassifier:
    """Fit a Random Forest using only explicitly scribbled pixels."""
    training_features, training_labels = build_training_set(features, scribbles)
    options = {**RANDOM_FOREST_DEFAULTS, **(rf_options or {})}
    classifier = RandomForestClassifier(**options)
    classifier.fit(training_features, training_labels)
    return classifier


def predict_pore_probability(
    classifier: ProbabilityClassifier, features: np.ndarray
) -> np.ndarray:
    """Predict the PORE probability by looking up its classifier class column."""
    feature_array = _validate_features(features)
    classes = np.asarray(classifier.classes_)
    pore_columns = np.flatnonzero(classes == PORE)
    if pore_columns.size != 1:
        raise ValueError(
            f"Classifier classes {classes.tolist()} do not contain exactly one "
            f"PORE label ({PORE})."
        )
    flat_features = feature_array.reshape(-1, feature_array.shape[-1])
    probabilities = np.asarray(classifier.predict_proba(flat_features))
    expected_shape = (flat_features.shape[0], classes.size)
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            "Classifier returned probability shape "
            f"{probabilities.shape}; expected {expected_shape}."
        )
    pore = probabilities[:, int(pore_columns[0])]
    pore = np.clip(pore, 0.0, 1.0).astype(np.float32, copy=False)
    return pore.reshape(feature_array.shape[:2])


def segment_image(
    image: np.ndarray,
    scribbles: np.ndarray,
    *,
    threshold: float = 0.5,
    feature_extractor: FeatureExtractor = DEFAULT_FEATURE_EXTRACTOR,
    rf_options: Mapping[str, Any] | None = None,
) -> SegmentationResult:
    """Run feature extraction, sparse-label training, and dense segmentation."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in the range [0, 1].")
    labels = validate_inputs(image, scribbles)
    features = _validate_features(feature_extractor(np.asarray(image)))
    if features.shape[:2] != np.asarray(image).shape:
        raise ValueError(
            "Feature extractor changed the spatial shape: "
            f"image={np.asarray(image).shape}, features={features.shape[:2]}."
        )
    classifier = train_classifier(features, labels, rf_options=rf_options)
    probability = predict_pore_probability(classifier, features)
    segmentation = probability >= threshold
    return SegmentationResult(
        probability=probability,
        segmentation=segmentation,
        classifier=classifier,
        feature_shape=features.shape,
    )
