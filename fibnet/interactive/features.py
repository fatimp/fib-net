"""Pixel-feature extraction for interactive segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from skimage.feature import multiscale_basic_features
from skimage.util import img_as_float32


class FeatureExtractor(Protocol):
    """Interface implemented by replaceable pixel-feature extractors."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return an ``(height, width, feature_count)`` feature tensor."""
        ...


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional grayscale image, received shape {array.shape}."
        )
    if array.size == 0:
        raise ValueError("The grayscale image cannot be empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Expected numeric image data, received dtype {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError("The grayscale image cannot contain NaN or infinite values.")
    return img_as_float32(array)


def extract_multiscale_features(
    image: np.ndarray,
    *,
    sigma_min: float = 1,
    sigma_max: float = 16,
) -> np.ndarray:
    """Extract intensity, edge, and texture features at multiple scales."""
    if sigma_min <= 0:
        raise ValueError("sigma_min must be positive.")
    if sigma_max < sigma_min:
        raise ValueError("sigma_max must be greater than or equal to sigma_min.")

    array = _validate_image(image)
    features = multiscale_basic_features(
        array,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        channel_axis=None,
    )
    result = np.asarray(features, dtype=np.float32)
    if result.shape[:2] != array.shape or result.ndim != 3:
        raise RuntimeError(
            "The feature extractor returned an unexpected tensor shape "
            f"{result.shape} for image shape {array.shape}."
        )
    if not np.isfinite(result).all():
        raise ValueError(
            "Feature extraction produced NaN or infinite values; check the input image."
        )
    return result


@dataclass(frozen=True)
class MultiscaleBasicFeatureExtractor:
    """Configurable callable wrapper around scikit-image's MVP extractor."""

    sigma_min: float = 1
    sigma_max: float = 16

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return extract_multiscale_features(
            image,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
        )


DEFAULT_FEATURE_EXTRACTOR = MultiscaleBasicFeatureExtractor()
