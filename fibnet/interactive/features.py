"""Pixel-feature extraction for interactive segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from skimage.feature import (
    hessian_matrix,
    hessian_matrix_eigvals,
    multiscale_basic_features,
    structure_tensor,
    structure_tensor_eigenvalues,
)
from skimage.filters import gaussian, laplace
from skimage.util import img_as_float32

FeatureMode = Literal["generic", "particleseg3d_style"]
FEATURE_MODES: tuple[FeatureMode, ...] = ("generic", "particleseg3d_style")


@dataclass(frozen=True)
class FeatureSet:
    """A feature tensor paired with stable, channel-aligned names."""

    feature_tensor: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.feature_tensor.ndim != 3:
            raise ValueError("feature_tensor must have three dimensions.")
        if len(self.feature_names) != self.feature_tensor.shape[-1]:
            raise ValueError(
                "feature_names must contain exactly one name per feature channel."
            )


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


def _format_sigma(sigma: float) -> str:
    return f"{sigma:g}".replace(".", "p")


def _generic_feature_names(
    sigma_min: float, sigma_max: float, feature_count: int
) -> tuple[str, ...]:
    num_sigma = int(np.log2(sigma_max / sigma_min) + 1)
    sigmas = np.logspace(np.log2(sigma_min), np.log2(sigma_max), num=num_sigma, base=2)
    names = []
    for sigma in sigmas:
        suffix = _format_sigma(float(sigma))
        names.extend(
            (
                f"gaussian_sigma{suffix}",
                f"gradient_magnitude_sigma{suffix}",
                f"hessian_eigenvalue_max_sigma{suffix}",
                f"hessian_eigenvalue_min_sigma{suffix}",
            )
        )
    if len(names) != feature_count:
        raise RuntimeError(
            "Could not align generic feature names with the scikit-image output: "
            f"names={len(names)}, channels={feature_count}."
        )
    return tuple(names)


def _normalized_intensity(image: np.ndarray) -> np.ndarray:
    array = _validate_image(image)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32, copy=False)


def extract_particleseg3d_style_features(
    image: np.ndarray,
    *,
    sigmas: tuple[float, ...] = (1, 2, 4, 8),
) -> FeatureSet:
    """Extract a compact 2D analogue of scribble-RF voxel features.

    The ParticleSeg3D paper describes its annotation classifier in terms of a
    broad handcrafted feature set rather than an exact public filter bank. This
    implementation therefore exposes an explicit, reproducible 2D analogue.
    """
    if not sigmas or any(sigma <= 0 for sigma in sigmas):
        raise ValueError("sigmas must contain at least one positive scale.")
    normalized = _normalized_intensity(image)
    channels_per_sigma = 12
    feature_count = 1 + channels_per_sigma * len(sigmas)
    features = np.empty((*normalized.shape, feature_count), dtype=np.float32)
    names: list[str] = []
    channel = 0

    def append(name: str, values: np.ndarray) -> None:
        nonlocal channel
        features[..., channel] = np.asarray(values, dtype=np.float32)
        names.append(name)
        channel += 1

    append("normalized_intensity", normalized)
    normalized_squared = np.square(normalized)
    for sigma in sigmas:
        suffix = _format_sigma(sigma)
        smoothed = gaussian(
            normalized,
            sigma=sigma,
            mode="reflect",
            preserve_range=True,
        ).astype(np.float32, copy=False)
        grad_y, grad_x = np.gradient(smoothed)
        grad_x = grad_x.astype(np.float32, copy=False)
        grad_y = grad_y.astype(np.float32, copy=False)
        grad_magnitude = np.hypot(grad_x, grad_y)
        grad_sin = np.divide(
            grad_y,
            grad_magnitude,
            out=np.zeros_like(grad_y),
            where=grad_magnitude > 0,
        )
        grad_cos = np.divide(
            grad_x,
            grad_magnitude,
            out=np.ones_like(grad_x),
            where=grad_magnitude > 0,
        )
        local_second_moment = gaussian(
            normalized_squared,
            sigma=sigma,
            mode="reflect",
            preserve_range=True,
        ).astype(np.float32, copy=False)
        local_variance = np.maximum(local_second_moment - np.square(smoothed), 0.0)
        local_std = np.sqrt(local_variance)

        hessian_elements = hessian_matrix(
            normalized,
            sigma=sigma,
            mode="reflect",
            order="rc",
            use_gaussian_derivatives=True,
        )
        hessian_eigenvalues = hessian_matrix_eigvals(hessian_elements)
        structure_elements = structure_tensor(
            normalized,
            sigma=sigma,
            mode="reflect",
            order="rc",
        )
        structure_eigenvalues = structure_tensor_eigenvalues(structure_elements)

        append(f"gaussian_sigma{suffix}", smoothed)
        append(f"grad_x_sigma{suffix}", grad_x)
        append(f"grad_y_sigma{suffix}", grad_y)
        append(f"grad_mag_sigma{suffix}", grad_magnitude)
        append(f"grad_sin_sigma{suffix}", grad_sin)
        append(f"grad_cos_sigma{suffix}", grad_cos)
        append(f"laplacian_sigma{suffix}", laplace(smoothed))
        append(f"local_std_sigma{suffix}", local_std)
        append(f"hessian_eigenvalue_max_sigma{suffix}", hessian_eigenvalues[0])
        append(f"hessian_eigenvalue_min_sigma{suffix}", hessian_eigenvalues[1])
        append(
            f"structure_tensor_eigenvalue_max_sigma{suffix}",
            structure_eigenvalues[0],
        )
        append(
            f"structure_tensor_eigenvalue_min_sigma{suffix}",
            structure_eigenvalues[1],
        )

    if channel != feature_count:
        raise RuntimeError(f"Generated {channel} channels, expected {feature_count}.")
    if not np.isfinite(features).all():
        raise ValueError(
            "ParticleSeg3D-style feature extraction produced non-finite values."
        )
    return FeatureSet(features, tuple(names))


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

    def extract(self, image: np.ndarray) -> FeatureSet:
        tensor = self(image)
        names = _generic_feature_names(self.sigma_min, self.sigma_max, tensor.shape[-1])
        return FeatureSet(tensor, names)


@dataclass(frozen=True)
class ParticleSeg3DStyleFeatureExtractor:
    """Callable 2D ParticleSeg3D-style handcrafted feature backend."""

    sigmas: tuple[float, ...] = (1, 2, 4, 8)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.extract(image).feature_tensor

    def extract(self, image: np.ndarray) -> FeatureSet:
        return extract_particleseg3d_style_features(image, sigmas=self.sigmas)


DEFAULT_FEATURE_EXTRACTOR = MultiscaleBasicFeatureExtractor()
PARTICLESEG3D_STYLE_FEATURE_EXTRACTOR = ParticleSeg3DStyleFeatureExtractor()


def get_feature_extractor(
    mode: FeatureMode,
) -> MultiscaleBasicFeatureExtractor | ParticleSeg3DStyleFeatureExtractor:
    """Return a configured extractor through the shared mode interface."""
    if mode == "generic":
        return DEFAULT_FEATURE_EXTRACTOR
    if mode == "particleseg3d_style":
        return PARTICLESEG3D_STYLE_FEATURE_EXTRACTOR
    raise ValueError(f"Unknown feature mode {mode!r}; expected one of {FEATURE_MODES}.")


def extract_features(image: np.ndarray, mode: FeatureMode = "generic") -> FeatureSet:
    """Extract a named feature set with the selected backend."""
    return get_feature_extractor(mode).extract(image)
