from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFilter

FEATURE_CHANNELS = {
    "grayscale": 1,
    "relief": 4,
    "stack_relief": 6,
}


def feature_channels(feature_mode: str) -> int:
    try:
        return FEATURE_CHANNELS[feature_mode]
    except KeyError as exc:
        valid_modes = ", ".join(sorted(FEATURE_CHANNELS))
        raise ValueError(
            f"Unknown feature_mode={feature_mode!r}. Expected one of: {valid_modes}"
        ) from exc


def image_to_feature_array(
    image: Image.Image,
    feature_mode: str = "grayscale",
    image_mean: float = 0.5,
    image_std: float = 0.5,
) -> np.ndarray:
    gray = image.convert("L")
    gray_array = np.asarray(gray, dtype=np.float32) / 255.0
    normalized_gray = (gray_array - image_mean) / image_std

    if feature_mode == "grayscale":
        return normalized_gray[None, :, :].astype(np.float32)

    if feature_mode == "stack_relief":
        return image_stack_to_feature_array(
            previous=image,
            current=image,
            next_image=image,
            image_mean=image_mean,
            image_std=image_std,
        )

    if feature_mode != "relief":
        feature_channels(feature_mode)

    blur_small = (
        np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float32)
        / 255.0
    )
    blur_large = (
        np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=8.0)), dtype=np.float32)
        / 255.0
    )
    local_contrast = (blur_small - blur_large) * 4.0

    grad_y, grad_x = np.gradient(blur_small)
    gradient_magnitude = np.sqrt((grad_x * grad_x) + (grad_y * grad_y)) * 8.0

    # In FIB-SEM slices, depressions often appear as asymmetric dark/light ramps.
    # The vertical derivative is a cheap cue for this relief-like perspective.
    vertical_relief = grad_y * 8.0

    return np.stack(
        [
            normalized_gray,
            np.clip(local_contrast, -1.0, 1.0),
            np.clip(gradient_magnitude, 0.0, 1.0),
            np.clip(vertical_relief, -1.0, 1.0),
        ],
        axis=0,
    ).astype(np.float32)


def image_to_feature_tensor(
    image: Image.Image,
    feature_mode: str = "grayscale",
    image_mean: float = 0.5,
    image_std: float = 0.5,
) -> torch.Tensor:
    return torch.from_numpy(
        image_to_feature_array(
            image,
            feature_mode=feature_mode,
            image_mean=image_mean,
            image_std=image_std,
        )
    )


def _normalized_gray_array(
    image: Image.Image, image_mean: float, image_std: float
) -> np.ndarray:
    gray = image.convert("L")
    gray_array = np.asarray(gray, dtype=np.float32) / 255.0
    return (gray_array - image_mean) / image_std


def image_stack_to_feature_array(
    previous: Image.Image,
    current: Image.Image,
    next_image: Image.Image,
    image_mean: float = 0.5,
    image_std: float = 0.5,
) -> np.ndarray:
    current_gray = current.convert("L")
    current_array = np.asarray(current_gray, dtype=np.float32) / 255.0
    blur_small = (
        np.asarray(
            current_gray.filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float32
        )
        / 255.0
    )
    blur_large = (
        np.asarray(
            current_gray.filter(ImageFilter.GaussianBlur(radius=8.0)), dtype=np.float32
        )
        / 255.0
    )
    local_contrast = (blur_small - blur_large) * 4.0

    grad_y, grad_x = np.gradient(blur_small)
    gradient_magnitude = np.sqrt((grad_x * grad_x) + (grad_y * grad_y)) * 8.0
    vertical_relief = grad_y * 8.0

    previous_array = _normalized_gray_array(previous, image_mean, image_std)
    current_normalized = (current_array - image_mean) / image_std
    next_array = _normalized_gray_array(next_image, image_mean, image_std)

    return np.stack(
        [
            previous_array,
            current_normalized,
            next_array,
            np.clip(local_contrast, -1.0, 1.0),
            np.clip(gradient_magnitude, 0.0, 1.0),
            np.clip(vertical_relief, -1.0, 1.0),
        ],
        axis=0,
    ).astype(np.float32)


def image_stack_to_feature_tensor(
    previous: Image.Image,
    current: Image.Image,
    next_image: Image.Image,
    image_mean: float = 0.5,
    image_std: float = 0.5,
) -> torch.Tensor:
    return torch.from_numpy(
        image_stack_to_feature_array(
            previous=previous,
            current=current,
            next_image=next_image,
            image_mean=image_mean,
            image_std=image_std,
        )
    )
