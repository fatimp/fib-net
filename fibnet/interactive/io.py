"""Image, scribble-mask, and result I/O for interactive segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.util import img_as_float32

from ..probability_io import save_probability_map


class ScribbleLabel(IntEnum):
    """Semantic values stored in a single-channel scribble mask."""

    UNLABELED = 0
    PORE = 1
    SOLID = 2


UNLABELED = int(ScribbleLabel.UNLABELED)
PORE = int(ScribbleLabel.PORE)
SOLID = int(ScribbleLabel.SOLID)
_VALID_LABELS = np.array([UNLABELED, PORE, SOLID])


@dataclass(frozen=True)
class OutputPaths:
    """Paths written for one interactive segmentation result."""

    probability: Path
    probability_preview: Path
    segmentation: Path


def _read_single_channel(path: str | Path, description: str) -> np.ndarray:
    input_path = Path(path)
    try:
        with Image.open(input_path) as opened:
            array = np.asarray(opened)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not read {description} '{input_path}': {error}"
        ) from error
    if array.ndim != 2:
        raise ValueError(
            f"Expected a single-channel {description}, received shape {array.shape} "
            f"from '{input_path}'."
        )
    return array


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Load a single-channel image as finite float32 data."""
    array = _read_single_channel(path, "grayscale image")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"Expected numeric grayscale data, received dtype {array.dtype}."
        )
    if not np.isfinite(array).all():
        raise ValueError("The grayscale image cannot contain NaN or infinite values.")
    return img_as_float32(array)


def validate_scribbles(
    scribbles: np.ndarray,
    *,
    expected_shape: tuple[int, int] | None = None,
    require_both_classes: bool = True,
) -> np.ndarray:
    """Validate and return a uint8 semantic scribble-label array."""
    array = np.asarray(scribbles)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional scribble mask, received shape {array.shape}."
        )
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            "Image and scribble mask must have identical shapes: "
            f"image={expected_shape}, scribbles={array.shape}."
        )
    if array.size == 0:
        raise ValueError("The scribble mask cannot be empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"Expected numeric scribble labels, received dtype {array.dtype}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Scribble labels cannot contain NaN or infinite values.")

    unknown = np.unique(array[~np.isin(array, _VALID_LABELS)])
    if unknown.size:
        values = ", ".join(str(value) for value in unknown.tolist())
        raise ValueError(
            f"Unknown scribble label value(s): {values}. Expected only 0, 1, or 2."
        )
    if require_both_classes and not np.any(array == PORE):
        raise ValueError("At least one pore scribble (label 1) is required.")
    if require_both_classes and not np.any(array == SOLID):
        raise ValueError("At least one solid scribble (label 2) is required.")
    return array.astype(np.uint8, copy=False)


def load_scribbles(path: str | Path) -> np.ndarray:
    """Load a semantic scribble mask without interpreting display colors."""
    return validate_scribbles(_read_single_channel(path, "scribble mask"))


def validate_inputs(image: np.ndarray, scribbles: np.ndarray) -> np.ndarray:
    """Validate an in-memory image/scribble pair and return normalized labels."""
    image_array = np.asarray(image)
    if image_array.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional grayscale image, "
            f"received shape {image_array.shape}."
        )
    if image_array.size == 0:
        raise ValueError("The grayscale image cannot be empty.")
    if not np.issubdtype(image_array.dtype, np.number):
        raise ValueError(
            f"Expected numeric image data, received dtype {image_array.dtype}."
        )
    if not np.isfinite(image_array).all():
        raise ValueError("The grayscale image cannot contain NaN or infinite values.")
    return validate_scribbles(scribbles, expected_shape=image_array.shape)


def save_outputs(
    output_dir: str | Path,
    probability: np.ndarray,
    segmentation: np.ndarray,
) -> OutputPaths:
    """Write the lossless probability map and the two requested PNG outputs."""
    directory = Path(output_dir)
    probability_array = np.asarray(probability, dtype=np.float32)
    segmentation_array = np.asarray(segmentation)
    if probability_array.ndim != 2:
        raise ValueError("The probability map must be two-dimensional.")
    if segmentation_array.shape != probability_array.shape:
        raise ValueError(
            "Probability and segmentation shapes differ: "
            f"{probability_array.shape} != {segmentation_array.shape}."
        )
    if not np.isfinite(probability_array).all() or np.any(
        (probability_array < 0.0) | (probability_array > 1.0)
    ):
        raise ValueError("Probability values must be finite and in the range [0, 1].")

    probability_path = directory / "probability.npy"
    preview_path = directory / "probability.png"
    segmentation_path = directory / "segmentation.png"
    save_probability_map(probability_path, probability_array)
    preview = np.rint(probability_array * 255.0).astype(np.uint8)
    binary = np.where(segmentation_array.astype(bool), 255, 0).astype(np.uint8)
    Image.fromarray(preview).save(preview_path)
    Image.fromarray(binary).save(segmentation_path)
    return OutputPaths(probability_path, preview_path, segmentation_path)
