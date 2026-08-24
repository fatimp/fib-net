"""Floating-point probability-map input and output."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_probability_map(path: str | Path, probability: np.ndarray) -> Path:
    """Save a probability map as an unquantized float32 NumPy array."""
    output_path = Path(path)
    if output_path.suffix.lower() != ".npy":
        raise ValueError("Probability maps must use the .npy format.")
    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("A probability map must be a two-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError("A probability map cannot contain NaN or infinite values.")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("Probability values must be in the range [0, 1].")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array, allow_pickle=False)
    return output_path


def load_probability_map(path: str | Path) -> np.ndarray:
    """Load and validate a float32 probability map."""
    input_path = Path(path)
    if input_path.suffix.lower() != ".npy":
        raise ValueError("Probability maps must use the .npy format.")
    array = np.load(input_path, allow_pickle=False)
    if array.dtype != np.float32:
        raise ValueError(f"Expected float32 probability data, received {array.dtype}.")
    if array.ndim != 2:
        raise ValueError("A probability map must be a two-dimensional array.")
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("Probability values must be finite and in the range [0, 1].")
    return array
