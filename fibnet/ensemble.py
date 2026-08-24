"""Probability-level model ensembling."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from .probability_io import load_probability_map, save_probability_map


def arithmetic_probability_mean(probabilities: Iterable[np.ndarray]) -> np.ndarray:
    """Average identically shaped probability maps without 8-bit quantization."""
    arrays = [
        np.asarray(probability, dtype=np.float32) for probability in probabilities
    ]
    if not arrays:
        raise ValueError("At least one probability map is required.")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("All probability maps must have identical shapes.")
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("Probability maps must be two-dimensional.")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Probability maps cannot contain NaN or infinite values.")
    return np.mean(np.stack(arrays), axis=0, dtype=np.float32).astype(
        np.float32, copy=False
    )


def probability_files(directory: Path) -> dict[str, Path]:
    return {
        path.name.removesuffix("_prob.npy"): path
        for path in directory.glob("*_prob.npy")
    }


def _stem_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average probability maps from independently trained FIB-NET models."
    )
    parser.add_argument("--probability-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in the range [0, 1].")
    indices = [probability_files(directory) for directory in args.probability_dirs]
    common_stems = set(indices[0])
    for index in indices[1:]:
        common_stems &= set(index)
    if not common_stems:
        raise FileNotFoundError("No matching *_prob.npy files were found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem in sorted(common_stems, key=_stem_sort_key):
        ensemble = arithmetic_probability_mean(
            load_probability_map(index[stem]) for index in indices
        )
        save_probability_map(args.output_dir / f"{stem}_prob.npy", ensemble)
        mask = (ensemble >= args.threshold).astype(np.uint8) * 255
        Image.fromarray(mask, mode="L").save(args.output_dir / f"{stem}_pred.png")


if __name__ == "__main__":
    main()
