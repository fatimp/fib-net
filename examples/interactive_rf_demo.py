"""Generate and segment a small synthetic image using sparse scribbles."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from fibnet.interactive import (
    PORE,
    SOLID,
    extract_multiscale_features,
    predict_pore_probability,
    train_classifier,
)
from fibnet.interactive.io import save_outputs


def make_synthetic_data(size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Create a textured two-phase image and sparse semantic labels."""
    rng = np.random.default_rng(42)
    y, x = np.mgrid[:size, :size]
    pore = (
        ((x - 39) ** 2 + (y - 42) ** 2 < 25**2)
        | ((x - 88) ** 2 + (y - 79) ** 2 < 29**2)
        | ((x - 31) ** 2 + (y - 103) ** 2 < 13**2)
    )
    background = 0.74 + 0.05 * np.sin(x / 10) + 0.03 * np.cos(y / 8)
    image = np.where(pore, background - 0.53, background)
    image += rng.normal(0.0, 0.025, image.shape)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    scribbles = np.zeros(image.shape, dtype=np.uint8)
    pore_points = [(42, 39), (31, 39), (53, 39), (79, 88), (68, 88), (90, 88)]
    solid_points = [(12, 12), (12, 64), (12, 115), (64, 12), (115, 64), (115, 115)]
    for row, column in pore_points:
        scribbles[row - 1 : row + 2, column - 1 : column + 2] = PORE
    for row, column in solid_points:
        scribbles[row - 1 : row + 2, column - 1 : column + 2] = SOLID
    return image, scribbles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/interactive_rf_demo"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, scribbles = make_synthetic_data()
    started = perf_counter()
    features = extract_multiscale_features(image)
    classifier = train_classifier(features, scribbles)
    probability = predict_pore_probability(classifier, features)
    segmentation = probability >= 0.5
    elapsed = perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(
        args.output_dir / "image.png"
    )
    Image.fromarray(scribbles).save(args.output_dir / "scribbles.png")
    save_outputs(args.output_dir, probability, segmentation)
    print(
        f"Synthetic RF segmentation complete in {elapsed:.3f} s; "
        f"feature tensor={features.shape}, dtype={features.dtype}, "
        f"size={features.nbytes / 1024**2:.2f} MiB; outputs={args.output_dir}"
    )


if __name__ == "__main__":
    main()
