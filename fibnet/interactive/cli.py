"""Command-line entry point for sparse-scribble Random Forest segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .classifier import SegmentationResult, segment_image
from .io import load_grayscale_image, load_scribbles, save_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment a grayscale FIB-SEM image by training a Random Forest on "
            "sparse pore/solid scribbles."
        )
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--scribbles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def run(
    image_path: str | Path,
    scribble_path: str | Path,
    output_dir: str | Path,
    *,
    threshold: float = 0.5,
) -> SegmentationResult:
    """Run the CLI pipeline programmatically and save its outputs."""
    image = load_grayscale_image(image_path)
    scribbles = load_scribbles(scribble_path)
    result = segment_image(image, scribbles, threshold=threshold)
    save_outputs(output_dir, result.probability, result.segmentation)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(
            args.image,
            args.scribbles,
            args.output_dir,
            threshold=args.threshold,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    feature_mebibytes = (
        result.feature_shape[0]
        * result.feature_shape[1]
        * result.feature_shape[2]
        * 4
        / 1024**2
    )
    print(
        f"Saved segmentation to {args.output_dir} "
        f"(features={result.feature_shape}, {feature_mebibytes:.2f} MiB)."
    )


if __name__ == "__main__":
    main()
