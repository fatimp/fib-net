from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from fibnet.dataset import build_paired_samples
from fibnet.training import build_parser, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained pore model on a few stack-specific masks and segment the full stack."
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=Path("target"))
    parser.add_argument("--ideal-dir", type=Path, default=Path("ideal"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/adapted_stack")
    )
    parser.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Optional labeled slice stems to use. By default all paired files from ideal-dir are used.",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument(
        "--train-mode",
        choices=("full", "patch"),
        default="patch",
        help="Use patch for full-resolution crops without resizing the whole slice.",
    )
    parser.add_argument("--patches-per-image", type=int, default=24)
    parser.add_argument("--positive-patch-ratio", type=float, default=0.75)
    parser.add_argument("--min-positive-fraction", type=float, default=0.005)
    parser.add_argument(
        "--feature-mode", choices=("grayscale", "relief", "stack_relief"), default=None
    )
    parser.add_argument("--model-arch", choices=("unet", "resunet"), default=None)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--augmentation-mode",
        choices=(
            "none",
            "safe",
            "affine_noise",
            "optimization_geometric",
            "optimization_geometric_intensity",
        ),
        default="optimization_geometric_intensity",
    )
    parser.add_argument("--mask-threshold", type=int, default=0)
    parser.add_argument("--bce-weight", type=float, default=0.10)
    parser.add_argument("--dice-weight", type=float, default=0.65)
    parser.add_argument("--tversky-weight", type=float, default=0.25)
    parser.add_argument("--tversky-alpha", type=float, default=0.35)
    parser.add_argument("--tversky-beta", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--hysteresis-low-threshold", type=float, default=None)
    parser.add_argument("--inference-mode", choices=("tile", "whole"), default="tile")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--smooth-radius", type=int, default=0)
    parser.add_argument("--smooth-blend", type=float, default=0.0)
    parser.add_argument("--min-object-size-2d", type=int, default=0)
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply optional stack post-processing after inference (disabled by default).",
    )
    parser.add_argument(
        "--overlay-stems",
        nargs="*",
        default=None,
        help="Optional stems for overlays after postprocessing. Default: labeled stems plus first/middle/last stack slice.",
    )
    return parser.parse_args()


def checkpoint_training_args(checkpoint_path: Path, device: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint.get("args", {})


def resolve_model_settings(args: argparse.Namespace) -> tuple[str, str]:
    checkpoint_args = checkpoint_training_args(args.base_checkpoint, args.device)
    feature_mode = args.feature_mode or checkpoint_args.get("feature_mode", "grayscale")
    model_arch = args.model_arch or checkpoint_args.get("model_arch", "unet")
    return feature_mode, model_arch


def build_train_args(
    args: argparse.Namespace, feature_mode: str, model_arch: str
) -> argparse.Namespace:
    train_args = build_parser().parse_args([])
    train_args.images_dir = args.target_dir
    train_args.masks_dir = args.ideal_dir
    train_args.output_dir = args.output_dir / "finetune"
    train_args.image_size = args.image_size
    train_args.feature_mode = feature_mode
    train_args.model_arch = model_arch
    train_args.train_mode = args.train_mode
    train_args.patches_per_image = args.patches_per_image
    train_args.positive_patch_ratio = args.positive_patch_ratio
    train_args.min_positive_fraction = args.min_positive_fraction
    train_args.augmentation_mode = args.augmentation_mode
    train_args.mask_threshold = args.mask_threshold
    train_args.min_mask_fraction = 0.0001
    train_args.max_mask_fraction = 0.85
    train_args.batch_size = args.batch_size
    train_args.epochs = args.epochs
    train_args.learning_rate = args.learning_rate
    train_args.weight_decay = args.weight_decay
    train_args.bce_weight = args.bce_weight
    train_args.dice_weight = args.dice_weight
    train_args.tversky_weight = args.tversky_weight
    train_args.tversky_alpha = args.tversky_alpha
    train_args.tversky_beta = args.tversky_beta
    train_args.resume_checkpoint = args.base_checkpoint
    train_args.val_ratio = 0.0
    train_args.split_mode = "random"
    train_args.num_workers = 0
    train_args.seed = args.seed
    train_args.early_stopping_patience = max(args.epochs + 1, 10)
    train_args.min_delta = 1e-4
    train_args.amp = args.amp
    train_args.device = args.device
    return train_args


def select_samples(samples, stems: list[str] | None):
    if stems is None:
        return samples
    wanted = {stem.lower() for stem in stems}
    selected = [
        sample for sample in samples if sample.image_path.stem.lower() in wanted
    ]
    found = {sample.image_path.stem.lower() for sample in selected}
    missing = sorted(wanted - found)
    if missing:
        raise FileNotFoundError(
            f"No paired target/ideal masks found for stems: {', '.join(missing)}"
        )
    return selected


def collect_stack_stems(target_dir: Path) -> list[str]:
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
    paths = [
        path
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ]
    paths.sort(key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)
    return [path.stem for path in paths]


def default_overlay_stems(target_dir: Path, labeled_stems: list[str]) -> list[str]:
    stack_stems = collect_stack_stems(target_dir)
    preview = []
    if stack_stems:
        preview = [stack_stems[0], stack_stems[len(stack_stems) // 2], stack_stems[-1]]
    return sorted(
        set(labeled_stems + preview),
        key=lambda stem: int(stem) if stem.isdigit() else stem,
    )


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_mode, model_arch = resolve_model_settings(args)

    samples, missing_masks, missing_images = build_paired_samples(
        args.target_dir, args.ideal_dir
    )
    samples = select_samples(samples, args.stems)
    if not 3 <= len(samples) <= 8:
        print(
            "Warning: this scenario is tuned for roughly 3-5 labeled slices; "
            f"received {len(samples)} paired slices.",
            file=sys.stderr,
        )

    train_args = build_train_args(
        args, feature_mode=feature_mode, model_arch=model_arch
    )
    result = train_model(
        train_args,
        samples,
        [],
        missing_masks,
        missing_images,
        total_samples=len(samples),
        excluded_samples=[],
    )

    checkpoint = train_args.output_dir / "last_model.pt"
    probability_dir = args.output_dir / "probabilities"
    postprocess_dir = args.output_dir / "postprocessed"
    overlay_stems = args.overlay_stems or default_overlay_stems(
        args.target_dir,
        labeled_stems=[sample.image_path.stem for sample in samples],
    )

    run_command(
        [
            sys.executable,
            "-m",
            "fibnet.inference",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(args.target_dir),
            "--context-dir",
            str(args.target_dir),
            "--output-dir",
            str(probability_dir),
            "--feature-mode",
            feature_mode,
            "--model-arch",
            model_arch,
            "--mode",
            args.inference_mode,
            "--image-size",
            str(args.image_size),
            "--tile-size",
            str(args.tile_size or args.image_size),
            "--overlap",
            str(args.overlap),
            "--threshold",
            str(args.threshold),
            "--save-probability",
            "--device",
            args.device,
        ]
    )

    if args.postprocess:
        run_command(
            [
                sys.executable,
                "-m",
                "fibnet.postprocessing",
                "--prob-dir",
                str(probability_dir),
                "--target-dir",
                str(args.target_dir),
                "--output-dir",
                str(postprocess_dir),
                "--threshold",
                str(args.threshold),
                *(
                    [
                        "--hysteresis-low-threshold",
                        str(args.hysteresis_low_threshold),
                    ]
                    if args.hysteresis_low_threshold is not None
                    else []
                ),
                "--smooth-radius",
                str(args.smooth_radius),
                "--smooth-blend",
                str(args.smooth_blend),
                "--support-radius",
                "1",
                "--min-support",
                "1",
                "--min-object-size-2d",
                str(args.min_object_size_2d),
                "--save-smoothed-probability",
                "--save-overlay",
                "--overlay-stems",
                *overlay_stems,
            ]
        )

    summary = {
        "base_checkpoint": str(args.base_checkpoint),
        "fine_tuned_checkpoint": str(checkpoint),
        "feature_mode": feature_mode,
        "model_arch": model_arch,
        "labeled_stems": [sample.image_path.stem for sample in samples],
        "train_mode": args.train_mode,
        "patches_per_image": args.patches_per_image,
        "inference_mode": args.inference_mode,
        "tile_size": args.tile_size or args.image_size,
        "overlap": args.overlap,
        "probability_dir": str(probability_dir),
        "postprocessing_enabled": args.postprocess,
        "postprocess_dir": str(postprocess_dir) if args.postprocess else None,
        "overlay_stems": overlay_stems,
        "train_result": result,
        "threshold": args.threshold,
        "hysteresis_low_threshold": args.hysteresis_low_threshold,
        "smooth_radius": args.smooth_radius,
        "smooth_blend": args.smooth_blend,
        "min_object_size_2d": args.min_object_size_2d,
    }
    (args.output_dir / "adaptation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
