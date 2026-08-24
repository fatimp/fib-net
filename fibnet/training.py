from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    GridPatchPoreSegmentationDataset,
    PoreSegmentationDataset,
    RandomPatchPoreSegmentationDataset,
    build_paired_samples,
    estimate_mask_fraction,
    get_group_ids,
    split_samples,
    split_samples_by_group,
)
from .image_features import feature_channels
from .losses import BCEDiceLoss
from .metrics import dice_coefficient, iou_score
from .model import build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a pore segmentation model.")
    parser.add_argument("--images-dir", type=Path, default=Path("data/original"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/segmented"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=("grayscale", "relief", "stack_relief"),
        default="grayscale",
    )
    parser.add_argument(
        "--model-arch", type=str, choices=("unet", "resunet"), default="unet"
    )
    parser.add_argument(
        "--train-mode", type=str, choices=("full", "patch"), default="full"
    )
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--positive-patch-ratio", type=float, default=0.7)
    parser.add_argument("--min-positive-fraction", type=float, default=0.01)
    parser.add_argument(
        "--augmentation-mode",
        type=str,
        choices=(
            "none",
            "safe",
            "all",
            "affine_noise",
            "optimization_geometric",
            "optimization_geometric_intensity",
        ),
        default="safe",
    )
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--min-mask-fraction", type=float, default=0.0)
    parser.add_argument("--max-mask-fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--auto-pos-weight", action="store_true")
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--bce-weight", type=float, default=0.4)
    parser.add_argument("--dice-weight", type=float, default=0.4)
    parser.add_argument("--tversky-weight", type=float, default=0.2)
    parser.add_argument("--tversky-alpha", type=float, default=0.4)
    parser.add_argument("--tversky-beta", type=float, default=0.6)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--continue-checkpoint",
        type=Path,
        default=None,
        help="Resume an interrupted run, including optimiser and scheduler state.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode", type=str, choices=("random", "group"), default="group"
    )
    parser.add_argument("--train-groups", nargs="*", default=None)
    parser.add_argument("--val-groups", nargs="*", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--scheduler-min-lr", type=float, default=0.0)
    parser.add_argument(
        "--val-mode",
        choices=("full_resize", "grid_patch"),
        default="full_resize",
    )
    parser.add_argument("--val-overlap", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        self.best_value = float("inf")
        self.bad_epochs = 0

    def step(self, current_value: float) -> bool:
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    optimizer: Adam | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    losses: list[float] = []
    dices: list[float] = []
    ious: list[float] = []

    progress = tqdm(dataloader, desc="train" if is_training else "val", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_training and scaler is not None:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        dices.append(dice_coefficient(logits.detach(), masks))
        ious.append(iou_score(logits.detach(), masks))
        progress.set_postfix(
            loss=f"{losses[-1]:.4f}", dice=f"{dices[-1]:.4f}", iou=f"{ious[-1]:.4f}"
        )

    if not losses:
        return {"loss": float("nan"), "dice": float("nan"), "iou": float("nan")}

    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
    }


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def estimate_foreground_fraction(samples, mask_threshold: int) -> float:
    fractions = [
        estimate_mask_fraction(sample.mask_path, positive_threshold=mask_threshold)
        for sample in samples
    ]
    if not fractions:
        return 0.0
    return float(np.mean(fractions))


def filter_samples_by_mask_fraction(samples, args: argparse.Namespace):
    kept_samples = []
    excluded_samples = []
    for sample in samples:
        mask_fraction = estimate_mask_fraction(
            sample.mask_path, positive_threshold=args.mask_threshold
        )
        if args.min_mask_fraction <= mask_fraction <= args.max_mask_fraction:
            kept_samples.append(sample)
        else:
            excluded_samples.append(
                {
                    "sample_id": sample.sample_id,
                    "mask_path": str(sample.mask_path),
                    "mask_fraction": mask_fraction,
                }
            )

    if not kept_samples:
        raise ValueError(
            "No samples remain after mask-fraction filtering. "
            f"Requested range: [{args.min_mask_fraction}, {args.max_mask_fraction}]"
        )
    return kept_samples, excluded_samples


def build_checkpoint_payload(
    model: torch.nn.Module,
    args: argparse.Namespace,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
) -> dict:
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "model_state_dict": model.state_dict(),
        "args": serializable_args,
        "epoch": epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
    }


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def choose_split(
    samples,
    args: argparse.Namespace,
):
    all_samples = list(samples)
    if args.train_groups:
        available_groups = get_group_ids(all_samples)
        train_group_set = {group.lower() for group in args.train_groups}
        samples = [
            sample
            for sample in all_samples
            if sample.group_id.lower() in train_group_set
        ]
        if not samples:
            raise ValueError(
                f"No samples found for train_groups={args.train_groups}. "
                f"Available groups: {available_groups}"
            )

    if args.val_groups:
        val_group_set = {group.lower() for group in args.val_groups}
        if args.train_groups and train_group_set & val_group_set:
            overlap = sorted(train_group_set & val_group_set)
            raise ValueError(f"train_groups and val_groups overlap: {overlap}")
        val_samples = [
            sample for sample in all_samples if sample.group_id.lower() in val_group_set
        ]
        if not val_samples:
            raise ValueError(
                f"No samples found for val_groups={args.val_groups}. "
                f"Available groups: {get_group_ids(all_samples)}"
            )
        return samples, val_samples
    if args.split_mode == "group":
        return split_samples_by_group(samples, val_ratio=args.val_ratio, seed=args.seed)
    return split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)


def train_model(
    args: argparse.Namespace,
    train_samples,
    val_samples,
    missing_masks,
    missing_images,
    total_samples: int | None = None,
    excluded_samples: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.train_mode == "patch":
        train_dataset = RandomPatchPoreSegmentationDataset(
            train_samples,
            patch_size=args.image_size,
            patches_per_image=args.patches_per_image,
            positive_patch_ratio=args.positive_patch_ratio,
            min_positive_fraction=args.min_positive_fraction,
            augment=True,
            feature_mode=args.feature_mode,
            mask_threshold=args.mask_threshold,
            augmentation_mode=args.augmentation_mode,
        )
    else:
        train_dataset = PoreSegmentationDataset(
            train_samples,
            image_size=args.image_size,
            augment=True,
            feature_mode=args.feature_mode,
            mask_threshold=args.mask_threshold,
            augmentation_mode=args.augmentation_mode,
        )
    if args.val_mode == "grid_patch":
        val_dataset = GridPatchPoreSegmentationDataset(
            val_samples,
            patch_size=args.image_size,
            overlap=args.val_overlap,
            feature_mode=args.feature_mode,
            mask_threshold=args.mask_threshold,
        )
    else:
        val_dataset = PoreSegmentationDataset(
            val_samples,
            image_size=args.image_size,
            augment=False,
            feature_mode=args.feature_mode,
            mask_threshold=args.mask_threshold,
            augmentation_mode="none",
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        )

    foreground_fraction = estimate_foreground_fraction(
        train_samples, args.mask_threshold
    )
    pos_weight = args.pos_weight
    if args.auto_pos_weight and foreground_fraction > 0.0:
        pos_weight = min(
            (1.0 - foreground_fraction) / foreground_fraction, args.max_pos_weight
        )

    model = build_model(
        args.model_arch, in_channels=feature_channels(args.feature_mode)
    ).to(args.device)
    if args.resume_checkpoint is not None and args.continue_checkpoint is not None:
        raise ValueError(
            "Use either --resume-checkpoint for transfer learning or "
            "--continue-checkpoint for interrupted-run recovery, not both."
        )
    initial_checkpoint_path = args.continue_checkpoint or args.resume_checkpoint
    initial_checkpoint = None
    if initial_checkpoint_path is not None:
        checkpoint = torch.load(
            initial_checkpoint_path, map_location=args.device, weights_only=False
        )
        initial_checkpoint = checkpoint
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_feature_mode = checkpoint_args.get("feature_mode", "grayscale")
        checkpoint_model_arch = checkpoint_args.get("model_arch", "unet")
        if checkpoint_feature_mode != args.feature_mode:
            raise ValueError(
                "resume-checkpoint feature_mode does not match current feature_mode: "
                f"{checkpoint_feature_mode!r} != {args.feature_mode!r}"
            )
        if checkpoint_model_arch != args.model_arch:
            raise ValueError(
                "resume-checkpoint model_arch does not match current model_arch: "
                f"{checkpoint_model_arch!r} != {args.model_arch!r}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])

    criterion = BCEDiceLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        tversky_weight=args.tversky_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        pos_weight=pos_weight,
    ).to(args.device)
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=args.scheduler_min_lr,
    )
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience, min_delta=args.min_delta
    )
    use_amp = bool(args.amp and args.device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    if args.continue_checkpoint is not None:
        assert initial_checkpoint is not None
        required_states = ("optimizer_state_dict", "scheduler_state_dict")
        missing_states = [
            name for name in required_states if name not in initial_checkpoint
        ]
        if missing_states:
            raise ValueError(
                "Continuation checkpoint lacks required state: "
                + ", ".join(missing_states)
            )
        optimizer.load_state_dict(initial_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(initial_checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in initial_checkpoint:
            scaler.load_state_dict(initial_checkpoint["scaler_state_dict"])
        best_val_loss = float(initial_checkpoint.get("best_val_loss", float("inf")))
        history = list(initial_checkpoint.get("history", []))
        start_epoch = int(initial_checkpoint["epoch"]) + 1
        early_stopping.best_value = float(
            initial_checkpoint.get("early_stopping_best_value", best_val_loss)
        )
        early_stopping.bad_epochs = int(
            initial_checkpoint.get("early_stopping_bad_epochs", 0)
        )
        restore_rng_state(initial_checkpoint.get("rng_state"))
    train_groups = sorted({sample.group_id for sample in train_samples})
    val_groups = sorted({sample.group_id for sample in val_samples})

    metadata = {
        "num_total_pairs": total_samples
        if total_samples is not None
        else len(train_samples) + len(val_samples),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "split_mode": args.split_mode,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "foreground_fraction": foreground_fraction,
        "mask_threshold": args.mask_threshold,
        "min_mask_fraction": args.min_mask_fraction,
        "max_mask_fraction": args.max_mask_fraction,
        "num_excluded_samples": len(excluded_samples or []),
        "excluded_samples": excluded_samples or [],
        "pos_weight": pos_weight,
        "missing_masks": [str(path) for path in missing_masks],
        "missing_images": [str(path) for path in missing_images],
    }
    save_json(args.output_dir / "dataset_summary.json", metadata)
    save_json(
        args.output_dir / "run_config.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            args.device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_metrics = (
            run_epoch(model, val_loader, criterion, args.device, use_amp=use_amp)
            if val_loader is not None
            else {
                "loss": train_metrics["loss"],
                "dice": train_metrics["dice"],
                "iou": train_metrics["iou"],
            }
        )
        scheduler.step(val_metrics["loss"])

        epoch_result = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f}, dice {train_metrics['dice']:.4f}, iou {train_metrics['iou']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, dice {val_metrics['dice']:.4f}, iou {val_metrics['iou']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint = build_checkpoint_payload(
                model=model,
                args=args,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_loss=best_val_loss,
            )
            torch.save(checkpoint, args.output_dir / "best_model.pt")

        last_checkpoint = build_checkpoint_payload(
            model=model,
            args=args,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val_loss=best_val_loss,
        )
        last_checkpoint.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
                "early_stopping_best_value": early_stopping.best_value,
                "early_stopping_bad_epochs": early_stopping.bad_epochs,
                "rng_state": capture_rng_state(),
            }
        )
        torch.save(last_checkpoint, args.output_dir / "last_model.pt")

        save_json(args.output_dir / "history.json", {"epochs": history})

        if val_loader is not None and early_stopping.step(val_metrics["loss"]):
            print(
                f"Early stopping at epoch {epoch:03d} after "
                f"{args.early_stopping_patience} epochs without validation improvement."
            )
            break

    best_epoch = min(history, key=lambda item: item["val_loss"]) if history else None
    summary = {
        "best_epoch": best_epoch,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "train_groups": train_groups,
        "val_groups": val_groups,
    }
    save_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    samples, missing_masks, missing_images = build_paired_samples(
        args.images_dir, args.masks_dir
    )
    samples, excluded_samples = filter_samples_by_mask_fraction(samples, args)
    train_samples, val_samples = choose_split(samples, args)
    train_model(
        args,
        train_samples,
        val_samples,
        missing_masks,
        missing_images,
        total_samples=len(samples) + len(excluded_samples),
        excluded_samples=excluded_samples,
    )


if __name__ == "__main__":
    main()
