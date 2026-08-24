from __future__ import annotations

import math
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .image_features import image_stack_to_feature_tensor, image_to_feature_tensor

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


@dataclass(frozen=True)
class SamplePair:
    image_path: Path
    mask_path: Path
    sample_id: str

    @property
    def group_id(self) -> str:
        return self.sample_id.split("/", 1)[0]


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _normalize_stem(stem: str) -> str:
    normalized = stem.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r" 2$", "", normalized)
    normalized = re.sub(r"_mask$", "", normalized)
    normalized = re.sub(r"-mask$", "", normalized)
    normalized = re.sub(r" mask$", "", normalized)
    return normalized


def _extract_numeric_id(stem: str) -> str | None:
    matches = re.findall(r"(\d+)", stem)
    if not matches:
        return None
    return str(int(matches[-1]))


def _build_index(root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in _iter_files(root):
        relative_parent = str(path.relative_to(root).parent).replace("\\", "/").lower()
        key = (relative_parent, _normalize_stem(path.stem))
        index[key] = path
    return index


def _build_numeric_index(root: Path) -> dict[tuple[str, str], list[Path]]:
    index: dict[tuple[str, str], list[Path]] = {}
    for path in _iter_files(root):
        relative_parent = str(path.relative_to(root).parent).replace("\\", "/").lower()
        numeric_id = _extract_numeric_id(_normalize_stem(path.stem))
        if numeric_id is None:
            continue
        key = (relative_parent, numeric_id)
        index.setdefault(key, []).append(path)
    return index


def build_paired_samples(
    images_root: str | Path = "data/original",
    masks_root: str | Path = "data/segmented",
) -> tuple[list[SamplePair], list[Path], list[Path]]:
    images_root = Path(images_root)
    masks_root = Path(masks_root)

    if not images_root.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_root}")
    if not masks_root.exists():
        raise FileNotFoundError(f"Masks directory does not exist: {masks_root}")

    image_index = _build_index(images_root)
    mask_index = _build_index(masks_root)
    image_numeric_index = _build_numeric_index(images_root)
    mask_numeric_index = _build_numeric_index(masks_root)

    image_keys = set(image_index)
    mask_keys = set(mask_index)

    samples: list[SamplePair] = []
    used_image_paths: set[Path] = set()
    used_mask_paths: set[Path] = set()

    for key in sorted(image_keys & mask_keys):
        image_path = image_index[key]
        mask_path = mask_index[key]
        samples.append(
            SamplePair(
                image_path=image_path,
                mask_path=mask_path,
                sample_id=f"{key[0]}/{key[1]}",
            )
        )
        used_image_paths.add(image_path)
        used_mask_paths.add(mask_path)

    numeric_keys = set(image_numeric_index) & set(mask_numeric_index)
    for parent, numeric_id in sorted(numeric_keys):
        image_candidates = [
            path
            for path in image_numeric_index[(parent, numeric_id)]
            if path not in used_image_paths
        ]
        mask_candidates = [
            path
            for path in mask_numeric_index[(parent, numeric_id)]
            if path not in used_mask_paths
        ]
        if len(image_candidates) != 1 or len(mask_candidates) != 1:
            continue
        image_path = image_candidates[0]
        mask_path = mask_candidates[0]
        samples.append(
            SamplePair(
                image_path=image_path,
                mask_path=mask_path,
                sample_id=f"{parent}/{_normalize_stem(image_path.stem)}",
            )
        )
        used_image_paths.add(image_path)
        used_mask_paths.add(mask_path)

    samples.sort(key=lambda sample: sample.sample_id)

    missing_masks = [
        image_index[key]
        for key in sorted(image_keys)
        if image_index[key] not in used_image_paths
    ]
    missing_images = [
        mask_index[key]
        for key in sorted(mask_keys)
        if mask_index[key] not in used_mask_paths
    ]
    return samples, missing_masks, missing_images


def split_samples(
    samples: list[SamplePair],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[SamplePair], list[SamplePair]]:
    if not samples:
        raise ValueError("No paired samples found.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in the range [0.0, 1.0).")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    if val_ratio > 0.0 and val_size == 0:
        val_size = 1
    if len(shuffled) > 1 and val_size >= len(shuffled):
        val_size = len(shuffled) - 1

    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]
    return train_samples, val_samples


def split_samples_by_group(
    samples: list[SamplePair],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[SamplePair], list[SamplePair]]:
    if not samples:
        raise ValueError("No paired samples found.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in the range [0.0, 1.0).")

    grouped_samples: dict[str, list[SamplePair]] = {}
    for sample in samples:
        grouped_samples.setdefault(sample.group_id, []).append(sample)

    groups = list(grouped_samples)
    random.Random(seed).shuffle(groups)

    target_val_size = len(samples) * val_ratio
    val_group_ids: set[str] = set()
    val_count = 0

    for group_id in groups:
        if val_ratio == 0.0:
            break
        if val_count >= target_val_size and val_group_ids:
            break
        val_group_ids.add(group_id)
        val_count += len(grouped_samples[group_id])

    if len(val_group_ids) == len(groups) and len(groups) > 1:
        removed_group = groups[-1]
        val_group_ids.remove(removed_group)

    train_samples = [
        sample for sample in samples if sample.group_id not in val_group_ids
    ]
    val_samples = [sample for sample in samples if sample.group_id in val_group_ids]
    return train_samples, val_samples


def split_samples_by_explicit_groups(
    samples: list[SamplePair],
    val_groups: Iterable[str],
) -> tuple[list[SamplePair], list[SamplePair]]:
    val_group_set = {group.lower() for group in val_groups}
    train_samples = [
        sample for sample in samples if sample.group_id.lower() not in val_group_set
    ]
    val_samples = [
        sample for sample in samples if sample.group_id.lower() in val_group_set
    ]
    if not val_samples:
        raise ValueError("No validation samples found for the requested groups.")
    if not train_samples:
        raise ValueError(
            "No training samples remain after selecting validation groups."
        )
    return train_samples, val_samples


def get_group_ids(samples: list[SamplePair]) -> list[str]:
    return sorted({sample.group_id for sample in samples})


def _sample_sequence_key(sample: SamplePair) -> tuple[str, int, str]:
    numeric_id = _extract_numeric_id(sample.image_path.stem)
    numeric_value = int(numeric_id) if numeric_id is not None else 0
    return sample.group_id, numeric_value, sample.sample_id


def _path_sequence_key(path: Path) -> tuple[int, str]:
    numeric_id = _extract_numeric_id(path.stem)
    numeric_value = int(numeric_id) if numeric_id is not None else 0
    return numeric_value, path.name.lower()


def _build_neighbor_image_paths(
    samples: list[SamplePair],
) -> dict[str, tuple[Path, Path]]:
    by_parent: dict[Path, list[SamplePair]] = {}
    for sample in samples:
        by_parent.setdefault(sample.image_path.parent, []).append(sample)

    neighbors: dict[str, tuple[Path, Path]] = {}
    for parent, parent_samples in by_parent.items():
        ordered_paths = sorted(
            [
                path
                for path in parent.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=_path_sequence_key,
        )
        path_indices = {
            path.resolve(): index for index, path in enumerate(ordered_paths)
        }
        for sample in sorted(parent_samples, key=_sample_sequence_key):
            index = path_indices.get(sample.image_path.resolve())
            if index is None:
                neighbors[sample.sample_id] = (sample.image_path, sample.image_path)
                continue
            previous_path = ordered_paths[index - 1] if index > 0 else sample.image_path
            next_path = (
                ordered_paths[index + 1]
                if index < len(ordered_paths) - 1
                else sample.image_path
            )
            neighbors[sample.sample_id] = (previous_path, next_path)
    return neighbors


def mask_to_array(mask: Image.Image, positive_threshold: int = 127) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=np.float32)
    return (mask_array <= float(positive_threshold)).astype(np.float32)


def estimate_mask_fraction(mask_path: Path, positive_threshold: int = 127) -> float:
    mask = Image.open(mask_path).convert("L")
    return float(mask_to_array(mask, positive_threshold=positive_threshold).mean())


class PoreSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[SamplePair],
        image_size: int = 256,
        augment: bool = False,
        image_mean: float = 0.5,
        image_std: float = 0.5,
        feature_mode: str = "grayscale",
        mask_threshold: int = 127,
        augmentation_mode: str = "safe",
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.image_mean = image_mean
        self.image_std = image_std
        self.feature_mode = feature_mode
        self.mask_threshold = mask_threshold
        self.augmentation_mode = augmentation_mode
        self.neighbor_image_paths = _build_neighbor_image_paths(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("L")
        mask = Image.open(sample.mask_path).convert("L")
        previous_path, next_path = self.neighbor_image_paths[sample.sample_id]
        previous_image = Image.open(previous_path).convert("L")
        next_image = Image.open(next_path).convert("L")

        if self.image_size > 0:
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            previous_image = previous_image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            next_image = next_image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            mask = mask.resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )

        if self.augment:
            (previous_image, image, next_image), mask = (
                self._apply_augmentations_to_images(
                    [previous_image, image, next_image],
                    mask,
                    mode=self.augmentation_mode,
                )
            )

        mask_array = mask_to_array(mask, positive_threshold=self.mask_threshold)

        if self.feature_mode == "stack_relief":
            image_tensor = image_stack_to_feature_tensor(
                previous_image,
                image,
                next_image,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )
        else:
            image_tensor = image_to_feature_tensor(
                image,
                feature_mode=self.feature_mode,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
        }

    @staticmethod
    def _apply_augmentations(
        image: Image.Image,
        mask: Image.Image,
        mode: str = "safe",
    ) -> tuple[Image.Image, Image.Image]:
        images, mask = PoreSegmentationDataset._apply_augmentations_to_images(
            [image], mask, mode=mode
        )
        return images[0], mask

    @staticmethod
    def _apply_augmentations_to_images(
        images: list[Image.Image],
        mask: Image.Image,
        mode: str = "safe",
    ) -> tuple[list[Image.Image], Image.Image]:
        if mode == "none":
            return images, mask
        if mode not in {
            "safe",
            "all",
            "affine_noise",
            "optimization_geometric",
            "optimization_geometric_intensity",
        }:
            raise ValueError(
                "augmentation_mode must be one of: none, safe, all, affine_noise, "
                "optimization_geometric, optimization_geometric_intensity."
            )

        if mode.startswith("optimization_"):
            images, mask = PoreSegmentationDataset._apply_optimization_geometry(
                images, mask
            )
            if mode == "optimization_geometric_intensity":
                images = [
                    PoreSegmentationDataset._apply_optimization_intensity(image)
                    for image in images
                ]
            return images, mask

        if random.random() < 0.5:
            images = [
                image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images
            ]
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if mode == "affine_noise":
            images, mask = PoreSegmentationDataset._apply_random_affine(images, mask)
            images = [PoreSegmentationDataset._apply_noise(image) for image in images]
            return images, mask

        if mode == "all":
            if random.random() < 0.5:
                images = [
                    image.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for image in images
                ]
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            rotations = random.randint(0, 3)
            if rotations == 1:
                images = [
                    image.transpose(Image.Transpose.ROTATE_90) for image in images
                ]
                mask = mask.transpose(Image.Transpose.ROTATE_90)
            elif rotations == 2:
                images = [
                    image.transpose(Image.Transpose.ROTATE_180) for image in images
                ]
                mask = mask.transpose(Image.Transpose.ROTATE_180)
            elif rotations == 3:
                images = [
                    image.transpose(Image.Transpose.ROTATE_270) for image in images
                ]
                mask = mask.transpose(Image.Transpose.ROTATE_270)
        return images, mask

    @staticmethod
    def _apply_random_affine(
        images: list[Image.Image],
        mask: Image.Image,
        max_degrees: float = 7.0,
        max_translate_fraction: float = 0.04,
        scale_range: tuple[float, float] = (0.92, 1.08),
        max_shear_degrees: float = 5.0,
    ) -> tuple[list[Image.Image], Image.Image]:
        width, height = images[0].size
        angle = math.radians(random.uniform(-max_degrees, max_degrees))
        shear = math.radians(random.uniform(-max_shear_degrees, max_shear_degrees))
        scale = random.uniform(*scale_range)
        translate_x = (
            random.uniform(-max_translate_fraction, max_translate_fraction) * width
        )
        translate_y = (
            random.uniform(-max_translate_fraction, max_translate_fraction) * height
        )

        cos_angle = math.cos(angle) * scale
        sin_angle = math.sin(angle) * scale
        shear_factor = math.tan(shear)
        matrix = np.array(
            [
                [
                    cos_angle + (shear_factor * sin_angle),
                    -sin_angle + (shear_factor * cos_angle),
                ],
                [sin_angle, cos_angle],
            ],
            dtype=np.float64,
        )
        inverse = np.linalg.inv(matrix)
        center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
        translation = np.array([translate_x, translate_y], dtype=np.float64)
        offset = center - (inverse @ (center + translation))
        coefficients = (
            float(inverse[0, 0]),
            float(inverse[0, 1]),
            float(offset[0]),
            float(inverse[1, 0]),
            float(inverse[1, 1]),
            float(offset[1]),
        )

        transformed_images = [
            image.transform(
                image.size,
                Image.Transform.AFFINE,
                coefficients,
                resample=Image.Resampling.BILINEAR,
                fillcolor=0,
            )
            for image in images
        ]
        transformed_mask = mask.transform(
            mask.size,
            Image.Transform.AFFINE,
            coefficients,
            resample=Image.Resampling.NEAREST,
            fillcolor=255,
        )
        return transformed_images, transformed_mask

    @staticmethod
    def _apply_noise(
        image: Image.Image, sigma_range: tuple[float, float] = (2.0, 10.0)
    ) -> Image.Image:
        sigma = random.uniform(*sigma_range)
        image_array = np.asarray(image, dtype=np.float32)
        noise = np.random.normal(0.0, sigma, image_array.shape).astype(np.float32)
        noisy = np.clip(image_array + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(noisy, mode="L")

    @staticmethod
    def _apply_optimization_geometry(
        images: list[Image.Image], mask: Image.Image
    ) -> tuple[list[Image.Image], Image.Image]:
        if random.random() < 0.5:
            images = [
                image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images
            ]
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            images = [
                image.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for image in images
            ]
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            images = [image.transpose(Image.Transpose.ROTATE_180) for image in images]
            mask = mask.transpose(Image.Transpose.ROTATE_180)

        width, height = images[0].size
        scale = random.uniform(0.95, 1.05)
        translate_x = random.uniform(-0.04, 0.04) * width
        translate_y = random.uniform(-0.04, 0.04) * height
        inverse_scale = 1.0 / scale
        center_x = width * 0.5
        center_y = height * 0.5
        coefficients = (
            inverse_scale,
            0.0,
            center_x - inverse_scale * (center_x + translate_x),
            0.0,
            inverse_scale,
            center_y - inverse_scale * (center_y + translate_y),
        )
        images = [
            image.transform(
                image.size,
                Image.Transform.AFFINE,
                coefficients,
                resample=Image.Resampling.BILINEAR,
                fillcolor=0,
            )
            for image in images
        ]
        mask = mask.transform(
            mask.size,
            Image.Transform.AFFINE,
            coefficients,
            resample=Image.Resampling.NEAREST,
            fillcolor=255,
        )
        return images, mask

    @staticmethod
    def _apply_optimization_intensity(image: Image.Image) -> Image.Image:
        array = np.asarray(image, dtype=np.float32)
        mean = float(array.mean())
        contrast = random.uniform(0.95, 1.05)
        gain = random.uniform(0.95, 1.05)
        sigma = random.uniform(2.0, 6.0)
        adjusted = ((array - mean) * contrast + mean) * gain
        adjusted += np.random.normal(0.0, sigma, array.shape).astype(np.float32)
        return Image.fromarray(np.clip(adjusted, 0.0, 255.0).astype(np.uint8), mode="L")


class GridPatchPoreSegmentationDataset(Dataset):
    """Deterministic native-resolution validation tiles."""

    def __init__(
        self,
        samples: list[SamplePair],
        patch_size: int = 384,
        overlap: int = 0,
        image_mean: float = 0.5,
        image_std: float = 0.5,
        feature_mode: str = "grayscale",
        mask_threshold: int = 127,
    ) -> None:
        if patch_size < 1:
            raise ValueError("patch_size must be positive.")
        if overlap < 0 or overlap >= patch_size:
            raise ValueError("overlap must be in the range [0, patch_size).")
        self.samples = samples
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.feature_mode = feature_mode
        self.mask_threshold = mask_threshold
        self.neighbor_image_paths = _build_neighbor_image_paths(samples)
        stride = patch_size - overlap
        self.tiles: list[tuple[int, int, int]] = []
        for sample_index, sample in enumerate(samples):
            with Image.open(sample.image_path) as image:
                width, height = image.size
            xs = self._axis_positions(width, patch_size, stride)
            ys = self._axis_positions(height, patch_size, stride)
            self.tiles.extend((sample_index, x, y) for y in ys for x in xs)

    @staticmethod
    def _axis_positions(length: int, patch_size: int, stride: int) -> list[int]:
        if length <= patch_size:
            return [0]
        positions = list(range(0, length - patch_size + 1, stride))
        final_position = length - patch_size
        if positions[-1] != final_position:
            positions.append(final_position)
        return positions

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_index, x, y = self.tiles[index]
        sample = self.samples[sample_index]
        image = Image.open(sample.image_path).convert("L")
        mask = Image.open(sample.mask_path).convert("L")
        previous_path, next_path = self.neighbor_image_paths[sample.sample_id]
        previous_image = Image.open(previous_path).convert("L")
        next_image = Image.open(next_path).convert("L")
        if mask.size != image.size:
            raise ValueError(
                f"Image/mask dimensions differ for validation sample {sample.sample_id}."
            )
        if previous_image.size != image.size:
            previous_image = previous_image.resize(
                image.size, Image.Resampling.BILINEAR
            )
        if next_image.size != image.size:
            next_image = next_image.resize(image.size, Image.Resampling.BILINEAR)

        box = (x, y, x + self.patch_size, y + self.patch_size)
        images = [
            candidate.crop(box) for candidate in (previous_image, image, next_image)
        ]
        mask = mask.crop(box)
        mask_array = mask_to_array(mask, positive_threshold=self.mask_threshold)
        if self.feature_mode == "stack_relief":
            image_tensor = image_stack_to_feature_tensor(
                images[0],
                images[1],
                images[2],
                image_mean=self.image_mean,
                image_std=self.image_std,
            )
        else:
            image_tensor = image_to_feature_tensor(
                images[1],
                feature_mode=self.feature_mode,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )
        return {
            "image": image_tensor,
            "mask": torch.from_numpy(mask_array).unsqueeze(0),
            "sample_id": sample.sample_id,
        }


class RandomPatchPoreSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[SamplePair],
        patch_size: int = 256,
        patches_per_image: int = 8,
        positive_patch_ratio: float = 0.7,
        min_positive_fraction: float = 0.01,
        augment: bool = True,
        image_mean: float = 0.5,
        image_std: float = 0.5,
        feature_mode: str = "grayscale",
        mask_threshold: int = 127,
        augmentation_mode: str = "safe",
    ) -> None:
        if patches_per_image < 1:
            raise ValueError("patches_per_image must be at least 1.")
        if not 0.0 <= positive_patch_ratio <= 1.0:
            raise ValueError("positive_patch_ratio must be in the range [0.0, 1.0].")
        self.samples = samples
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.positive_patch_ratio = positive_patch_ratio
        self.min_positive_fraction = min_positive_fraction
        self.augment = augment
        self.image_mean = image_mean
        self.image_std = image_std
        self.feature_mode = feature_mode
        self.mask_threshold = mask_threshold
        self.augmentation_mode = augmentation_mode
        self.neighbor_image_paths = _build_neighbor_image_paths(samples)

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index % len(self.samples)]
        image = Image.open(sample.image_path).convert("L")
        mask = Image.open(sample.mask_path).convert("L")
        previous_path, next_path = self.neighbor_image_paths[sample.sample_id]
        previous_image = Image.open(previous_path).convert("L")
        next_image = Image.open(next_path).convert("L")

        images, mask = self._crop_images_and_mask(
            [previous_image, image, next_image], mask
        )

        if self.augment:
            images, mask = PoreSegmentationDataset._apply_augmentations_to_images(
                images,
                mask,
                mode=self.augmentation_mode,
            )

        mask_array = mask_to_array(mask, positive_threshold=self.mask_threshold)
        if self.feature_mode == "stack_relief":
            image_tensor = image_stack_to_feature_tensor(
                images[0],
                images[1],
                images[2],
                image_mean=self.image_mean,
                image_std=self.image_std,
            )
        else:
            image_tensor = image_to_feature_tensor(
                images[1],
                feature_mode=self.feature_mode,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )

        return {
            "image": image_tensor,
            "mask": torch.from_numpy(mask_array).unsqueeze(0),
            "sample_id": sample.sample_id,
        }

    def _crop_pair(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        images, mask = self._crop_images_and_mask([image], mask)
        return images[0], mask

    def _crop_images_and_mask(
        self,
        images: list[Image.Image],
        mask: Image.Image,
    ) -> tuple[list[Image.Image], Image.Image]:
        image = images[0]
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        images = [
            candidate.resize(image.size, Image.Resampling.BILINEAR)
            if candidate.size != image.size
            else candidate
            for candidate in images
        ]

        width, height = image.size
        padded_width = max(width, self.patch_size)
        padded_height = max(height, self.patch_size)
        if (padded_width, padded_height) != image.size:
            padded_images = []
            for image in images:
                padded_image = Image.new("L", (padded_width, padded_height), color=0)
                padded_image.paste(image, (0, 0))
                padded_images.append(padded_image)
            padded_mask = Image.new("L", (padded_width, padded_height), color=255)
            padded_mask.paste(mask, (0, 0))
            images = padded_images
            mask = padded_mask
            width, height = images[0].size

        x, y = self._choose_crop_origin(mask, width, height)
        box = (x, y, x + self.patch_size, y + self.patch_size)
        return [image.crop(box) for image in images], mask.crop(box)

    def _choose_crop_origin(
        self, mask: Image.Image, width: int, height: int
    ) -> tuple[int, int]:
        max_x = width - self.patch_size
        max_y = height - self.patch_size
        want_positive = random.random() < self.positive_patch_ratio
        if not want_positive:
            return random.randint(0, max_x), random.randint(0, max_y)

        mask_array = np.asarray(mask, dtype=np.uint8)
        positive_y, positive_x = np.where(mask_array <= self.mask_threshold)
        if len(positive_x) == 0:
            return random.randint(0, max_x), random.randint(0, max_y)

        min_positive_pixels = int(
            self.patch_size * self.patch_size * self.min_positive_fraction
        )
        best_x = random.randint(0, max_x)
        best_y = random.randint(0, max_y)
        best_positive_pixels = -1
        for _ in range(32):
            point_index = random.randrange(len(positive_x))
            center_x = int(positive_x[point_index])
            center_y = int(positive_y[point_index])
            x = min(max(center_x - random.randint(0, self.patch_size - 1), 0), max_x)
            y = min(max(center_y - random.randint(0, self.patch_size - 1), 0), max_y)
            patch = mask_array[y : y + self.patch_size, x : x + self.patch_size]
            positive_pixels = int((patch <= self.mask_threshold).sum())
            if positive_pixels > best_positive_pixels:
                best_x = x
                best_y = y
                best_positive_pixels = positive_pixels
            if positive_pixels >= min_positive_pixels:
                return x, y

        return best_x, best_y
