from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.stem.lower())
    ]


def collect_mask_paths(mask_dir: Path, suffix: str | None) -> list[Path]:
    paths = [
        path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if suffix:
        paths = [path for path in paths if path.stem.endswith(suffix)]
    paths.sort(key=natural_key)
    if not paths:
        suffix_text = f" with suffix {suffix!r}" if suffix else ""
        raise FileNotFoundError(f"No mask files{suffix_text} found in {mask_dir}.")
    return paths


def load_stack(
    mask_dir: Path, pore_value: str, suffix: str | None
) -> tuple[list[str], np.ndarray]:
    paths = collect_mask_paths(mask_dir, suffix=suffix)
    masks = []
    stems = []
    for path in paths:
        stem = path.stem
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        array = np.asarray(Image.open(path).convert("L"))
        if pore_value == "zero":
            mask = array == 0
        elif pore_value == "nonzero":
            mask = array > 0
        else:
            raise ValueError("pore_value must be 'zero' or 'nonzero'.")
        masks.append(mask)
        stems.append(stem)

    first_shape = masks[0].shape
    for stem, mask in zip(stems, masks, strict=False):
        if mask.shape != first_shape:
            raise ValueError(
                f"All masks in one stack must have the same shape. "
                f"First shape is {first_shape}, but {stem!r} has {mask.shape}."
            )
    return stems, np.stack(masks, axis=0)


def count_vertices(mask: np.ndarray) -> int:
    z, y, x = mask.shape
    vertices = np.zeros((z + 1, y + 1, x + 1), dtype=bool)
    vertices[:-1, :-1, :-1] |= mask
    vertices[1:, :-1, :-1] |= mask
    vertices[:-1, 1:, :-1] |= mask
    vertices[:-1, :-1, 1:] |= mask
    vertices[1:, 1:, :-1] |= mask
    vertices[1:, :-1, 1:] |= mask
    vertices[:-1, 1:, 1:] |= mask
    vertices[1:, 1:, 1:] |= mask
    return int(vertices.sum())


def count_edges(mask: np.ndarray) -> int:
    z, y, x = mask.shape
    total = 0

    edges_z = np.zeros((z, y + 1, x + 1), dtype=bool)
    edges_z[:, :-1, :-1] |= mask
    edges_z[:, 1:, :-1] |= mask
    edges_z[:, :-1, 1:] |= mask
    edges_z[:, 1:, 1:] |= mask
    total += int(edges_z.sum())

    edges_y = np.zeros((z + 1, y, x + 1), dtype=bool)
    edges_y[:-1, :, :-1] |= mask
    edges_y[1:, :, :-1] |= mask
    edges_y[:-1, :, 1:] |= mask
    edges_y[1:, :, 1:] |= mask
    total += int(edges_y.sum())

    edges_x = np.zeros((z + 1, y + 1, x), dtype=bool)
    edges_x[:-1, :-1, :] |= mask
    edges_x[1:, :-1, :] |= mask
    edges_x[:-1, 1:, :] |= mask
    edges_x[1:, 1:, :] |= mask
    total += int(edges_x.sum())

    return total


def count_faces(mask: np.ndarray) -> int:
    z, y, x = mask.shape
    total = 0

    faces_z = np.zeros((z + 1, y, x), dtype=bool)
    faces_z[:-1, :, :] |= mask
    faces_z[1:, :, :] |= mask
    total += int(faces_z.sum())

    faces_y = np.zeros((z, y + 1, x), dtype=bool)
    faces_y[:, :-1, :] |= mask
    faces_y[:, 1:, :] |= mask
    total += int(faces_y.sum())

    faces_x = np.zeros((z, y, x + 1), dtype=bool)
    faces_x[:, :, :-1] |= mask
    faces_x[:, :, 1:] |= mask
    total += int(faces_x.sum())

    return total


def surface_area_6(mask: np.ndarray) -> int:
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    surface = 0
    surface += np.logical_and(center, ~padded[:-2, 1:-1, 1:-1]).sum()
    surface += np.logical_and(center, ~padded[2:, 1:-1, 1:-1]).sum()
    surface += np.logical_and(center, ~padded[1:-1, :-2, 1:-1]).sum()
    surface += np.logical_and(center, ~padded[1:-1, 2:, 1:-1]).sum()
    surface += np.logical_and(center, ~padded[1:-1, 1:-1, :-2]).sum()
    surface += np.logical_and(center, ~padded[1:-1, 1:-1, 2:]).sum()
    return int(surface)


def stack_metrics(mask: np.ndarray) -> dict[str, float]:
    voxel_count = int(mask.size)
    pore_voxels = int(mask.sum())
    vertices = count_vertices(mask)
    edges = count_edges(mask)
    faces = count_faces(mask)
    cubes = pore_voxels
    euler = vertices - edges + faces - cubes
    surface = surface_area_6(mask)
    slice_porosity = mask.mean(axis=(1, 2))
    return {
        "num_slices": float(mask.shape[0]),
        "height": float(mask.shape[1]),
        "width": float(mask.shape[2]),
        "voxel_count": float(voxel_count),
        "pore_voxels": float(pore_voxels),
        "porosity": pore_voxels / voxel_count if voxel_count else 0.0,
        "surface_area": float(surface),
        "specific_surface": surface / voxel_count if voxel_count else 0.0,
        "surface_per_pore_voxel": surface / pore_voxels if pore_voxels else 0.0,
        "euler_characteristic": float(euler),
        "euler_density_per_million_voxels": (euler / voxel_count * 1_000_000.0)
        if voxel_count
        else 0.0,
        "vertices": float(vertices),
        "edges": float(edges),
        "faces": float(faces),
        "cubes": float(cubes),
        "slice_porosity_mean": float(slice_porosity.mean()),
        "slice_porosity_std": float(slice_porosity.std()),
        "slice_porosity_min": float(slice_porosity.min()),
        "slice_porosity_max": float(slice_porosity.max()),
    }


def compare_metrics(
    reference: dict[str, float], candidate: dict[str, float]
) -> dict[str, float]:
    comparison = {}
    for key, reference_value in reference.items():
        candidate_value = candidate[key]
        comparison[f"reference_{key}"] = reference_value
        comparison[f"candidate_{key}"] = candidate_value
        comparison[f"delta_{key}"] = candidate_value - reference_value
        comparison[f"relative_delta_{key}"] = (
            (candidate_value - reference_value) / reference_value
            if abs(reference_value) > 1e-12
            else 0.0
        )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two segmentation stacks as whole 3D volumes."
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-pore-value", choices=("zero", "nonzero"), default="zero"
    )
    parser.add_argument(
        "--candidate-pore-value", choices=("zero", "nonzero"), default="nonzero"
    )
    parser.add_argument("--reference-suffix", type=str, default=None)
    parser.add_argument("--candidate-suffix", type=str, default="_pred")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/stack_metric_comparison.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_stems, reference = load_stack(
        args.reference_dir,
        pore_value=args.reference_pore_value,
        suffix=args.reference_suffix,
    )
    candidate_stems, candidate = load_stack(
        args.candidate_dir,
        pore_value=args.candidate_pore_value,
        suffix=args.candidate_suffix,
    )

    reference_metrics = stack_metrics(reference)
    candidate_metrics = stack_metrics(candidate)
    comparison = compare_metrics(reference_metrics, candidate_metrics)
    summary = {
        "reference_dir": str(args.reference_dir),
        "candidate_dir": str(args.candidate_dir),
        "reference_num_files": len(reference_stems),
        "candidate_num_files": len(candidate_stems),
        "reference_stems_preview": reference_stems[:5],
        "candidate_stems_preview": candidate_stems[:5],
        "reference": reference_metrics,
        "candidate": candidate_metrics,
        "comparison": comparison,
        "notes": {
            "porosity": "Pore voxels / all voxels for the whole stack.",
            "euler_characteristic": "3D cubical Euler characteristic V - E + F - C for the pore voxel complex.",
            "euler_density_per_million_voxels": "Euler characteristic normalized by stack volume.",
            "specific_surface": "6-neighbor pore/solid boundary faces divided by all voxels.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"reference files={len(reference_stems)} shape={tuple(reference.shape)}")
    print(f"candidate files={len(candidate_stems)} shape={tuple(candidate.shape)}")
    print(
        "porosity reference={:.4f} candidate={:.4f} delta={:+.4f}".format(
            reference_metrics["porosity"],
            candidate_metrics["porosity"],
            comparison["delta_porosity"],
        )
    )
    print(
        "euler_density reference={:.2f} candidate={:.2f} delta={:+.2f}".format(
            reference_metrics["euler_density_per_million_voxels"],
            candidate_metrics["euler_density_per_million_voxels"],
            comparison["delta_euler_density_per_million_voxels"],
        )
    )
    print(
        "specific_surface reference={:.4f} candidate={:.4f} delta={:+.4f}".format(
            reference_metrics["specific_surface"],
            candidate_metrics["specific_surface"],
            comparison["delta_specific_surface"],
        )
    )
    print(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
