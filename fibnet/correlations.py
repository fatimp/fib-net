from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
AXES = {"z": 0, "y": 1, "x": 2}


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
    stems = []
    masks = []
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
        stems.append(stem)
        masks.append(mask)
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("All masks in one stack must have the same image shape.")
    return stems, np.stack(masks, axis=0)


def surface_voxels(pore: np.ndarray) -> np.ndarray:
    padded = np.pad(pore, 1, constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    has_solid_neighbor = (
        ~padded[:-2, 1:-1, 1:-1]
        | ~padded[2:, 1:-1, 1:-1]
        | ~padded[1:-1, :-2, 1:-1]
        | ~padded[1:-1, 2:, 1:-1]
        | ~padded[1:-1, 1:-1, :-2]
        | ~padded[1:-1, 1:-1, 2:]
    )
    return center & has_solid_neighbor


def shifted_pair(
    a: np.ndarray, b: np.ndarray, axis: int, distance: int
) -> tuple[np.ndarray, np.ndarray]:
    if distance == 0:
        return a, b
    left = [slice(None), slice(None), slice(None)]
    right = [slice(None), slice(None), slice(None)]
    left[axis] = slice(None, -distance)
    right[axis] = slice(distance, None)
    return a[tuple(left)], b[tuple(right)]


def directional_correlations(
    pore: np.ndarray,
    max_distance: int,
    step: int,
    axes: list[str],
) -> list[dict[str, float | str]]:
    surface = surface_voxels(pore)
    rows: list[dict[str, float | str]] = []
    for axis_name in axes:
        axis = AXES[axis_name]
        max_axis_distance = min(max_distance, pore.shape[axis] - 1)
        for distance in range(0, max_axis_distance + 1, step):
            surface_a, surface_b = shifted_pair(surface, surface, axis, distance)
            surface_for_void, pore_b = shifted_pair(surface, pore, axis, distance)
            sample_count = surface_a.size
            rows.append(
                {
                    "axis": axis_name,
                    "distance": float(distance),
                    "sample_count": float(sample_count),
                    "fss": float(
                        np.logical_and(surface_a, surface_b).sum() / sample_count
                    ),
                    "fsv": float(
                        np.logical_and(surface_for_void, pore_b).sum() / sample_count
                    ),
                    "surface_fraction_a": float(surface_a.mean()),
                    "surface_fraction_b": float(surface_b.mean()),
                    "pore_fraction_b": float(pore_b.mean()),
                }
            )
    return rows


def radial_average(rows: list[dict[str, float | str]]) -> list[dict[str, float]]:
    by_distance: dict[float, list[dict[str, float | str]]] = {}
    for row in rows:
        by_distance.setdefault(float(row["distance"]), []).append(row)
    averaged = []
    for distance in sorted(by_distance):
        group = by_distance[distance]
        weights = np.asarray(
            [float(row["sample_count"]) for row in group], dtype=np.float64
        )
        averaged.append(
            {
                "distance": distance,
                "sample_count": float(weights.sum()),
                "fss": float(
                    np.average([float(row["fss"]) for row in group], weights=weights)
                ),
                "fsv": float(
                    np.average([float(row["fsv"]) for row in group], weights=weights)
                ),
                "surface_fraction_a": float(
                    np.average(
                        [float(row["surface_fraction_a"]) for row in group],
                        weights=weights,
                    )
                ),
                "surface_fraction_b": float(
                    np.average(
                        [float(row["surface_fraction_b"]) for row in group],
                        weights=weights,
                    )
                ),
                "pore_fraction_b": float(
                    np.average(
                        [float(row["pore_fraction_b"]) for row in group],
                        weights=weights,
                    )
                ),
            }
        )
    return averaged


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute directional Fss/Fsv-like surface correlations for a 3D mask stack."
    )
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--pore-value", choices=("zero", "nonzero"), default="nonzero")
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--max-distance", type=int, default=64)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument(
        "--axes", nargs="+", choices=("z", "y", "x"), default=["z", "y", "x"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stems, pore = load_stack(
        args.mask_dir, pore_value=args.pore_value, suffix=args.suffix
    )
    rows = directional_correlations(
        pore,
        max_distance=args.max_distance,
        step=args.step,
        axes=args.axes,
    )
    radial = radial_average(rows)
    payload = {
        "mask_dir": str(args.mask_dir),
        "num_slices": len(stems),
        "shape": list(pore.shape),
        "pore_fraction": float(pore.mean()),
        "max_distance": args.max_distance,
        "step": args.step,
        "axes": args.axes,
        "directional": rows,
        "radial_average": radial,
        "notes": {
            "fss": "Directional digital proxy for surface-surface correlation: P(surface at x and surface at x+r).",
            "fsv": "Directional digital proxy for surface-void correlation: P(surface at x and pore/void at x+r).",
            "surface": "Pore voxels touching solid/outside in the 6-neighborhood.",
            "boundary": "Non-periodic; only overlapping shifted regions are used.",
        },
    }
    json_path = args.output_prefix.with_suffix(".json")
    directional_csv_path = args.output_prefix.with_name(
        f"{args.output_prefix.name}_directional.csv"
    )
    radial_csv_path = args.output_prefix.with_name(
        f"{args.output_prefix.name}_radial.csv"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(directional_csv_path, rows)
    write_csv(radial_csv_path, radial)
    print(f"shape={tuple(pore.shape)} pore_fraction={pore.mean():.4f}")
    print(f"r0 fss={radial[0]['fss']:.6f} fsv={radial[0]['fsv']:.6f}")
    print(f"Saved JSON to {json_path}")
    print(f"Saved directional CSV to {directional_csv_path}")
    print(f"Saved radial CSV to {radial_csv_path}")


if __name__ == "__main__":
    main()
