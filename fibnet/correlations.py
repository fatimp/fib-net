from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
AXES_2D = {"y": 0, "x": 1}
AXES_3D = {"z": 0, "y": 1, "x": 2}
JULIA_DIR = Path(__file__).with_name("julia")
JULIA_SCRIPT_2D = JULIA_DIR / "surface_correlations_2d.jl"
JULIA_SCRIPT_3D = JULIA_DIR / "surface_correlations.jl"


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


def _run_julia(
    pore: np.ndarray,
    max_distance: int,
    step: int,
    axes: list[str],
    boundary: str,
    filter_width: int,
    julia_executable: str,
) -> list[dict[str, float | str]]:
    # CorrelationFunctions.jl assumes that void is zero. The bridge passes
    # pore=False/solid=True as UInt8 0/1 and evaluates the solid phase (1).
    solid = np.logical_not(pore).astype(np.uint8)
    with tempfile.TemporaryDirectory(prefix="fibnet-correlations-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "stack.raw"
        output_path = temp_path / "correlations.tsv"
        input_path.write_bytes(solid.tobytes(order="F"))
        julia_script = JULIA_SCRIPT_2D if pore.ndim == 2 else JULIA_SCRIPT_3D
        command = [
            julia_executable,
            "--startup-file=no",
            f"--project={JULIA_DIR}",
            str(julia_script),
            str(input_path),
            *(str(size) for size in pore.shape),
            str(max_distance),
            str(step),
            ",".join(axes),
            boundary,
            str(filter_width),
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except OSError as error:
            raise RuntimeError(
                f"Could not start Julia executable {julia_executable!r}. Install Julia "
                "1.10 or newer and instantiate fibnet/julia/Project.toml."
            ) from error
        except subprocess.CalledProcessError as error:
            details = error.stderr.strip() or error.stdout.strip() or str(error)
            raise RuntimeError(
                "CorrelationFunctions.jl failed. Run `julia --project=fibnet/julia "
                "-e 'using Pkg; Pkg.instantiate()'` once before computing "
                f"correlations. Julia reported:\n{details}"
            ) from error

        with output_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return [
                {
                    "axis": row["axis"],
                    "distance": float(row["distance"]),
                    "sample_count": float(row["sample_count"]),
                    "fss": float(row["fss"]),
                    "fsv": float(row["fsv"]),
                }
                for row in reader
            ]


def directional_correlations(
    pore: np.ndarray,
    max_distance: int,
    step: int,
    axes: list[str],
    *,
    boundary: str = "nonperiodic",
    filter_width: int = 7,
    julia_executable: str = "julia",
) -> list[dict[str, float | str]]:
    pore = np.asarray(pore, dtype=bool)
    if pore.ndim not in {2, 3}:
        raise ValueError("pore must be a two- or three-dimensional array.")
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative.")
    if step <= 0:
        raise ValueError("step must be positive.")
    valid_axes = AXES_2D if pore.ndim == 2 else AXES_3D
    if not axes or any(axis not in valid_axes for axis in axes):
        choices = ", ".join(valid_axes)
        raise ValueError(f"axes must contain one or more of: {choices}.")
    if len(set(axes)) != len(axes):
        raise ValueError("axes must not contain duplicates.")
    if boundary not in {"nonperiodic", "periodic"}:
        raise ValueError("boundary must be 'nonperiodic' or 'periodic'.")
    if filter_width not in {5, 7}:
        raise ValueError("filter_width must be 5 or 7.")
    return _run_julia(
        pore,
        max_distance,
        step,
        axes,
        boundary,
        filter_width,
        julia_executable,
    )


def correlation_curves_2d(
    pore: np.ndarray,
    *,
    max_distance: int = 64,
    step: int = 2,
    julia_executable: str = "julia",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid-pair-weighted x/y Fss and Fsv curves for one 2D mask."""
    rows = directional_correlations(
        pore,
        max_distance=max_distance,
        step=step,
        axes=["y", "x"],
        boundary="nonperiodic",
        filter_width=7,
        julia_executable=julia_executable,
    )
    averaged = radial_average(rows)
    return (
        np.asarray([row["distance"] for row in averaged], dtype=np.int32),
        np.asarray([row["fss"] for row in averaged], dtype=np.float64),
        np.asarray([row["fsv"] for row in averaged], dtype=np.float64),
    )


def normalized_rmse(manual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute manuscript NRMSE: RMSE divided by the manual curve range."""
    manual = np.asarray(manual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if manual.shape != predicted.shape:
        raise ValueError("Manual and predicted curves must have the same shape.")
    scale = float(np.ptp(manual))
    if scale <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.sqrt(np.mean((predicted - manual) ** 2)) / scale)


def _compare_correlation_curves(
    lags: np.ndarray,
    manual_fss: np.ndarray,
    manual_fsv: np.ndarray,
    predicted_fss: np.ndarray,
    predicted_fsv: np.ndarray,
    pixel_size: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    fss_nrmse = normalized_rmse(manual_fss, predicted_fss)
    fsv_nrmse = normalized_rmse(manual_fsv, predicted_fsv)
    metrics = {
        "fss_rmse": float(np.sqrt(np.mean((predicted_fss - manual_fss) ** 2))),
        "fss_nrmse": fss_nrmse,
        "fsv_rmse": float(np.sqrt(np.mean((predicted_fsv - manual_fsv) ** 2))),
        "fsv_nrmse": fsv_nrmse,
        "e_cf": (fss_nrmse + fsv_nrmse) / 2.0,
    }
    curves = [
        {
            "lag_pixels": int(lag),
            "lag_um": float(lag * pixel_size),
            "Fss_manual": float(manual_fss[index]),
            "Fss_predicted": float(predicted_fss[index]),
            "Fsv_manual": float(manual_fsv[index]),
            "Fsv_predicted": float(predicted_fsv[index]),
        }
        for index, lag in enumerate(lags)
    ]
    return metrics, curves


def evaluate_correlation_pair_2d(
    manual_pore: np.ndarray,
    predicted_pore: np.ndarray,
    pixel_size: float,
    *,
    max_distance: int = 64,
    step: int = 2,
    julia_executable: str = "julia",
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Evaluate one manuscript mask pair with native 2D Julia correlations."""
    manual_pore = np.asarray(manual_pore, dtype=bool)
    predicted_pore = np.asarray(predicted_pore, dtype=bool)
    if manual_pore.ndim != 2 or predicted_pore.ndim != 2:
        raise ValueError("Manuscript correlation masks must be two-dimensional.")
    if manual_pore.shape != predicted_pore.shape:
        raise ValueError("Manual and predicted masks must have the same shape.")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive.")

    lags, manual_fss, manual_fsv = correlation_curves_2d(
        manual_pore,
        max_distance=max_distance,
        step=step,
        julia_executable=julia_executable,
    )
    predicted_lags, predicted_fss, predicted_fsv = correlation_curves_2d(
        predicted_pore,
        max_distance=max_distance,
        step=step,
        julia_executable=julia_executable,
    )
    if not np.array_equal(lags, predicted_lags):
        raise ValueError("Manual and predicted Julia lag grids differ.")
    return _compare_correlation_curves(
        lags,
        manual_fss,
        manual_fsv,
        predicted_fss,
        predicted_fsv,
        pixel_size,
    )


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
        description="Compute native 2D or 3D Fss/Fsv with CorrelationFunctions.jl."
    )
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--pore-value", choices=("zero", "nonzero"), default="nonzero")
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--manual-dir", type=Path)
    parser.add_argument(
        "--manual-pore-value", choices=("zero", "nonzero"), default="zero"
    )
    parser.add_argument("--manual-suffix", type=str, default=None)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--mode", choices=("slices2d", "volume3d"), default="slices2d")
    parser.add_argument("--max-distance", type=int, default=64)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--axes", nargs="+", choices=("z", "y", "x"), default=None)
    parser.add_argument(
        "--boundary",
        choices=("nonperiodic", "periodic"),
        default="nonperiodic",
    )
    parser.add_argument("--filter-width", type=int, choices=(5, 7), default=7)
    parser.add_argument(
        "--julia", default="julia", help="Path to the Julia executable."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stems, pore = load_stack(
        args.mask_dir, pore_value=args.pore_value, suffix=args.suffix
    )
    axes = args.axes or (["y", "x"] if args.mode == "slices2d" else ["z", "y", "x"])
    per_slice = []
    all_directional = []
    all_radial = []
    metrics_rows = []
    curve_rows = []
    predicted_curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if args.mode == "slices2d":
        for stem, mask in zip(stems, pore, strict=True):
            rows = directional_correlations(
                mask,
                max_distance=args.max_distance,
                step=args.step,
                axes=axes,
                boundary=args.boundary,
                filter_width=args.filter_width,
                julia_executable=args.julia,
            )
            radial = radial_average(rows)
            all_directional.extend({"slice": stem, **row} for row in rows)
            all_radial.extend({"slice": stem, **row} for row in radial)
            per_slice.append({"slice": stem, "radial_average": radial})
            predicted_curves[stem] = (
                np.asarray([row["distance"] for row in radial], dtype=np.int32),
                np.asarray([row["fss"] for row in radial], dtype=np.float64),
                np.asarray([row["fsv"] for row in radial], dtype=np.float64),
            )

        if args.manual_dir is not None:
            if (
                axes != ["y", "x"]
                or args.boundary != "nonperiodic"
                or args.filter_width != 7
            ):
                raise ValueError(
                    "Paired manuscript evaluation requires axes y x, nonperiodic "
                    "boundaries, and ConvKernel(7)."
                )
            manual_stems, manual = load_stack(
                args.manual_dir,
                pore_value=args.manual_pore_value,
                suffix=args.manual_suffix,
            )
            if manual_stems != stems:
                raise ValueError("Manual and predicted mask stems do not match.")
            for stem, manual_mask in zip(stems, manual, strict=True):
                manual_lags, manual_fss, manual_fsv = correlation_curves_2d(
                    manual_mask,
                    max_distance=args.max_distance,
                    step=args.step,
                    julia_executable=args.julia,
                )
                predicted_lags, predicted_fss, predicted_fsv = predicted_curves[stem]
                if not np.array_equal(manual_lags, predicted_lags):
                    raise ValueError("Manual and predicted Julia lag grids differ.")
                metrics, curves = _compare_correlation_curves(
                    manual_lags,
                    manual_fss,
                    manual_fsv,
                    predicted_fss,
                    predicted_fsv,
                    args.pixel_size,
                )
                metrics_rows.append({"slice": stem, **metrics})
                curve_rows.extend({"slice": stem, **row} for row in curves)
        rows = all_directional
        radial = all_radial
    else:
        if args.manual_dir is not None:
            raise ValueError("--manual-dir is supported only with --mode slices2d.")
        rows = directional_correlations(
            pore,
            max_distance=args.max_distance,
            step=args.step,
            axes=axes,
            boundary=args.boundary,
            filter_width=args.filter_width,
            julia_executable=args.julia,
        )
        radial = radial_average(rows)
    payload = {
        "mask_dir": str(args.mask_dir),
        "num_slices": len(stems),
        "shape": list(pore.shape),
        "pore_fraction": float(pore.mean()),
        "max_distance": args.max_distance,
        "step": args.step,
        "mode": args.mode,
        "axes": axes,
        "directional": rows,
        "radial_average": radial,
        "per_slice": per_slice,
        "per_slice_metrics": metrics_rows,
        "metric_means": {
            key: float(np.mean([float(row[key]) for row in metrics_rows]))
            for key in ("fss_rmse", "fss_nrmse", "fsv_rmse", "fsv_nrmse", "e_cf")
        }
        if metrics_rows
        else {},
        "comparison_curves": curve_rows,
        "method": {
            "package": "CorrelationFunctions.jl",
            "functions": {"fss": "Directional.surf2", "fsv": "Directional.surfvoid"},
            "version": "0.14.0",
            "surface_filter": f"ConvKernel({args.filter_width}); convolution-based interfacial field",
            "boundary": args.boundary,
            "phase_encoding": "pore/void=0, solid=1; solid/void interface",
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
    if radial:
        print(f"r0 fss={radial[0]['fss']:.6f} fsv={radial[0]['fsv']:.6f}")
    if metrics_rows:
        metrics_csv_path = args.output_prefix.with_name(
            f"{args.output_prefix.name}_metrics.csv"
        )
        curves_csv_path = args.output_prefix.with_name(
            f"{args.output_prefix.name}_curves.csv"
        )
        write_csv(metrics_csv_path, metrics_rows)
        write_csv(curves_csv_path, curve_rows)
        print(f"Saved metrics CSV to {metrics_csv_path}")
        print(f"Saved curves CSV to {curves_csv_path}")
    print(f"Saved JSON to {json_path}")
    print(f"Saved directional CSV to {directional_csv_path}")
    print(f"Saved radial CSV to {radial_csv_path}")


if __name__ == "__main__":
    main()
