from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT = ROOT / "experiments" / "manuscript_final"
DEFAULT_SHAPE = ROOT / "experiments" / "article_boundary_shape_analysis"
DEFAULT_OUTPUT = ROOT / "docs" / "figures"

MANUAL = "#3f3f3f"
SCRATCH = "#d69b3d"
CANONICAL = "#668da8"
OPTIMIZED = "#238b82"
SAM = "#806da8"
COLORS = {
    "Manual": MANUAL,
    "scratch": SCRATCH,
    "pretrained": CANONICAL,
    "optimized": OPTIMIZED,
    "sam2.1": SAM,
    "Optimized FIB-NET": OPTIMIZED,
    "SAM 2.1": SAM,
}
LABELS = {
    "scratch": "Scratch",
    "pretrained": "Canonical",
    "optimized": "Optimized FIB-NET",
    "sam2.1": "SAM 2.1",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "axes.linewidth": 1.4,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 3.0,
            "lines.markersize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_target_curves(manuscript: Path):
    paths = (
        manuscript / "results" / "blind_correlation_curves.csv",
        manuscript
        / "fibnet_optimized_final"
        / "results"
        / "historical_blind_cf_curves.csv",
        manuscript / "sam2_benchmark" / "results" / "sam2_blind_correlation_curves.csv",
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        for row in read_csv(path):
            model = row.get("method") or row.get("model")
            grouped[(str(model), row["slice"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["lag_pixels"]))
    return grouped


def shape_errors(grouped) -> dict[str, np.ndarray]:
    output = {}
    for model in ("scratch", "pretrained", "optimized", "sam2.1"):
        values = []
        for (row_model, _), rows in grouped.items():
            if row_model != model:
                continue
            component_errors = []
            for correlation in ("Fss", "Fsv"):
                manual = np.asarray(
                    [float(row[f"{correlation}_manual"]) for row in rows]
                )
                predicted = np.asarray(
                    [float(row[f"{correlation}_predicted"]) for row in rows]
                )
                manual /= manual[0]
                predicted /= predicted[0]
                component_errors.append(
                    float(np.sqrt(np.mean((predicted - manual) ** 2)))
                )
            values.append(float(np.mean(component_errors)))
        output[model] = np.asarray(values)
    return output


def mean_curve(grouped, model: str, field: str, normalized: bool):
    manual_curves = []
    predicted_curves = []
    lags = None
    for (row_model, _), rows in grouped.items():
        if row_model != model:
            continue
        lags = np.asarray([float(row["lag_um"]) for row in rows])
        manual = np.asarray([float(row[f"{field}_manual"]) for row in rows])
        predicted = np.asarray([float(row[f"{field}_predicted"]) for row in rows])
        if normalized:
            manual = manual / manual[0]
            predicted = predicted / predicted[0]
        manual_curves.append(manual)
        predicted_curves.append(predicted)
    if lags is None:
        raise ValueError(f"No curves found for {model}.")
    return lags, np.asarray(manual_curves), np.asarray(predicted_curves)


def draw_curve(ax, x, values, label, color):
    mean = np.mean(values, axis=0)
    sd = np.std(values, axis=0, ddof=1)
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.14, linewidth=0)


def amplitude_figure(
    manuscript: Path, output: Path, correlation: str, filename: str
) -> None:
    grouped = load_target_curves(manuscript)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    symbol = correlation.lower()

    for normalized, ax, title in (
        (False, axes[0], f"Absolute $F_{{{symbol}}}$ amplitude"),
        (True, axes[1], f"Zero-lag-normalized $F_{{{symbol}}}$"),
    ):
        x, manual, optimized = mean_curve(grouped, "optimized", correlation, normalized)
        _, _, sam = mean_curve(grouped, "sam2.1", correlation, normalized)
        draw_curve(ax, x, manual, "Manual", MANUAL)
        draw_curve(ax, x, optimized, "Optimized FIB-NET", OPTIMIZED)
        draw_curve(ax, x, sam, "SAM 2.1", SAM)
        ax.set_xlabel(r"$r\;(\mu\mathrm{m})$")
        ax.set_ylabel(
            f"$F_{{{symbol}}}(r)$"
            if not normalized
            else f"$F_{{{symbol}}}(r)/F_{{{symbol}}}(0)$"
        )
        ax.set_title(title, pad=10)
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, loc="upper right")

    errors = shape_errors(grouped)
    order = ("scratch", "pretrained", "optimized", "sam2.1")
    means = [float(np.mean(errors[model])) for model in order]
    standard_errors = [
        float(np.std(errors[model], ddof=1) / np.sqrt(len(errors[model])))
        for model in order
    ]
    x_positions = np.arange(len(order))
    axes[2].bar(
        x_positions,
        means,
        yerr=standard_errors,
        capsize=5,
        color=[COLORS[model] for model in order],
        edgecolor="white",
        linewidth=1.2,
    )
    axes[2].set_xticks(x_positions, [LABELS[model] for model in order], rotation=18)
    axes[2].set_ylabel("Shape-only error, $E_{shape}$")
    axes[2].set_title("Shape discrepancy after normalization", pad=10)
    axes[2].grid(axis="y", alpha=0.18)
    for position, value in zip(x_positions, means, strict=True):
        axes[2].text(
            position,
            value * 0.56,
            f"{value:.3f}",
            ha="center",
            va="center",
            color="white",
            fontsize=13,
            fontweight="bold",
        )

    for label, ax in zip(("(a)", "(b)", "(c)"), axes, strict=True):
        ax.text(
            -0.13, 1.06, label, transform=ax.transAxes, fontsize=19, fontweight="bold"
        )

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / filename, dpi=300)
    plt.close(fig)


def boundary_values(rows: list[dict[str, str]], method: str):
    distances = (1, 2, 3, 5, 10)
    lookup = {(row["method"], row["metric"]): row for row in rows}
    means = []
    standard_deviations = []
    for distance in distances:
        row = lookup[(method, f"all_within_{distance}px_fraction")]
        means.append(100 * float(row["mean"]))
        standard_deviations.append(100 * float(row["std"]))
    return np.asarray(distances), np.asarray(means), np.asarray(standard_deviations)


def shape_boundary_figure(shape_dir: Path, output: Path) -> None:
    shape_rows = read_csv(shape_dir / "shape_factor_per_slice.csv")
    boundary_rows = read_csv(shape_dir / "boundary_aggregated.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)

    segmentations = ("Manual", "Optimized FIB-NET", "SAM 2.1")
    values = [
        np.asarray(
            [
                float(row["median_F"])
                for row in shape_rows
                if row["segmentation"] == segmentation
            ]
        )
        for segmentation in segmentations
    ]
    boxes = axes[0].boxplot(
        values,
        positions=np.arange(3),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2.3},
        whiskerprops={"linewidth": 1.7},
        capprops={"linewidth": 1.7},
    )
    for box, segmentation in zip(boxes["boxes"], segmentations, strict=True):
        box.set_facecolor(COLORS[segmentation])
        box.set_alpha(0.9)
    point_offsets = np.linspace(-0.13, 0.13, 5)
    for position, (segmentation, observations) in enumerate(
        zip(segmentations, values, strict=True)
    ):
        axes[0].scatter(
            position + point_offsets,
            observations,
            s=42,
            color=COLORS[segmentation],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    axes[0].set_xticks(np.arange(3), segmentations, rotation=12)
    axes[0].set_ylabel("Slice-level median shape factor, $F$")
    axes[0].set_title("Pore-component geometry", pad=10)
    axes[0].grid(axis="y", alpha=0.18)

    for method in ("Optimized FIB-NET", "SAM 2.1"):
        distance, mean, sd = boundary_values(boundary_rows, method)
        axes[1].errorbar(
            distance,
            mean,
            yerr=sd,
            marker="o" if method == "Optimized FIB-NET" else "s",
            capsize=5,
            color=COLORS[method],
            label=method,
        )
    axes[1].set_xticks((1, 2, 3, 5, 10))
    axes[1].set_xlabel("Distance to manual boundary (px)")
    axes[1].set_ylabel("Disagreement within distance (%)")
    axes[1].set_ylim(0, 85)
    axes[1].set_title("Boundary-related disagreement", pad=10)
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False, loc="lower right")

    for label, ax in zip(("(a)", "(b)"), axes, strict=True):
        ax.text(
            -0.12, 1.06, label, transform=ax.transAxes, fontsize=19, fontweight="bold"
        )

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "figure_shape_boundary.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render explanatory manuscript PNGs.")
    parser.add_argument("--manuscript-dir", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--shape-dir", type=Path, default=DEFAULT_SHAPE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    amplitude_figure(
        args.manuscript_dir,
        args.output_dir,
        correlation="Fsv",
        filename="figure_amplitude_effect.png",
    )
    amplitude_figure(
        args.manuscript_dir,
        args.output_dir,
        correlation="Fss",
        filename="figure_amplitude_effect_fss.png",
    )
    shape_boundary_figure(args.shape_dir, args.output_dir)
    print(f"Saved publication PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
