from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from fibnet import correlations


def julia_executable() -> str:
    executable = os.environ.get("JULIA_EXE") or shutil.which("julia")
    if executable is None:
        pytest.skip("Julia is not installed or JULIA_EXE is not set.")
    return executable


def test_directional_correlations_uses_julia_bridge(monkeypatch) -> None:
    pore = np.zeros((2, 3, 4), dtype=bool)
    pore[1, 2, 3] = True
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        input_path = Path(command[4])
        observed["raw"] = input_path.read_bytes()
        output_path = Path(command[-1])
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("axis", "distance", "sample_count", "fss", "fsv"))
            writer.writerow(("z", 0, 24, 1.25, -0.125))

    monkeypatch.setattr(correlations.subprocess, "run", fake_run)
    rows = correlations.directional_correlations(
        pore,
        max_distance=8,
        step=2,
        axes=["z"],
        boundary="periodic",
        filter_width=5,
        julia_executable="julia-custom",
    )

    command = observed["command"]
    assert command[0] == "julia-custom"
    assert command[1:3] == ["--startup-file=no", f"--project={correlations.JULIA_DIR}"]
    assert command[5:8] == ["2", "3", "4"]
    assert command[8:13] == ["8", "2", "z", "periodic", "5"]
    expected = np.logical_not(pore).astype(np.uint8).tobytes(order="F")
    assert observed["raw"] == expected
    assert observed["kwargs"] == {
        "check": True,
        "capture_output": True,
        "text": True,
    }
    assert rows == [
        {
            "axis": "z",
            "distance": 0.0,
            "sample_count": 24.0,
            "fss": 1.25,
            "fsv": -0.125,
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_distance": -1}, "non-negative"),
        ({"step": 0}, "positive"),
        ({"axes": []}, "one or more"),
        ({"axes": ["x", "x"]}, "duplicates"),
        ({"boundary": "closed"}, "nonperiodic"),
        ({"filter_width": 3}, "5 or 7"),
    ],
)
def test_directional_correlations_validates_arguments(kwargs, message) -> None:
    options = {"max_distance": 2, "step": 1, "axes": ["x"]}
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        correlations.directional_correlations(np.zeros((2, 2, 2)), **options)


def test_radial_average_uses_available_sample_counts() -> None:
    rows = [
        {
            "axis": "x",
            "distance": 1.0,
            "sample_count": 3.0,
            "fss": 2.0,
            "fsv": 4.0,
        },
        {
            "axis": "y",
            "distance": 1.0,
            "sample_count": 1.0,
            "fss": 6.0,
            "fsv": 8.0,
        },
    ]
    assert correlations.radial_average(rows) == [
        {"distance": 1.0, "sample_count": 4.0, "fss": 3.0, "fsv": 5.0}
    ]


def test_native_2d_julia_interface_and_nonperiodic_boundary() -> None:
    pore = np.zeros((21, 21), dtype=bool)
    pore[:, :10] = True

    rows = correlations.directional_correlations(
        pore,
        max_distance=4,
        step=2,
        axes=["y", "x"],
        boundary="nonperiodic",
        filter_width=7,
        julia_executable=julia_executable(),
    )

    assert [row["sample_count"] for row in rows if row["axis"] == "x"] == [
        441.0,
        399.0,
        357.0,
    ]
    averaged = correlations.radial_average(rows)
    assert averaged[0]["fss"] == pytest.approx(0.010198842609576705)
    assert averaged[0]["fsv"] == pytest.approx(0.023809523809522153)
    assert averaged[2]["fss"] == pytest.approx(0.005686475921242795)
    assert averaged[2]["fsv"] == pytest.approx(0.011904761904761082)


@pytest.mark.parametrize(
    ("path", "group", "expected"),
    [
        (
            "results/scenario1_summary.csv",
            "soil1",
            (0.15014060557741268, 0.6990865163979532, 0.42461356098768305),
        ),
        (
            "results/scenario1_summary.csv",
            "soil9",
            (0.09597282141545356, 0.2937133207645575, 0.1948430710900055),
        ),
        (
            "results/blind_summary.csv",
            "scratch",
            (0.0741981475823761, 0.1919681640376743, 0.13308315581002517),
        ),
        (
            "results/blind_summary.csv",
            "pretrained",
            (0.07893345957800411, 0.2468392095659957, 0.1628863345719999),
        ),
        (
            "fibnet_optimized_final/results/historical_blind_summary.csv",
            "optimized",
            (0.08189843724127246, 0.21284706824730004, 0.14737275274428624),
        ),
        (
            "sam2_benchmark/results/sam2_blind_summary.csv",
            "sam2.1",
            (0.0908932131657775, 0.33445929302423894, 0.21267625309500823),
        ),
    ],
)
def test_authoritative_manuscript_cf_triplets(path, group, expected) -> None:
    manuscript = Path(__file__).parents[1] / "experiments" / "manuscript_final"
    rows = list(csv.DictReader((manuscript / path).open(encoding="utf-8", newline="")))
    if "metric" in rows[0]:
        group_column = "sequence" if "sequence" in rows[0] else "model"
        actual = tuple(
            float(
                next(
                    row
                    for row in rows
                    if row[group_column] == group and row["metric"] == metric
                )["mean"]
            )
            for metric in ("fss_nrmse", "fsv_nrmse", "e_cf")
        )
    else:
        row = next(row for row in rows if row["model"] == group)
        actual = tuple(
            float(row[f"{metric}_mean"])
            for metric in ("fss_nrmse", "fsv_nrmse", "e_cf")
        )
    assert actual == pytest.approx(expected, abs=1e-15)


def test_authoritative_loo_triplet() -> None:
    path = (
        Path(__file__).parents[1]
        / "experiments/manuscript_final/fibnet_optimization/ablation_pass2/results/pareto_candidates.csv"
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    row = next(row for row in rows if row["candidate_id"] == "G_t0.60")
    assert tuple(
        float(row[key]) for key in ("fss_nrmse", "fsv_nrmse", "e_cf")
    ) == pytest.approx(
        (0.06935461082821912, 0.19797769630801615, 0.13366615356811765),
        abs=1e-15,
    )
