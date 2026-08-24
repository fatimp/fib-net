from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path, key: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload[key]


def index_by_distance(rows: list[dict]) -> dict[float, dict]:
    return {float(row["distance"]): row for row in rows}


def compare_rows(
    reference_rows: list[dict], candidate_rows: list[dict]
) -> tuple[list[dict], dict]:
    reference = index_by_distance(reference_rows)
    candidate = index_by_distance(candidate_rows)
    distances = sorted(set(reference) & set(candidate))
    if not distances:
        raise ValueError("No matching distances found between correlation curves.")

    rows = []
    for distance in distances:
        ref = reference[distance]
        cand = candidate[distance]
        row = {"distance": distance}
        for name in ("fss", "fsv"):
            ref_value = float(ref[name])
            cand_value = float(cand[name])
            row[f"reference_{name}"] = ref_value
            row[f"candidate_{name}"] = cand_value
            row[f"delta_{name}"] = cand_value - ref_value
            row[f"relative_delta_{name}"] = (
                (cand_value - ref_value) / ref_value if abs(ref_value) > 1e-12 else 0.0
            )
        rows.append(row)

    summary = {}
    for name in ("fss", "fsv"):
        ref_values = np.asarray(
            [row[f"reference_{name}"] for row in rows], dtype=np.float64
        )
        cand_values = np.asarray(
            [row[f"candidate_{name}"] for row in rows], dtype=np.float64
        )
        delta = cand_values - ref_values
        summary[name] = {
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "mean_reference": float(ref_values.mean()),
            "mean_candidate": float(cand_values.mean()),
            "mean_delta": float(delta.mean()),
            "relative_mean_delta": float(delta.mean() / ref_values.mean())
            if abs(ref_values.mean()) > 1e-12
            else 0.0,
            "pearson": float(np.corrcoef(ref_values, cand_values)[0, 1])
            if len(rows) > 1
            else 1.0,
        }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Fss/Fsv correlation curve JSON files."
    )
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument(
        "--curve-key",
        choices=("radial_average", "directional"),
        default="radial_average",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = compare_rows(
        load_rows(args.reference_json, args.curve_key),
        load_rows(args.candidate_json, args.curve_key),
    )
    payload = {
        "reference_json": str(args.reference_json),
        "candidate_json": str(args.candidate_json),
        "curve_key": args.curve_key,
        "summary": summary,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(
        "Fss mae={:.6g} rmse={:.6g} mean_delta={:+.6g} pearson={:.4f}".format(
            summary["fss"]["mae"],
            summary["fss"]["rmse"],
            summary["fss"]["mean_delta"],
            summary["fss"]["pearson"],
        )
    )
    print(
        "Fsv mae={:.6g} rmse={:.6g} mean_delta={:+.6g} pearson={:.4f}".format(
            summary["fsv"]["mae"],
            summary["fsv"]["rmse"],
            summary["fsv"]["mean_delta"],
            summary["fsv"]["pearson"],
        )
    )
    print(f"Saved JSON to {args.output_json}")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
