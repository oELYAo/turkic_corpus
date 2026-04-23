#!/usr/bin/env python3
"""Merge native + normalized downstream summary CSVs into one JSON table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "downstream"


def read_summary(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            c = row["condition"]
            d = {
                "las_mean": float(row["las_mean"]),
                "las_std": float(row["las_std"]),
                "uas_mean": float(row["uas_mean"]),
                "uas_std": float(row["uas_std"]),
            }
            if row.get("n_runs"):
                d["k_folds"] = float(row["k_folds"])
                d["n_init_seeds"] = float(row["n_init_seeds"])
                d["n_runs"] = float(row["n_runs"])
            elif row.get("n_folds") is not None:
                d["n_folds"] = float(row["n_folds"])
            rows[c] = d
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=16000)
    args = p.parse_args()
    v = args.vocab_size
    native_p = OUT_DIR / f"ud_depparse_tatar_native_v{v}_summary.csv"
    norm_p = OUT_DIR / f"ud_depparse_tatar_normalized_v{v}_summary.csv"
    if not native_p.is_file() or not norm_p.is_file():
        raise SystemExit(f"Need both {native_p} and {norm_p}. Run run_downstream_depparse.py for each script.")
    merged = {
        "vocab_size": v,
        "native": read_summary(native_p),
        "normalized": read_summary(norm_p),
    }
    out = OUT_DIR / f"ud_depparse_tatar_merged_v{v}.json"
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
