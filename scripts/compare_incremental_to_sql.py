"""Diffs the C++ incremental feature engine's output (cpp/apps/incremental_bench,
read via read_incremental_output below) against obsr.features.build_features's
real DuckDB SQL window-function pass, column by column, over the full
6.4M-row store. Both passes are sorted ORDER BY ticker, date, event_seq
before comparing, matching scripts/export_messages_flat.py's row order.

Usage:
    python scripts/export_messages_flat.py --out /tmp/obsr_messages.bin
    ./cpp/build/incremental_bench /tmp/obsr_messages.bin /tmp/obsr_incremental_out.bin
    python scripts/compare_incremental_to_sql.py --incremental-out /tmp/obsr_incremental_out.bin
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from obsr.features import FEATURE_COLUMNS, build_features
from obsr.store import connect, messages_relation

NUM_FEATURES = len(FEATURE_COLUMNS)
assert NUM_FEATURES == 43


def read_incremental_output(path: str, n_rows: int) -> np.ndarray:
    arr = np.fromfile(path, dtype="<f8")
    assert arr.size == n_rows * NUM_FEATURES, (
        f"expected {n_rows * NUM_FEATURES} doubles, read {arr.size}"
    )
    return arr.reshape(n_rows, NUM_FEATURES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--incremental-out", default="/tmp/obsr_incremental_out.bin")
    ap.add_argument("--report", default="docs/incremental_vs_sql_report.json")
    args = ap.parse_args()

    con = connect()
    rel = messages_relation(con, args.data_dir)
    sql_feat = build_features(con, rel)
    sql_feat = sql_feat.sort_values(["ticker", "date", "event_seq"]).reset_index(drop=True)
    print(f"SQL pass: {len(sql_feat):,} rows, {NUM_FEATURES} feature columns")

    inc = read_incremental_output(args.incremental_out, len(sql_feat))

    per_column = []
    for j, col in enumerate(FEATURE_COLUMNS):
        sql_vals = sql_feat[col].to_numpy(dtype=np.float64)
        inc_vals = inc[:, j]

        both_nan = np.isnan(sql_vals) & np.isnan(inc_vals)
        nan_mismatch = np.isnan(sql_vals) != np.isnan(inc_vals)
        comparable = ~both_nan & ~nan_mismatch
        exact_match = comparable & (sql_vals == inc_vals)
        diff = np.full(len(sql_vals), np.nan)
        diff[comparable] = np.abs(sql_vals[comparable] - inc_vals[comparable])

        n_comparable = int(comparable.sum())
        n_exact = int(exact_match.sum())
        max_abs_diff = float(np.nanmax(diff)) if n_comparable > 0 else 0.0
        n_nan_mismatch = int(nan_mismatch.sum())

        per_column.append({
            "column": col,
            "n_rows": len(sql_vals),
            "n_both_nan": int(both_nan.sum()),
            "n_nan_mismatch": n_nan_mismatch,
            "n_comparable": n_comparable,
            "n_exact_match": n_exact,
            "bit_identical": bool(n_nan_mismatch == 0 and n_exact == n_comparable),
            "max_abs_diff": max_abs_diff,
        })
        status = "BIT-IDENTICAL" if per_column[-1]["bit_identical"] else "close"
        print(f"{col:28s} {status:14s} exact={n_exact}/{n_comparable} "
              f"nan_mismatch={n_nan_mismatch} max_abs_diff={max_abs_diff:.3e}")

    n_bit_identical_cols = sum(1 for c in per_column if c["bit_identical"])
    summary = {
        "n_rows": len(sql_feat),
        "n_columns": NUM_FEATURES,
        "n_bit_identical_columns": n_bit_identical_cols,
        "columns": per_column,
    }
    with open(args.report, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{n_bit_identical_cols} / {NUM_FEATURES} columns bit-identical to the SQL pass")
    print(f"full report written to {args.report}")


if __name__ == "__main__":
    main()
