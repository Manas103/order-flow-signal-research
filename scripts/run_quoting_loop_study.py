"""Measures how much the online incremental signal, wired into a simple
quoting-loop pull decision, cuts adverse selection paid per fill. The
reported quantile (0.20) was fixed before any result was seen; 0.10 and
0.25 were also run as a sweep (docs/quoting_loop_sweep.json, DESIGN.md).

Usage: python scripts/run_quoting_loop_study.py [--data-dir data] [--quantile 0.2]
"""
from __future__ import annotations

import argparse
import json

from obsr.features import build_features
from obsr.quoting_loop import build_fill_table, measure_signal_aware_reduction
from obsr.store import connect, messages_relation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--quantile", type=float, default=0.20)
    ap.add_argument("--report", default="docs/quoting_loop_output.json")
    args = ap.parse_args()

    con = connect()
    rel = messages_relation(con, args.data_dir)
    feat = build_features(con, rel)
    print(f"features: {len(feat):,} rows")

    fills = build_fill_table(feat)
    print(f"bid-side fills with a defined horizon and lagged signal: {len(fills):,}")

    result = measure_signal_aware_reduction(fills, pull_quantile=args.quantile)
    print(json.dumps(result, indent=2))

    with open(args.report, "w") as f:
        json.dump(result, f, indent=2)
    print(f"written to {args.report}")


if __name__ == "__main__":
    main()
