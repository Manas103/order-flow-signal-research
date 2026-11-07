"""Exports the Parquet message store to a flat, packed binary file the C++
incremental feature engine (cpp/) reads directly, with no CSV or Parquet
parsing on the C++ side. Row order is ORDER BY ticker, date, event_seq,
the exact order compare_incremental_to_sql.py re-sorts the DuckDB SQL
feature pass into before comparing them positionally; this script and
that one must agree on that order or the comparison is meaningless.

Usage: python scripts/export_messages_flat.py [--data-dir data] [--out /tmp/messages.bin]
"""
from __future__ import annotations

import argparse

import numpy as np

from obsr.store import connect, messages_relation

# Byte-for-byte the same layout as cpp/include/raw_event.hpp's RawEvent
# (align=False: no padding between fields, matching #pragma pack(1)).
RAW_EVENT_DTYPE = np.dtype(
    [
        ("event_seq", "<i8"),
        ("group_id", "<i4"),
        ("msg_type_id", "<i4"),
        ("side_id", "<i4"),
        ("ts", "<f8"),
        ("qty", "<f8"),
        ("size_delta", "<f8"),
        ("bid_px", "<f8", (5,)),
        ("bid_sz", "<f8", (5,)),
        ("ask_px", "<f8", (5,)),
        ("ask_sz", "<f8", (5,)),
    ],
    align=False,
)

MSG_TYPE_ID = {"ADD": 0, "CANCEL": 1, "EXECUTE": 2, "DELETE": 3}
SIDE_ID = {"B": 0, "S": 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="/tmp/obsr_messages.bin")
    args = ap.parse_args()

    con = connect()
    rel = messages_relation(con, args.data_dir)
    df = con.execute(f"SELECT * FROM {rel} ORDER BY ticker, date, event_seq").fetchdf()
    print(f"loaded {len(df):,} rows")

    df["group_id"] = (
        df["ticker"].astype(str) + "|" + df["date"].astype(str)
    ).astype("category").cat.codes.astype(np.int32)

    out = np.zeros(len(df), dtype=RAW_EVENT_DTYPE)
    out["event_seq"] = df["event_seq"].to_numpy()
    out["group_id"] = df["group_id"].to_numpy()
    out["msg_type_id"] = df["msg_type"].map(MSG_TYPE_ID).to_numpy()
    out["side_id"] = df["side"].map(SIDE_ID).fillna(-1).to_numpy()
    out["ts"] = df["ts"].to_numpy()
    out["qty"] = df["qty"].to_numpy()
    out["size_delta"] = df["size_delta"].to_numpy()
    for k in range(5):
        out["bid_px"][:, k] = df[f"bid_px_{k + 1}"].to_numpy()
        out["bid_sz"][:, k] = df[f"bid_sz_{k + 1}"].to_numpy()
        out["ask_px"][:, k] = df[f"ask_px_{k + 1}"].to_numpy()
        out["ask_sz"][:, k] = df[f"ask_sz_{k + 1}"].to_numpy()

    out.tofile(args.out)
    print(f"wrote {len(out):,} rows ({out.nbytes:,} bytes) to {args.out}")
    print(f"item size: {RAW_EVENT_DTYPE.itemsize} bytes "
          f"(must equal sizeof(RawEvent) in cpp/include/raw_event.hpp)")


if __name__ == "__main__":
    main()
