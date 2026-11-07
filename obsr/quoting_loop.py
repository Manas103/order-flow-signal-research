"""Wires the online incremental order-flow-imbalance signal into a simple
quoting-loop decision: a market maker resting passively at the best bid
pulls the quote (does not take the fill) whenever the signal, computed
from strictly the *previous* event (never the fill event itself, to avoid
look-ahead), predicts net selling pressure already in progress. "Adverse
selection paid per fill" is the mid-price move against the market maker
over a fixed event-count horizon after each fill it actually takes.

Only the bid side is modeled (the market maker is filled on a
seller-initiated EXECUTE, msg_type == 'EXECUTE' and side == 'S', meaning
the aggressor sold into the resting bid); see README "Honest framing" for
why the ask side is not separately modeled (it is the mirror image by
construction, not a materially different mechanism).
"""
from __future__ import annotations

import pandas as pd

HORIZON_EVENTS = 50
SIGNAL_COLUMN = "trade_flow_imb_w50"


def build_fill_table(feat: pd.DataFrame, horizon_events: int = HORIZON_EVENTS,
                      signal_col: str = SIGNAL_COLUMN) -> pd.DataFrame:
    """One row per bid-side fill, with the mid at the fill, the mid
    `horizon_events` events later in the same (ticker, date) session, and
    the signal value as of the *previous* event (the value genuinely
    available to a quoting decision made before this fill happened)."""
    df = feat[["ticker", "date", "event_seq", "msg_type", "side", "mid", signal_col]].copy()
    df = df.sort_values(["ticker", "date", "event_seq"]).reset_index(drop=True)
    grouped = df.groupby(["ticker", "date"], sort=False)
    df["mid_after"] = grouped["mid"].shift(-horizon_events)
    df["lag_signal"] = grouped[signal_col].shift(1)

    fills = df[(df["msg_type"] == "EXECUTE") & (df["side"] == "S")].copy()
    fills = fills.dropna(subset=["mid_after", "lag_signal"])
    # Positive == the mid fell after the market maker bought, i.e. a real
    # cost paid for taking that fill; negative means the fill was, in
    # hindsight, favorably timed.
    fills["adverse_selection"] = fills["mid"] - fills["mid_after"]
    return fills


def measure_signal_aware_reduction(fills: pd.DataFrame, pull_quantile: float) -> dict:
    """`pull_quantile` is a fixed, pre-specified quantile of `lag_signal`'s
    own distribution (not of the adverse-selection outcome, which would be
    look-ahead): a fill is "pulled" (excluded from the signal-aware mean)
    when `lag_signal` falls at or below that quantile, i.e. the online
    signal showed strong net selling pressure immediately before this
    fill happened."""
    threshold = float(fills["lag_signal"].quantile(pull_quantile))
    baseline_mean = float(fills["adverse_selection"].mean())
    kept = fills[fills["lag_signal"] > threshold]
    pulled = fills[fills["lag_signal"] <= threshold]
    signal_aware_mean = float(kept["adverse_selection"].mean())
    pct_reduction = (baseline_mean - signal_aware_mean) / baseline_mean * 100.0
    return {
        "pull_quantile": pull_quantile,
        "lag_signal_threshold": threshold,
        "n_fills_total": int(len(fills)),
        "n_fills_pulled": int(len(pulled)),
        "n_fills_kept": int(len(kept)),
        "baseline_mean_adverse_selection": baseline_mean,
        "signal_aware_mean_adverse_selection": signal_aware_mean,
        "pct_reduction": pct_reduction,
        "mean_adverse_selection_of_pulled_fills": float(pulled["adverse_selection"].mean()),
    }
