"""Small, hand-worked checks for obsr.quoting_loop: build_fill_table's
look-ahead avoidance (lag_signal must come from strictly the previous
event) and measure_signal_aware_reduction's arithmetic, on a synthetic
session built to have a known answer, not the real 6.4M-row dataset."""
import pandas as pd

from obsr.quoting_loop import build_fill_table, measure_signal_aware_reduction


def _session(rows):
    df = pd.DataFrame(rows)
    df["ticker"] = "TST"
    df["date"] = "2025-01-01"
    return df


def test_fill_table_uses_previous_event_signal_not_current():
    # event_seq 0..4; a bid-side fill (EXECUTE/S) at seq 2. lag_signal for
    # that fill must be the signal value at seq 1, not seq 2's own value
    # (which would be look-ahead: the quote decision happens before the
    # fill that reveals it).
    rows = [
        {"event_seq": 0, "msg_type": "ADD", "side": "B", "mid": 100.0, "sig": 10.0},
        {"event_seq": 1, "msg_type": "ADD", "side": "B", "mid": 100.0, "sig": -999.0},
        {"event_seq": 2, "msg_type": "EXECUTE", "side": "S", "mid": 100.0, "sig": 12345.0},
        {"event_seq": 3, "msg_type": "ADD", "side": "B", "mid": 99.0, "sig": 1.0},
        {"event_seq": 4, "msg_type": "ADD", "side": "B", "mid": 98.0, "sig": 1.0},
    ]
    feat = _session(rows)
    fills = build_fill_table(feat, horizon_events=2, signal_col="sig")
    assert len(fills) == 1
    row = fills.iloc[0]
    assert row["event_seq"] == 2
    assert row["lag_signal"] == -999.0  # seq 1's value, not seq 2's 12345.0
    # mid_after = mid at seq 2+2=4 -> 98.0; adverse_selection = mid - mid_after = 100-98=2.0
    assert row["mid_after"] == 98.0
    assert row["adverse_selection"] == 2.0


def test_fill_table_ignores_ask_side_fills_and_add_events():
    rows = [
        {"event_seq": 0, "msg_type": "ADD", "side": "B", "mid": 100.0, "sig": 1.0},
        {"event_seq": 1, "msg_type": "EXECUTE", "side": "B", "mid": 100.0, "sig": 1.0},  # ask-side fill
        {"event_seq": 2, "msg_type": "CANCEL", "side": "S", "mid": 100.0, "sig": 1.0},
        {"event_seq": 3, "msg_type": "ADD", "side": "S", "mid": 100.0, "sig": 1.0},
    ]
    feat = _session(rows)
    fills = build_fill_table(feat, horizon_events=1, signal_col="sig")
    assert len(fills) == 0


def test_measure_signal_aware_reduction_arithmetic():
    # Four fills: two with a low lag_signal (below the median -> pulled)
    # and high adverse selection, two with a high lag_signal (kept) and
    # low adverse selection. pull_quantile=0.5 pulls the bottom half.
    fills = pd.DataFrame({
        "lag_signal": [-100.0, -90.0, 10.0, 20.0],
        "adverse_selection": [5.0, 5.0, 1.0, 1.0],
    })
    result = measure_signal_aware_reduction(fills, pull_quantile=0.5)
    assert result["n_fills_total"] == 4
    assert result["n_fills_pulled"] == 2
    assert result["n_fills_kept"] == 2
    assert result["baseline_mean_adverse_selection"] == 3.0
    assert result["signal_aware_mean_adverse_selection"] == 1.0
    assert abs(result["pct_reduction"] - (200.0 / 3.0)) < 1e-9
    assert result["mean_adverse_selection_of_pulled_fills"] == 5.0
