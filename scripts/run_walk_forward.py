"""Walk-forward evaluation of the price-move signal, with significance tests.

The base study (scripts/run_research.py) scores one purged split. This asks the
follow-up questions a single split cannot answer:

1. Is the result stable through time? Expanding-window walk-forward: for every
   test day from the 11th on, train on all earlier days except a one-day
   embargo, then test on that day (6 folds, every ticker).
2. Does the tree beat a linear model because it is nonlinear, or only because
   it sees more features? Two linear baselines: one-feature OFI (the
   Cont-Kukanov-Stoikov form) and ridge regression on all 43 features.
3. Is any edge real? Diebold-Mariano tests on squared error and a session
   bootstrap interval for the R^2 difference, both treating each (ticker, day)
   session as one independent unit, plus the per-session information
   coefficient (Spearman) and its t-statistic.

Training rows are thinned to every STRIDE-th event within a session. Labels at
horizon h >= 10 overlap their neighbours almost entirely, so adjacent rows
carry nearly the same information, and thinning keeps each of the 54 fits
(6 folds x 3 horizons x 3 models) to seconds. Test rows are never thinned.

Usage: python scripts/run_walk_forward.py [--data-dir data] [--stride 10]
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from obsr.features import FEATURE_COLUMNS, build_features
from obsr.labels import HORIZONS, label_col
from obsr.stats import (
    diebold_mariano_sessions, hit_rate, information_coefficient, r2_oos,
    session_bootstrap_r2_diff, walk_forward_folds,
)
from obsr.store import connect, messages_relation

MIN_TRAIN_DAYS = 9
EMBARGO_DAYS = 1
GBT_PARAMS = dict(  # identical to run_research.py, the reported third attempt
    n_estimators=200, num_leaves=15, max_depth=5, learning_rate=0.03,
    min_child_samples=1000, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=-1,
)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, col: str) -> dict[str, np.ndarray]:
    X, y = train[list(FEATURE_COLUMNS)], train[col]
    gbt = lgb.LGBMRegressor(**GBT_PARAMS).fit(X, y)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    ofi = LinearRegression().fit(train[["ofi_agg5"]], y)
    Xt = test[list(FEATURE_COLUMNS)]
    return {"gbt": gbt.predict(Xt), "ridge_43": ridge.predict(Xt), "linear_ofi": ofi.predict(test[["ofi_agg5"]])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", default="docs/walk_forward_results.json")
    args = ap.parse_args()

    t0 = time.time()
    con = connect()
    feat = build_features(con, messages_relation(con, args.data_dir))
    feat["session"] = feat["ticker"].astype(str) + "/" + feat["date"].astype(str)
    dates = sorted(feat["date"].unique())
    print(f"features: {len(feat):,} rows x {len(FEATURE_COLUMNS)} columns, {feat['ticker'].nunique()} tickers, "
          f"{len(dates)} days, built in {time.time() - t0:.1f}s; training stride {args.stride}")

    folds = list(walk_forward_folds(dates, MIN_TRAIN_DAYS, EMBARGO_DAYS))
    report = {"stride": args.stride, "min_train_days": MIN_TRAIN_DAYS, "embargo_days": EMBARGO_DAYS,
              "n_folds": len(folds), "horizons": {}}
    for h in HORIZONS:
        col = label_col(h)
        data = feat.dropna(subset=list(FEATURE_COLUMNS) + [col])
        per_fold, pooled = [], []
        for train_dates, _, test_date in folds:
            tr = data[data["date"].isin(train_dates)]
            tr = tr[tr["event_seq"] % args.stride == 0]
            te = data[data["date"] == test_date]
            preds = fit_predict(tr, te, col)
            ref = float(tr[col].mean())
            row = {"test_date": str(test_date), "n_train": len(tr), "n_test": len(te)}
            row.update({f"r2_{m}": r2_oos(te[col], p, ref) for m, p in preds.items()})
            per_fold.append(row)
            pooled.append(pd.DataFrame({"y": te[col].to_numpy(), "session": te["session"].to_numpy(),
                                        "ref": ref, **{m: p for m, p in preds.items()}}))
            print(f"h={h:5d} test={test_date} n_train={len(tr):>8,} n_test={len(te):>8,}  "
                  + "  ".join(f"{m}={row[f'r2_{m}']:+.4f}" for m in preds))
        P = pd.concat(pooled, ignore_index=True)
        # pooled R^2 uses each fold's own training mean as the no-skill reference
        sst = float(((P["y"] - P["ref"]) ** 2).sum())
        r2 = {m: 1 - float(((P["y"] - P[m]) ** 2).sum()) / sst for m in ("gbt", "ridge_43", "linear_ofi")}
        fold_r2 = pd.DataFrame(per_fold)
        out = {
            "pooled_r2": r2,
            "fold_r2_mean": {m: float(fold_r2[f"r2_{m}"].mean()) for m in r2},
            "fold_r2_std": {m: float(fold_r2[f"r2_{m}"].std(ddof=1)) for m in r2},
            "folds_gbt_beats_ofi": int((fold_r2["r2_gbt"] > fold_r2["r2_linear_ofi"]).sum()),
            "folds_gbt_beats_ridge": int((fold_r2["r2_gbt"] > fold_r2["r2_ridge_43"]).sum()),
            "per_fold": per_fold,
        }
        P["ref_pred"] = P["ref"]
        for name, b in (("vs_linear_ofi", "linear_ofi"), ("vs_ridge_43", "ridge_43"), ("vs_train_mean", "ref_pred")):
            dm = diebold_mariano_sessions(P["y"], P["gbt"], P[b], P["session"])
            out[f"dm_gbt_{name}"] = dm.__dict__
        # bootstrap R^2 differences with per-row reference = that fold's training mean
        for name, b in (("vs_linear_ofi", "linear_ofi"), ("vs_ridge_43", "ridge_43")):
            yc = P["y"] - P["ref"]
            pt, lo, hi = session_bootstrap_r2_diff(yc, P["gbt"] - P["ref"], P[b] - P["ref"], P["session"], mean_ref=0.0)
            out[f"r2_diff_gbt_{name}"] = {"point": pt, "ci95": [lo, hi]}
        for m in ("gbt", "ridge_43", "linear_ofi"):
            ic = information_coefficient(P["y"], P[m], P["session"])
            out[f"ic_{m}"] = ic.__dict__
            out[f"hit_rate_{m}"] = hit_rate(P["y"], P[m])
        report["horizons"][str(h)] = out
        print(f"h={h:5d} pooled R^2: " + "  ".join(f"{m}={v:+.4f}" for m, v in r2.items())
              + f" | GBT>OFI in {out['folds_gbt_beats_ofi']}/{len(folds)} folds, GBT>ridge in {out['folds_gbt_beats_ridge']}/{len(folds)}")
        for name in ("vs_linear_ofi", "vs_ridge_43", "vs_train_mean"):
            d = out[f"dm_gbt_{name}"]
            print(f"         DM GBT {name}: mean loss diff {d['mean_loss_diff']:+.3e}, t={d['t_stat']:+.2f}, "
                  f"p={d['p_value']:.3g} ({d['n_sessions']} sessions)")
        for name in ("vs_linear_ofi", "vs_ridge_43"):
            r = out[f"r2_diff_gbt_{name}"]
            print(f"         R^2 diff GBT {name}: {r['point']:+.4f}, 95% session-bootstrap CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
        for m in ("gbt", "ridge_43", "linear_ofi"):
            ic = out[f"ic_{m}"]
            print(f"         IC {m}: {ic['mean_ic']:+.4f} (t={ic['t_stat']:+.2f}, p={ic['p_value']:.3g}), "
                  f"hit rate {out[f'hit_rate_{m}']:.4f}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\ndone in {time.time() - t0:.1f}s; results in {args.out}")


if __name__ == "__main__":
    main()
