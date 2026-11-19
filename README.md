# Order-Flow Signal Research

[![CI](https://github.com/Manas103/order-flow-signal-research/actions/workflows/ci.yml/badge.svg)](https://github.com/Manas103/order-flow-signal-research/actions/workflows/ci.yml)

Message-level market microstructure research on a simulated limit order
book: 6.4 million order messages, 43 features built in one DuckDB SQL
window pass, a gradient-boosted model tested against two linear baselines
with purged walk-forward validation and session-level significance tests,
plus three follow-up studies (queue position and fill probability, a
10-minute volatility forecast, and an online C++ feature engine driving a
quoting rule).

Python (pandas, NumPy, SciPy, scikit-learn, LightGBM), DuckDB SQL over
Hive-partitioned Parquet, C++20, pytest, GitHub Actions, Docker.

## Honest framing

- **The data is synthetic.** `obsr/simulate.py` generates an ITCH-style
  add/execute/cancel/delete stream for 5 made-up tickers over 16 trading
  days, calibrated to published microstructure facts (bursty arrivals,
  heavy-tailed sizes, order flow with short-horizon predictive power). It is
  not fit to any real dataset.
- **The signal the models find belongs to this simulator.** Prices follow a
  latent momentum state that the order flow partly reveals. The study shows
  that the pipeline can find a signal and test it honestly, not that the
  same signal exists in a real market.
- **The first version of this study got its main result wrong.** It
  reported no usable signal. A one-sided book at the start of every session
  produced a fake mid at half the real price, and the 109 bad rows (out of
  6.4M) carried almost all of the label variance, squeezing every model's
  R^2 to zero. The walk-forward work exposed it. The numbers below are after
  the fix, with the model's hyperparameters unchanged. The full account is in
  [DESIGN.md](DESIGN.md#findings).

## Results

| Study | Measured |
|---|---|
| Price-move forecast, 10 events ahead (walk-forward, 6 folds, 30 sessions) | GBT R^2 **+0.154** vs ridge on 43 features +0.013 and one-feature OFI +0.000; GBT wins every fold; Diebold-Mariano p = 7.7e-11 vs ridge |
| Same, 100 events ahead | GBT R^2 **+0.106** vs ridge +0.019; wins every fold; p = 1.2e-05 |
| Same, 1,000 events ahead | GBT +0.017 vs ridge +0.021; **no significant difference** (p = 0.82) |
| Fill probability at the touch, 1-second horizon | rank 1: **0.5377**, rank 5: **0.0410**, smooth near-geometric decay |
| Rank-aware quote-and-wait vs always crossing | **0.64%** lower effective spread, essentially no improvement |
| 10-minute realized volatility forecast (140 test bins) | RMSPE **0.540** vs **7.265** for the previous-window baseline (92.6% lower; the baseline is unusually weak here, see Limitations) |
| C++ online feature engine vs the SQL pass, 6.4M rows | **38 of 43** columns bit-identical, the other 5 within 1.7e-16, 0 NULL mismatches; about **0.1us** median per event |
| Pulling the bid on a weak trade-flow signal | **7.18%** less adverse selection per fill at the 20% quantile (4.06% at 10%, 9.75% at 25%) |

Every number comes from a committed script and raw output under `docs/`.

## How it is built

```
obsr/simulate.py        message generator (seeded, deterministic)
obsr/book.py            price-time priority book, rejects crossing adds
obsr/store.py           Hive-partitioned Parquet (ticker=/date=), read through DuckDB
obsr/features.py        43 features + forward labels in one SQL window pass, PARTITION BY ticker, date
obsr/stats.py           walk-forward folds, Diebold-Mariano, session bootstrap, information coefficient
obsr/queue_position.py  L3 replay recovering each order's queue rank
obsr/fill_probability.py, obsr/execution_cost.py   fill probability by rank (one SQL pass), cost comparison
obsr/volatility.py      10-minute bins, LightGBM forecast, RMSPE
cpp/                    IncrementalFeatureEngine: all 43 features per event, O(1) amortized
obsr/quoting_loop.py    signal-aware quote pulling
scripts/                one entry point per study, each writes docs/
```

Design choices worth knowing:

- **Leakage is prevented by construction.** Every window is
  `PARTITION BY ticker, date`, so no feature or label crosses a session, and
  the split is by whole days with a one-day embargo (wider than the longest
  label horizon).
- **Significance is tested per session, not per event.** A horizon-h label
  overlaps its next h-1 neighbors, so event-level standard errors would be
  far too small. Each (ticker, day) session counts as one independent unit.
- **Two linear baselines, not one.** One-feature OFI tests the textbook
  linear relationship; ridge on all 43 features tests whether the tree wins
  through nonlinearity or just through more inputs. At 10 and 100 events it
  is nonlinearity.
- **Every core computation has an independent check:** the book against a
  brute-force oracle, the SQL labels against a pandas re-implementation, the
  C++ engine against the SQL pass on all 6.4M rows.

[DESIGN.md](DESIGN.md) has the architecture, every validation step, what
broke along the way, and the full results tables.

## Build and reproduce

```bash
pip install -r requirements.txt
python -m pytest tests -q                     # 31 tests, about 15 seconds
scripts/reproduce.sh                          # generates data/ (220 MB), reruns every study into repro_out/
CPP=1 scripts/reproduce.sh                    # also builds the C++ engine and diffs it against SQL (Linux)
```

Or in Docker:

```bash
docker build -t order-flow-signal-research .
docker run --rm order-flow-signal-research    # tests
```

CI runs the Python tests and builds the C++ engine twice, in Release and
with AddressSanitizer and UndefinedBehaviorSanitizer (halting on the first
report).

## Limitations

- Synthetic data throughout, from one generator seed. The walk-forward test
  covers 6 days x 5 tickers = 30 sessions, so the significance tests have 29
  degrees of freedom; that is ample for p-values near 1e-10, and the reason
  the 1,000-event result is read as no evidence rather than evidence of no
  effect.
- One-feature OFI scores about zero because of a generator calibration
  choice (momentum shows in aggressor trade flow, not resting-side OFI). It
  is not a statement about OFI in real markets.
- R^2 and information coefficient are forecast metrics. No transaction
  costs, market impact or capacity are modeled for the price-move forecast.
- Walk-forward training rows are thinned to every 10th event to keep 54
  model fits fast; test rows are never thinned.
- The rank-aware execution comparison is an expected-value calculation over
  12,720 decision points, not a path-dependent backtest, and ignores the
  rule's own impact and competing orders.
- The generator has no volatility clustering, so the previous-window
  volatility baseline is far weaker than it would be on real data, which
  inflates the 92.6% gain. The high/low regime split (95.6% vs 53.5%)
  reverses if regimes are defined by message count instead of volatility.
- 5 of the 43 C++ columns differ from SQL in the last bit because
  floating-point addition order differs. Latency is timed with a clock read
  around each call, so it is an upper bound, and the latest run was on a
  busy machine.
- The quoting rule covers the bid side only. Pulled fills are dropped rather
  than repriced, and the spread a real market maker gives up by pulling is
  not charged against the reduction.
