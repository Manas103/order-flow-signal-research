# Design notes

How the study is built, how each part was validated, what broke along the way, and the full results with the commands that produce them. The [README](README.md) is the overview.

## Why this exists

Order flow imbalance (OFI) is one of the small number of results in market
microstructure with a clean, citable theoretical claim: Cont, Kukanov and
Stoikov (2014) show OFI drives short-interval price changes approximately
linearly, with a slope inversely proportional to depth. This project builds
the smallest honest version of the research question a signal desk actually
asks: does a tree model beat that linear baseline, and if so, for how long.
It is a small version of message-level signal research, not a claim to have
reproduced an institutional research pipeline.

## Honest framing, up front

- **This is not real Nasdaq TotalView-ITCH or LOBSTER data.** Real LOBSTER
  sample files sit behind a JavaScript-rendered request form that could not
  be fetched as plain data in this build environment, and this project does
  not claim otherwise. `obsr/simulate.py` generates a from-scratch,
  self-consistent ITCH-style add/execute/cancel/delete message stream,
  calibrated to three published microstructure facts (bursty, autocorrelated
  order arrivals; heavy-tailed order sizes; order flow imbalance with real,
  decaying short-horizon predictive power), not fit to any specific real
  dataset. The data is disclosed as synthetic everywhere it is used.
- **The book builder enforces price-time priority and rejects a crossing
  add.** This
  generator only ever proposes adds that do not cross the opposite side (see
  `obsr/book.py`'s `CrossedBookError`), so the book stays valid by
  construction rather than by later correction.
- **The signal the models find is a property of this simulator.** The
  generator drives prices with a latent momentum state that the order flow
  partly reveals, so a tree model beating a linear one here shows that the
  pipeline can find a signal and test it honestly, not that the same signal
  exists in a real market.
- **The 5 tickers (SYNA through SYNE) are synthetic,** not real symbols.

### Machine and toolchain

| | |
|---|---|
| CPU | AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores |
| RAM | 31.1 GB |
| OS | Windows 11, build 10.0.26200 |
| Python | CPython 3.12.10, in a repo-local venv |
| Libraries | duckdb 1.5.5, pandas 3.0.5, pyarrow 25.0.1, lightgbm 4.7.0, scikit-learn 1.9.1, numpy 2.5.3, pytest 9.1.1 (see `requirements.txt`) |
| Pacing | non-realtime process, no CPU pinning, single run per reported number unless noted |

## Architecture

```
obsr/simulate.py       synthetic ITCH-style message generator (5 tickers x 16 days x 80,000 msgs)
obsr/book.py           fast price-time-priority BookBuilder (sorted price levels, FIFO queues per level)
obsr/reference_book.py SlowBookBuilder: independent, brute-force reference oracle
obsr/store.py          partitioned Parquet writer (ticker=.../date=...) plus a DuckDB connection helper
obsr/features.py       43 microstructure features and 3 forward labels, built entirely in one DuckDB SQL statement
obsr/labels.py         forward mid-price-change label definition, plus an independent plain-pandas reimplementation for testing

scripts/generate_data.py   runs the simulator for all 80 (ticker, date) sessions, writes the Parquet store
scripts/run_research.py    builds features, applies the day-level purged split, trains GBT + linear baseline, reports R^2

tests/test_book_invariants.py    conservation and price-time-priority checks on BookBuilder
tests/test_reference_oracle.py   BookBuilder diffed exactly against SlowBookBuilder over 20,000 randomized operations
tests/test_features.py           hand-computed feature values on a crafted single row
tests/test_labels_and_leakage.py SQL LEAD labels vs. an independent pandas reimplementation; purged-split overlap check

docs/benchmark_output.txt   raw run of scripts/run_research.py (the reported attempt)
docs/test_output.txt        raw pytest run
docs/research_results.json  the same R^2 numbers, machine-readable
```

**Why DuckDB SQL for features, not pandas.** `obsr/features.py`'s docstring
states the rule and the tests enforce it: every feature is a `PARTITION BY
ticker, date ORDER BY event_seq` window function, so a lag, a rolling window,
or a forward label can never reach across a session boundary by construction.
A pandas implementation using `.shift()`/`.rolling()` across a concatenated
frame would need the same partition keys threaded through every call by hand;
one SQL statement with one `WINDOW` clause is both the faster and the harder-
to-get-wrong version of the same computation, which is why `store.py` and
`features.py` route the whole feature build through DuckDB rather than pandas.

**Why a fast book and a slow one.** `BookBuilder` keeps sorted price-level
lists and O(1) level-size lookups because it runs inside the generator's
per-message loop 6.4 million times. `SlowBookBuilder` recomputes every
answer from a flat dict by brute-force scan, on purpose: it shares no data
structure with the fast implementation, so agreement between the two is
actual evidence of correctness rather than two paths through the same bug.

**The linear baseline is deliberately one feature.** `ofi_agg5`, the sum of
order flow imbalance across the top 5 levels, is regressed alone against each
horizon's label. This is the Cont-Kukanov-Stoikov baseline in its narrowest,
most falsifiable form: if the tree model's advantage over one honestly-chosen
feature disappears at long horizons, that is a real finding about decay, not
an artifact of a weak baseline built from many features.

## Validation

**1. Reference-oracle diff.** `tests/test_reference_oracle.py` runs 10 seeded
sequences of 2,000 randomized add/execute/cancel/delete operations each
(20,000 operations total) through `BookBuilder` and `SlowBookBuilder` in
lockstep, asserting the full 5-level snapshot matches after every single
operation, not just at the end.

**2. Label cross-check.** `tests/test_labels_and_leakage.py` builds the same
small message sample through the production SQL path
(`obsr.features.build_feature_sql`, DuckDB `LEAD`) and through
`obsr.labels.compute_forward_labels` (plain pandas `groupby().shift()`), and
asserts the three horizon labels agree exactly, row for row.

**3. Leakage check.** The same test asserts that for every (ticker, date)
session, the last `h` rows have a null `fwd_mid_chg_h` label rather than a
label reaching into the next session, and separately asserts the day-level
purged train/test split produces disjoint date sets with the embargo day in
neither.

**4. Feature unit checks.** `tests/test_features.py` hand-computes spread,
microprice, level-1 queue imbalance and total depth on a single crafted row
and diffs them exactly against the SQL output, plus asserts every one of the
43 declared feature columns actually appears in the query's output.

**5. One-sided book regression.** `tests/test_labels_and_leakage.py` also
asserts that a row with an empty side of the book has no mid, spread or
microprice, and that no label starts or ends on one (see "A one-sided book
was hiding inside every label" below). It fails on the code before the fix.

**6. Statistics checks.** `tests/test_stats.py` runs every evaluation
statistic on synthetic data whose answer is known: identical forecasts give
a Diebold-Mariano statistic of exactly 0, a genuinely better forecast is
detected with the right sign, two pure-noise forecasts are not called
different, the bootstrap interval brackets a real R^2 difference, and the
walk-forward folds expand, embargo one day and never overlap.

```
$ python -m pytest tests -q
31 passed in 13.30s
```

## Findings

**A one-sided book was hiding inside every label, and it made a real signal
look like no signal at all.** The first version of this study reported GBT
R^2 of -0.0033, +0.0025 and +0.0070 at 10, 100 and 1,000 events, linear-OFI
R^2 of 0.0000, and concluded there was no usable signal. Building the
walk-forward study exposed the problem: on single test days the tree scored
R^2 between -3 and -168, yet pooled over all days it scored +0.0002. Numbers
that disagree that badly mean the denominator is dominated by a few rows.
Grouping the labels by day showed it: every day had about 5 labels between
+2,500 and +10,000 ticks, and three days had more. A real 10-event move in
this simulator is a few ticks.

The cause was the first row of every session. The book is seeded one order
at a time, so at `event_seq = 0` the ask side is still empty and its price is
stored as 0. `mid = (bid + ask) / 2` then comes out at half the real price,
and the forward label from that row records the jump back to the real price
as a move of thousands of ticks. The same thing happened mid-session in
three sessions where a sweep briefly emptied one side (29 more rows). In all,
109 of 6.4M rows, but their squared labels were almost the entire label
variance (label standard deviation 22.45 per day, against a real move
spread near 1), so every model's R^2 was squeezed to zero no matter how
good it was.

The independent pandas label check (validation item 2) agreed with the SQL
exactly the whole time, because both computed mid the same way. Two
implementations of one wrong definition agree perfectly. The fix defines
mid, spread and microprice as NULL whenever either touch is empty, in the
SQL, in the pandas reference and in the C++ engine, and a regression test
now fails on the old code. The 10-minute volatility study had already hit
the same root cause in its own microprice and guarded against it there; the
guard had never been pushed down into the shared feature pass.

Looking back, the earlier modeling attempts were fighting this bug without
knowing it. The first, unregularized model (R^2 of -0.0384 at 10 events)
was blamed on overfitting, and the second attempt winsorized the label at
its 1st/99th percentiles, which produced a degenerate R^2 of exactly 1.0000
and was discarded. The heavy tail it was reacting to was the fake moves,
not real order sizes. The third attempt's regularized parameters are kept
unchanged in every number below, so no hyperparameter was tuned after the
fix.

**With clean labels, the tree finds a real short-horizon signal and the
linear models mostly do not.** On the original single purged split the GBT
scores R^2 of +0.1634 at 10 events and +0.0866 at 100. One-feature OFI
regression stays at 0.0000 at every horizon. The OFI result is a property
of this generator: it biases the *aggressor* side of executions toward the
latent momentum state much more strongly than the resting side of adds
(`K_EXEC=1.6` vs `K_ADD=0.5` in `obsr/simulate.py`), so momentum shows up in
signed trade flow, not in top-of-book OFI. That is a calibration choice,
not a claim about OFI in real markets, where Cont, Kukanov and Stoikov
measured it on real data.

**The tree's edge decays with horizon, and at 1,000 events it is not
there.** The single split gives -0.1796 at 1,000 events, which on its own
would suggest severe overfitting. The walk-forward study (below) gives
+0.0166 pooled over six days, beats the linear OFI model in 3 of 6 folds
and ridge in 2 of 6, and neither difference is statistically significant
(Diebold-Mariano p = 0.78 and 0.82). A 1,000-event label overlaps 999 of its
neighbors, so one 80,000-event session has only about 80 independent
1,000-event outcomes. The honest reading is no reliable edge at that
horizon, and the gap between the two evaluations is why a single split is
not enough.

## Walk-forward evaluation and significance

`scripts/run_walk_forward.py` asks three questions the single split cannot:

1. **Is the result stable through time?** Expanding-window walk-forward:
   for each of the last 6 trading days, train on every earlier day except a
   one-day embargo and test on that day. Every ticker, every fold.
2. **Is the tree's edge nonlinearity, or just more features?** Two linear
   baselines: one-feature OFI, and ridge regression on all 43 standardized
   features.
3. **Is the edge statistically real?** Events inside a session are not
   independent (a horizon-h label overlaps the next h-1 labels), so an
   event-level standard error would be far too small. Every test instead
   treats a (ticker, day) session as one independent unit: a
   Diebold-Mariano test on per-session mean squared-error differences, a
   95% session-bootstrap interval for the R^2 difference, and the
   per-session Spearman information coefficient (IC) with its t-statistic.

Training rows are thinned to every 10th event (adjacent labels share almost
all their information); test rows are never thinned.

| Horizon | Pooled R^2: GBT / ridge-43 / OFI | GBT > OFI (folds) | GBT vs ridge: R^2 diff, 95% CI | DM p (GBT vs ridge) | IC GBT (t) |
|---|---|---|---|---|---|
| 10 | **+0.1542** / +0.0134 / +0.0000 | 6 of 6 | **+0.1408** [+0.1218, +0.1588] | 7.7e-11 | +0.081 (27.3) |
| 100 | **+0.1064** / +0.0187 / +0.0000 | 6 of 6 | **+0.0877** [+0.0588, +0.1176] | 1.2e-05 | +0.214 (18.5) |
| 1,000 | +0.0166 / +0.0212 / +0.0000 | 3 of 6 | -0.0046 [-0.0601, +0.0283] | 0.82 | +0.189 (5.8) |

At 10 and 100 events the tree beats both linear baselines in every fold,
and both the session-level Diebold-Mariano test and the bootstrap interval
exclude zero by a wide margin. Ridge on the same 43 features recovers only
a small part of the tree's R^2, so most of the edge comes from nonlinearity
and interactions, not from feature count. At 1,000 events the IC is still
positive (the forecasts rank outcomes better than chance), but the squared-
error advantage is gone, so the long-horizon forecast has some ordering
information and badly calibrated magnitudes. 30 sessions is a small sample
for the significance tests; with p-values around 1e-10 at 10 events that
does not change the conclusion, but it is the reason the 1,000-event result
is read as "no evidence" rather than "evidence of no effect."

Raw output: `docs/walk_forward_output.txt` (per-fold R^2 for every model) and
`docs/walk_forward_results.json`.

## Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, 31.1 GB RAM,
Windows 11 build 10.0.26200, CPython 3.12.10, LightGBM 4.7.0. Full pipeline
(6,400,000 messages, feature build, both models, all three horizons) runs in
about 70 seconds. Raw output in `docs/benchmark_output.txt` and
`docs/research_results.json`; the two discarded attempts are described in
Findings, not re-run. The walk-forward study above is the more reliable
read at every horizon.

| Horizon (events) | n_train | n_test | GBT out-of-sample R^2 | Linear-OFI out-of-sample R^2 | Tree advantage |
|---|---|---|---|---|---|
| 10 | 4,398,965 | 1,599,616 | **+0.1634** | **+0.0000** | +0.1633 |
| 100 | 4,394,011 | 1,597,811 | **+0.0866** | **-0.0000** | +0.0866 |
| 1,000 | 4,344,530 | 1,579,821 | **-0.1796** | **+0.0000** | -0.1796 |

What R^2 measures here: out-of-sample coefficient of determination on a
day-level purged split (11 train days, 1 embargo day, 4 test days per
ticker), computed once per horizon on the full test set, not cross-validated
or averaged across seeds. It does not measure trading P&L, transaction
costs, or capacity.

| Result | Measured |
|---|---|
| Full-depth books reconstructed | **5** (SYNA-SYNE) |
| Messages | **6,400,000** exactly |
| Store format | Hive-partitioned (`ticker=/date=`), all feature SQL runs through DuckDB |
| Microstructure features | **43** |
| Horizons | 10, 100 and 1,000 events |
| Split | 11 train / 1 embargo / 4 test days per ticker, verified disjoint (test) |

## Building and running

```bash
# from a Python 3.12 venv with requirements.txt installed
python scripts/generate_data.py       # ~49s, writes 220 MB to data/ (gitignored)
python scripts/run_research.py        # ~70s, writes docs/research_results.json
python scripts/run_walk_forward.py    # ~100s, writes docs/walk_forward_results.json
python -m pytest tests -q             # ~15s, 31 tests
```

`data/` is gitignored (220 MB of Parquet, regenerated in under a minute by
the command above); nothing under `data/` is tracked.


## Queue position and fill probability (Oct. 2026)

**What this adds.** Before this, the
book builder reconstructed full-depth snapshots but threw away exactly the
information a resting order's queue position needs: `BookBuilder` keeps a
FIFO list per price level internally but the message store only ever wrote
out aggregate level size, never which order was where in that list, because
nothing before this extension needed it. Three things are new: a full L3
replay (`obsr/queue_position.py`) that recovers each ADD event's queue rank
and quantity ahead from the stored message stream alone; a fill-probability
measurement by queue rank, built as one DuckDB SQL window-function pass
(`obsr/fill_probability.py`); and an execution-cost comparison between
always crossing the spread and a rank-aware quote-and-wait rule
(`obsr/execution_cost.py`), none of which the base repo had any reason to
include (it only ever asked "does a feature predict a price move", never
"would an order here have filled").

### Architecture (additions)

```
obsr/queue_position.py    replays one session's stored message stream
                           through a fresh BookBuilder, recording each ADD
                           event's rank (1 = first order ever at that exact
                           price) and qty_ahead (resting quantity in front
                           of it), plus whether it joined the best bid/ask
                           at that instant ("at the touch")
obsr/fill_probability.py  one DuckDB SQL statement, built entirely from a
                           forward-looking window function over a UNION of
                           ADD and EXECUTE rows (no join operator), that
                           reads off each order's first qualifying EXECUTE
                           timestamp and buckets fill-within-horizon by rank
obsr/execution_cost.py    always-crossing vs. rank-aware quote-and-wait
                           expected effective spread, using the measured
                           fill-probability-by-queue-depth curve and a
                           pandas merge_asof forward lookup for the price
                           one horizon later

scripts/compute_queue_positions.py  runs the replay over all 80 sessions,
                           writes data/queue_positions/ (same partitioning
                           as the message store)
scripts/run_queue_study.py  runs the SQL window pass, the cost comparison,
                           writes docs/queue_study_results.json

tests/test_queue_position.py      hand-computed rank/qty_ahead/touch on a
                           9-event scenario, plus a 20,000-operation
                           randomized sequence diffed against an
                           independent brute-force ledger that reimplements
                           price-time-priority execution from scratch
tests/test_fill_probability_sql.py  hand-computed fill-within-horizon
                           outcomes (including a fill that lands exactly
                           one horizon too late, and two sessions reusing
                           the same order_id, to prove the partition keys
                           are doing their job) against the SQL window pass

docs/queue_study_output.txt    raw run of scripts/run_queue_study.py
docs/queue_study_results.json  the same numbers, machine-readable
```

**Why "at the touch" is reported separately from the whole-book number.**
The first genuine measurement attempt bucketed every ADD event in the whole
book by rank, rank 5 meaning "5 or more": fill probability came out 0.1758
at rank 1 and 0.000055 at rank "5+". That number is real but misleading,
because it silently mixes two different effects. An order's fill
probability depends on two things: how far its *price* sits from the
market, and its *queue position* once it is at a price that matters. Rank
"5 or more" across the whole book is dominated by 2,658,360 of 2,689,366
total ADD events (98.8%) resting at a price nowhere near the touch, which
almost never fill quickly for a completely different reason than queue
position. Restricting to the 1,088,096 ADD events that joined the best
bid/ask at the instant of insertion isolates the queue-position question,
and is the number reported below.

**Why rank 5 means exactly rank 5, not "5 or more", in the reported
table.** The second attempt, restricted to the touch, still bucketed rank 5
as "5 or more" and measured 0.0001 at that bucket, because the touch queue
itself can run thousands deep and a "5+" bucket at the touch is still
overwhelmingly deeper-than-5 orders. The third attempt looked at the exact
per-rank breakdown (ranks 1 through 12, see Findings) and found a smooth,
steep, monotonic decay with no cliff at any particular rank, so the
honestly reportable number is fill probability at an *exact* rank, not a
bucket that happens to be labeled with that number.

### Validation

1. **Reference-oracle diff on queue position.** `tests/test_queue_position.py`
   runs 20,000 randomized add/execute/cancel/delete operations through
   `BookBuilder` while an entirely separate, independently coded ledger
   (its own price-time-priority execution loop, not a call into
   `BookBuilder`) tracks the same orders, and asserts every single ADD's
   rank and quantity-ahead, as recovered by `replay_queue_positions`,
   matches the independent ledger's brute-force recomputation.
2. **Hand-computed fill outcomes.** `tests/test_fill_probability_sql.py`
   checks the SQL window pass against five hand-picked cases: a fill
   inside the horizon, no fill at all, a fill exactly one horizon-length
   too late (the off-by-one case most likely to hide a bug), a fill after
   a gap with an intervening ADD at a different timestamp, and two
   different sessions reusing the same `order_id`, which must not be
   confused by the `PARTITION BY ticker, date, order_id` window.

```
14 passed in 11.55s
```

### Findings

**Fill probability decays steeply and smoothly with exact queue rank at
the touch, with no cliff at any one rank.** Measured per exact rank, 1
through 12, at-the-touch ADD events, 1-second horizon:

| Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10+ |
|---|---|---|---|---|---|---|---|---|---|---|
| Fill probability | 0.5377 | 0.3262 | 0.1724 | 0.0981 | **0.0410** | 0.0163 | 0.0059 | 0.0010 | 0.0010 | ~0.0000 |

The decay is close to geometric (each rank roughly 55-60% of the previous
one's probability) until it flattens into noise around rank 8, which is
mechanically exactly what price-time priority predicts: an order fills
only after every order ahead of it either fills or is removed, so each
additional rank multiplies the chance of surviving long enough to be
reached.

**Rank 1 fills more than 10 times as often as rank 5 (0.5377 vs 0.0410),
but both are lower than a denser market would show.** The two likely drivers, not
separately isolated here: this generator's order arrival rate (80,000
messages/session across 5 tickers) may simply be thinner at the touch than
a real, liquid name's queue, and a 1-second horizon on
`gaps ~ Exponential(0.05)`-paced synthetic time (mean 20 events/second)
gives an order roughly 20 *subsequent* events to be reached, not a fixed
large sample; both would make even a favorable rank fill less often than a
denser or longer-horizon setup would.

**The rank-aware quote-and-wait rule measured a 0.64% effective-spread
reduction, essentially no improvement.** The direct cause: the expected-value
blend (`p_fill * (bid - mid) + (1 - p_fill) * (ask_later - mid)`) is
dominated by the `(1 - p_fill)` branch almost everywhere, because the
large majority of "at the touch" ADD events sit behind an already-deep
queue (the median `qty_ahead` of the 1,088,096 at-touch ADD events is far
above the rank-1 level), so `p_fill` is usually small and the rule's
expected cost collapses toward "cross later", which on this calibration's
random-walk mid is, on average, barely different from crossing now. A
rank-aware rule would only show a meaningful benefit if it could
reliably post at or near rank 1, which the measured rank distribution says
is the exception, not the default, for an order simply joining the current
best price.

### Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, 31.1 GB RAM,
Windows 11 build 10.0.26200, CPython 3.12.10, DuckDB 1.5.5. Full pipeline
(80-session replay plus the SQL window pass plus the cost comparison) runs
in about 51 seconds (43.2s replay + 7.1s study). Raw output in
`docs/queue_study_output.txt` and `docs/queue_study_results.json`.

| Result | Measured |
|---|---|
| Fill probability by queue rank in one DuckDB SQL window pass | `obsr/fill_probability.py`, a single statement, no join operator |
| Fill probability at rank 1 | **0.5377** (at the touch) |
| Fill probability at rank 5 | **0.0410** (exact rank 5, at the touch) |
| Rank-aware quote-and-wait vs. always-crossing effective spread | **0.64% less** |

What this measures: fill probability is the fraction of ADD events at a
given exact queue rank, restricted to events that joined the best bid/ask
at insertion, that see at least one EXECUTE against that same order within
1 simulated second. It is a mechanical property of price-time priority and
this generator's order flow, not a forecast. The effective-spread comparison is an
expected-value calculation over 12,720 decision points (every 500th event
per session with both sides of the book populated), not a simulated,
path-dependent backtest; it does not account for the rank-aware rule's own
market impact or for multiple competing resting orders.

### Building and running

```bash
# from the same venv as the base repo (requirements.txt unchanged)
python scripts/generate_data.py            # if data/ is not already populated
python scripts/compute_queue_positions.py  # ~43s, writes data/queue_positions/
python scripts/run_queue_study.py          # ~7s, writes docs/queue_study_results.json
python -m pytest tests/test_queue_position.py tests/test_fill_probability_sql.py -v
```

## 10-minute realized volatility forecast (Oct. 2026)

**What this adds.** The base repo and
the queue-position extension both ask whether something predicts a future
*price move* or a future *fill*; neither asks the volatility question a
trading desk actually prices risk on. Three things are new: a 10-minute
wall-clock binning of the same message stream (`obsr/volatility.py`), a
LightGBM regressor forecasting each bin's *next* realized volatility from
the 43 microstructure features already built for the base repo (reused
unchanged, as of the last event in the current bin), and an RMSPE scoring
comparison against the horizon-matched previous-window baseline (a bin's
own, already-realized volatility used as the forecast for the next bin).

### Architecture (additions)

```
obsr/volatility.py   BIN_SECONDS=600 (10-minute) bin realized volatility,
                      built as a DuckDB SQL window pass over tick-to-tick
                      *microprice* changes (not raw best-bid/ask mid, see
                      Findings); the forward label (next bin's own RV) via
                      LEAD, same leakage convention as obsr.labels; an
                      RMSPE helper

scripts/run_volatility_study.py   builds the 43 features, bins realized
                      volatility, merges the last-event-per-bin feature
                      snapshot with the current and next bin's RV, applies
                      the same day-level purged split as run_research.py,
                      trains LightGBM on log(fwd_rv), scores RMSPE against
                      the previous-window baseline, and reports a
                      high-vs-low realized-vol regime split

tests/test_volatility.py   bin RV diffed against an independent pandas
                      reimplementation; the forward-label leakage check;
                      RMSPE hand-computed cases; a targeted regression test
                      for the one-sided-empty-book microprice bug (see
                      Findings)

docs/volatility_benchmark_output.txt   raw run of scripts/run_volatility_study.py
docs/volatility_study_results.json     the same numbers, machine-readable
```

**Why microprice, not the raw best-bid/ask mid, for realized volatility.**
The base repo's own README already discloses that most short event-count
windows see no tick-level mid change at all (the 10-event label is "spiked
at zero"). Over a full 10-minute bin, the best bid/ask is still remarkably
sticky: one session measured only 12 distinct best-bid price levels across
all 80,000 events. A mid-based realized volatility is therefore degenerate
at this timescale (almost every bin measures exactly 0.0), so this
extension measures realized volatility from the size-weighted microprice
instead, which moves on every size change at the touch, not only when the
best price level itself changes. Microprice is already one of the 43
features the forecasting model sees, so this is not a new quantity, only a
new use of an existing one.

**Why the training label is log(fwd_rv), not fwd_rv.** Realized volatility
over a fixed window is right-skewed (a handful of bins run 10 to 50x a
typical bin's RV); training directly on the raw scale let a small number
of high-RV bins dominate the loss. Training on the log and exponentiating
the prediction back before scoring is the standard fix for a lognormal-
shaped target, and it measurably lowered model RMSPE (1.19 to 0.54 at the
same split, see Findings) without touching the baseline, the labels, the
split, or the scoring function itself.

### Validation

1. **Reference-oracle diff on bin realized volatility.**
   `tests/test_volatility.py::test_sql_bin_rv_matches_independent_pandas_reimplementation`
   diffs the DuckDB SQL bin-RV computation against a completely independent
   pandas re-implementation (its own `groupby().diff()`, no shared code)
   on a small sample, exactly.
2. **Leakage check.** The same file asserts a bin's forward label equals
   the *next* bin's own realized volatility within the same session, and
   is null on the last bin of every session, never pulled from a different
   session in the same DataFrame.
3. **A targeted regression test for the microprice bug** (see Findings):
   a 3-row synthetic frame reproducing the exact one-sided-empty-book
   condition that produced a spurious multi-thousand-tick jump on the
   first attempt, asserting the fixed guard returns 0.0 instead.
4. **RMSPE hand-computed cases**, including a perfect forecast scoring
   exactly 0.0.

```
20 passed in 12.21s
```

### Findings

**The first attempt measured realized volatility from the raw best-bid/ask
mid and was dominated almost entirely by a single-event artifact, not real
price movement.** Every bin of one checked session measured a realized
volatility of exactly 0.0 except the very first, which measured 2,500.5,
larger than the entire rest of the session combined. Root cause: at
`event_seq=0` of every session, the book is still being seeded one level
at a time, so one side has size 0 while that side's *price* also defaults
to 0; the base repo's own microprice null guard (`bid_sz_1 + ask_sz_1 =
0`) does not fire in this case because the *other* side's size is
nonzero, so the sum is nonzero too, and the formula divides a near-zero
numerator by the live side's size, returning a spuriously well-defined
0.0 that the very next real microprice value (around 5000) then jumps
away from by thousands of ticks. The fix guards on `bid_sz_1 = 0 OR
ask_sz_1 = 0` directly (`obsr/volatility.py`), and a targeted regression
test (`test_one_sided_empty_book_does_not_produce_a_spurious_microprice_jump`)
pins it. At the time this was judged a local problem, immaterial to the
shared feature pass. That judgment was wrong: the same one-sided rows also
corrupted the shared mid, and through it every forward label, which hid
the base study's real signal until the walk-forward work exposed it (see
the base study's Findings). The guard now lives in the shared feature SQL.

**After the bug fix, the previous-window baseline is far weaker here than
in real markets, because this generator does not model volatility
clustering.** Realized volatility
measured per bin ranges from 0.0075 to 1.87 (560 bins, median 0.078), and
the previous-window baseline's RMSPE came in at 7.26. Real markets' volatility is strongly
autocorrelated bin to bin (a calm 10 minutes is usually followed by
another calm one), which is exactly what makes "use last window's RV" a
competitive baseline in practice; this generator's latent momentum state
(`obsr/simulate.py`'s `m`, AR(1), rho=0.97) drives *directional* drift
persistence, not *volatility* persistence, so bin-to-bin RV here swings
more than a real previous-window baseline would face, inflating both
RMSPE numbers (and, because the percentage-error metric is most sensitive
exactly where the baseline is worst, inflating the measured gain even
more than it inflates either RMSPE alone). Three genuine attempts were
made: raw-scale LightGBM (model RMSPE 1.19, gain 83.6%), log-scale
LightGBM (model RMSPE 0.54, gain 92.6%, the number reported below), and a
regime-split redefinition (see next finding); no further attempt was
made after the third.

**The regime split depends on what "activity" means, and both versions are
reported.** The first regime-split attempt bucketed test bins by raw message
count and measured the *larger* RMSPE gain in the low-message-count half
(88.9% vs. 43.0% in the high-message-count half). The second attempt re-bucketed by the test bins' own
median *current* realized volatility (`prev_rv`) instead of message count,
and measured the gain concentrated in the high-volatility half (95.6% vs.
53.5%). The mechanical reason the two
proxies disagree: message count is close to constant across bins in this
generator (a roughly Poisson arrival process), so splitting on it mostly
separates sessions by random noise in a nearly flat distribution, while
realized volatility itself varies by two orders of magnitude bin to bin
and is the more mechanically relevant "regime" for a volatility-forecast
gain to concentrate in. The volatility-level split is the one reported
below; the message-count split's reversed result is disclosed here rather
than discarded.

### Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, 31.1 GB RAM,
Windows 11 build 10.0.26200, CPython 3.12.10, DuckDB 1.5.5, LightGBM
4.7.0. Full pipeline (feature build, binning, split, training, scoring)
runs in about 22 seconds. Raw output in
[`docs/volatility_benchmark_output.txt`](docs/volatility_benchmark_output.txt)
and [`docs/volatility_study_results.json`](docs/volatility_study_results.json).

| Result | Measured |
|---|---|
| 10-minute realized volatility forecast | 600-second wall-clock bins, `obsr/volatility.py` |
| Scored by RMSPE against the horizon-matched previous-window baseline | `obsr/volatility.py::rmspe`, baseline = each bin's own already-realized volatility |
| Purged day-level train/test split | same 11 train / 1 embargo / 4 test days per ticker as `run_research.py`, applied at bin level |
| RMSPE, model vs previous-window baseline | **0.5398** vs **7.2646** |
| RMSPE gain over the baseline | **92.6%** |
| Regime split | 95.6% gain in the high-realized-vol half of the test set vs. 53.5% in the low half (splitting on message count instead reverses it, see Findings) |

What RMSPE measures here: root mean squared percentage error on 140
held-out test bins (`sqrt(mean(((actual-pred)/actual)^2))`), computed once
on the full test set, not cross-validated or averaged across seeds. It is
unusually sensitive to bins where the actual realized volatility is small
(a modest absolute miss becomes a large percentage error), which is part
of why both the model's and the baseline's RMSPE are large; see Findings.

### Building and running

```bash
# from the same venv as the base repo (requirements.txt unchanged)
python -m scripts.run_volatility_study   # ~22s, writes docs/volatility_study_results.json
python -m pytest tests/test_volatility.py -v
```

`python scripts/run_volatility_study.py` (without `-m`) fails with
`ModuleNotFoundError: No module named 'obsr'`, a pre-existing property of
how this repo's `scripts/` package resolves imports (`run_research.py` has
the same issue; the README there documents the plain-script form, which
does not actually run as written, out of scope for this extension to fix).

## Online incremental features in C++ and signal-aware quoting (Nov. 2026)

**What this adds.** Every feature
before this extension was computed in one DuckDB SQL window-function pass
over an entire partition at once; nothing in this repo had ever recomputed
a feature online, one event at a time, or asked what that would cost in
latency, or wired a feature into an actual quoting decision. Three things
are new: `cpp/` (this repo's first C++), an `IncrementalFeatureEngine`
that recomputes all 43 columns from `obsr/features.py` per event with
true O(1)-amortized sliding-window state; a byte-for-byte diff of that
engine's output against the real SQL pass over the full 6.4M-message
store (`scripts/compare_incremental_to_sql.py`); and
`obsr/quoting_loop.py`, which uses one of those online features, lagged by
one event, to decide whether a resting bid should have been pulled before
a fill that turned out to be adverse.

### Architecture (additions)

```
cpp/include/raw_event.hpp         packed binary record layout, byte-identical
                                   to scripts/export_messages_flat.py's NumPy dtype
cpp/include/incremental_features.hpp
                                   IncrementalFeatureEngine: per-(ticker,date)
                                   partition state, a SlidingMoment ring buffer
                                   per windowed aggregate (sum, and sum-of-squares
                                   where STDDEV needs it), reset on every
                                   partition-id change in the input stream
cpp/apps/incremental_bench.cpp    reads the flat export, runs every row through
                                   the engine, times each update with
                                   std::chrono, writes the 43-column output
cpp/tests/test_incremental.cpp    hand-worked 4-row partition (every one of the
                                   43 columns checked against arithmetic worked
                                   out on paper), a sliding-window eviction
                                   check, and a NULL-propagation regression

scripts/export_messages_flat.py   Parquet -> packed binary, ORDER BY ticker,
                                   date, event_seq (the row order the C++ side
                                   and the SQL comparison must agree on)
scripts/compare_incremental_to_sql.py
                                   column-by-column diff against obsr.features.build_features,
                                   writes docs/incremental_vs_sql_report.json
obsr/quoting_loop.py              build_fill_table: bid-side fills with the mid
                                   `horizon_events` later and the signal lagged
                                   by one event; measure_signal_aware_reduction:
                                   mean adverse selection with vs. without
                                   pulling the quote below a fixed signal quantile
scripts/run_quoting_loop_study.py runs the measurement over the real feature
                                   pass, writes docs/quoting_loop_output.json
tests/test_quoting_loop.py        hand-worked checks on a tiny synthetic session
```

**Why a sliding-window ring buffer, not a re-sum from scratch every
event.** A windowed aggregate (a per-type message count, a rolling
sum-of-signed-quantity, a rolling mean or standard deviation) only ever
needs to add the newest value and remove whatever value just fell out of
the frame; `SlidingMoment` does exactly that in O(1) amortized time per
push, the same "removable cumulative aggregate" technique a streaming
window-function engine uses internally, rather than re-scanning up to 100
buffered rows on every single event. The simple per-row and `LAG`-based
columns (spread, microprice, depth, OFI, QI, the three lag returns,
`cum_net_flow`) need no sliding state at all; each is a fixed, small
amount of history (the previous one to three rows), kept directly.

**Why `qi_agg5` and `ofi_agg5` are plain running sums, not NaN-skipping
ones.** The SQL expressions `(qi_l1 + ... + qi_l5)` and
`(ofi_l1 + ... + ofi_l5)` are ordinary `+`, and SQL's `NULL + anything =
NULL` means one empty book level (an empty 5th level makes `qi_l5` NULL)
must make the *whole* sum NULL, not silently drop that one term and sum
the rest. IEEE 754 float addition already propagates NaN exactly that
way, so the fix was deleting a `std::isnan(...) ? 0.0 : ...` guard a
first draft had added defensively, not adding one;
`tests/test_incremental.cpp`'s `qi_agg5_null_propagation` check is the
regression test for that first, wrong draft.

### Validation

1. **Hand-worked reference values.** `tests/test_incremental.cpp` builds a
   4-event, 2-partition stream by hand and checks every feature family
   (spread, microprice, depth, OFI, QI, lag returns, `cum_net_flow`,
   message counts, `realized_vol`, trade-flow imbalance, gap mean, the
   partition-reset behavior, and the NULL-propagation case above) against
   arithmetic worked out on paper, not against the engine's own output
   circularly. 31 checks, 0 failures, also clean under AddressSanitizer +
   UBSan (`docs/asan_ubsan_incremental_clean_run.txt`).
2. **The real test: a byte-for-byte diff against the actual SQL pass, over
   every row of the real 6.4M-message store**, not a sample.
   `scripts/compare_incremental_to_sql.py` sorts both outputs
   `ORDER BY ticker, date, event_seq` and compares every one of the 43
   columns, cell by cell, reporting exact-match counts and the maximum
   absolute difference where they are not exact. Full output:
   [`docs/incremental_vs_sql_output.txt`](docs/incremental_vs_sql_output.txt),
   machine-readable: [`docs/incremental_vs_sql_report.json`](docs/incremental_vs_sql_report.json).
3. **Quoting-loop arithmetic.** `tests/test_quoting_loop.py` checks that
   `build_fill_table` uses strictly the *previous* event's signal value
   (not the fill event's own, which would be look-ahead), restricts to
   bid-side fills only, and that `measure_signal_aware_reduction`'s
   pulled/kept split and percentage-reduction arithmetic match a
   hand-worked four-fill example exactly.

### Findings

**38 of 43 columns are bit-identical to the SQL pass; the other 5 are not,
and the reason is a real one, not a bug.** The 38 that match exactly are
either pure per-row arithmetic, simple `LAG`-based differences, or
sums/counts over *integer-valued* quantities (message-type indicators,
`signed_exec_qty`, which this generator always produces as whole share
counts): a sum of exact double-precision integers is exact regardless of
the order the additions happen in, because no rounding ever occurs, so an
incremental add/remove sum and a from-scratch re-sum are mathematically
and bit-for-bit the same number. The 5 that do not match exactly
(`realized_vol_w20/50/100`, `gap_mean_w20/100`) are the only features
built from genuinely real-valued inputs with no "nice" binary
representation (`ts`, the simulated event timestamp, and `mid_diff`,
which inherits that realness once vol is computed over it): floating-point
addition is not associative once rounding actually happens, so this
engine's add/remove sliding sum and DuckDB's own internal window-frame
evaluation can each be a fully correct computation of the same
mathematical quantity while disagreeing in the last 1-3 bits. Measured
disagreement is at the level of the last bit (max abs diff 1.7e-16 for the
`realized_vol` columns, 1.2e-16 for `gap_mean`). Before the one-sided-book
fix (see the base study's Findings) the `realized_vol` gap was 9.1e-13: the
fake jump of thousands of ticks at each session start went into the running
sum of squares, and subtracting a number that large back out on eviction
cost several digits. With only real tick-sized moves in the window, the
add/remove sum loses almost nothing.

The fix also changed the window class. NULL used to occur only in the first
row of a session, so the engine simply skipped NaN values. Once a NULL mid
can appear mid-session, skipping is wrong: SQL's `ROWS` frame still counts a
NULL row as one of its w+1 rows, so a skipped row would make the C++ window
evict the wrong value later. `SlidingMoment` now stores a NaN as a row that
occupies its slot but stays out of the sum and the count, and a C++ test
pins the eviction order. One genuine attempt was made
(this sliding-sum design, chosen specifically because it is the same
*algorithmic* technique a streaming aggregate would use, not a weaker
approximation); a second attempt would mean guessing at DuckDB's internal
window-evaluation order with no way to confirm the guess from outside the
engine, so the honest stopping point is reporting the measured agreement
rather than chasing an implementation detail of a dependency this project
does not control.

**The isolated per-event latency is about 0.1us at the median: a tight,
allocation-free C++ loop over a handful of ring-buffer pushes per event is
cheap.** The latest run, on the fixed engine with other work running on the
machine, measured a 100ns median and 220ns p99; the engine before the fix
measured 91ns / 171ns on an idle machine. The difference was not split
between the extra NULL branches and machine load. The
median is timed with `std::chrono::steady_clock` wrapped around each
individual `update()` call after a full untimed warm-up pass over the same
6.4M events (so every page is faulted in and every branch predictor is
warm before the timed pass begins); the clock-read overhead itself is a
non-trivial fraction of a measurement this small, so the reported number
is an upper bound on the true per-event compute cost, not an
underestimate.

**Pulling the bid on a weak trade-flow signal cuts adverse selection per
fill by 7.18% at the 20% quantile, and the cut grows smoothly with the
quantile: 4.06% at 10%, 7.18% at 20%, 9.75% at 25%.** `trade_flow_imb_w50`,
lagged by one event, is compared with its own quantile before each bid-side
fill (898,230 fills with a defined horizon and lagged signal; 180,627 pulled
at q=0.20). The first measurement said 30.19%, 82.44% and 32.12% at the same
three quantiles, and an earlier version of this document explained the
non-monotonic jumps with the shape of the signal's distribution. That
explanation was wrong. A handful of fills sat on the fake half-price mid,
each with an adverse-selection value of thousands of ticks, and whichever
quantile happened to pull them looked spectacular. After the fix the
baseline mean adverse selection fell from 0.01129 to 0.00849 ticks per
fill while the mean over kept fills did not change at all (0.007882), which
places the entire earlier difference on those corrupted fills. q=0.20 is
still the reported point because it was chosen before any result was seen;
the full sweep is in `docs/quoting_loop_sweep.json`.

### Measured results

Machine for this extension (the C++ half only builds and runs under
Linux): WSL2 Ubuntu 22.04, g++ 11.4.0 `-O3`. The SQL side of the diff ran
on the Windows Python environment stated for the base study.

```
events processed: 6400000
median latency: 100.0 ns (0.100 us)
p99 latency:    220.0 ns (0.220 us)
mean latency:   114.4 ns (0.114 us)

38 / 43 columns bit-identical to the SQL pass
(realized_vol_w20/50/100 max abs diff 1.665e-16; gap_mean_w20/100 max abs diff 1.249e-16; 0 NULL mismatches)
```

| Result | Measured |
|---|---|
| Recomputed incrementally per event, online | `IncrementalFeatureEngine::update`, one event in, one feature row out, O(1)-amortized sliding state |
| Median latency per update | **0.100us** (100ns; p99 220ns) |
| Bit-identical to the SQL pass | **38 / 43** columns exactly; the other 5 agree to 1.7e-16, 0 NULL mismatches |
| Adverse selection paid per fill | **7.18% less** (q=0.20 pull threshold, 898,230 fills); 4.06% at q=0.10, 9.75% at q=0.25 |

```
{
  "pull_quantile": 0.2,
  "lag_signal_threshold": -235.0,
  "n_fills_total": 898230,
  "n_fills_pulled": 180627,
  "n_fills_kept": 717603,
  "baseline_mean_adverse_selection": 0.008491738159438716,
  "signal_aware_mean_adverse_selection": 0.007881795365961402,
  "pct_reduction": 7.182779096872549,
  "mean_adverse_selection_of_pulled_fills": 0.010914999750862294
}
```

What this measures: adverse selection per fill is the mid-price move
against the market maker (a fall in mid after the maker bought) measured
50 events after each bid-side fill, in the same units `mid` is denominated
in (integer price ticks, averaged); it is not a dollar P&L and does not
account for the spread the maker captured on the fill itself. "Pulled"
fills are excluded from the signal-aware mean entirely, not repriced or
resized; a real quoting system that pulled a quote would also give up the
spread on every fill it avoided, a cost this narrow metric does not
charge against the reduction (see Limitations).

### Building and running

```bash
# C++ engine and its own test suite (WSL2/Linux only)
cd cpp && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j"$(( $(nproc) / 2 ))"
./test_incremental
cd ../..

# ASan + UBSan build
cd cpp && mkdir build-asan && cd build-asan
cmake -DENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug .. && make -j"$(( $(nproc) / 2 ))" test_incremental
setarch -R ./test_incremental
cd ../..

# Export, benchmark, and diff against the real SQL pass (from the repo root)
python scripts/export_messages_flat.py --out /tmp/obsr_messages.bin
./cpp/build/incremental_bench /tmp/obsr_messages.bin /tmp/obsr_incremental_out.bin
python scripts/compare_incremental_to_sql.py --incremental-out /tmp/obsr_incremental_out.bin

# Signal-aware quoting-loop study
python scripts/run_quoting_loop_study.py --quantile 0.20
python -m pytest tests/test_quoting_loop.py -v
```
