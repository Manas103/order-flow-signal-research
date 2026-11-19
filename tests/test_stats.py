"""The evaluation statistics behave correctly on data whose answer is known."""
import numpy as np
import pytest

from obsr.stats import (
    diebold_mariano_sessions, hit_rate, information_coefficient, r2_oos,
    session_bootstrap_r2_diff, walk_forward_folds,
)


def _sessions(n_sessions=30, per=400, seed=0):
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(n_sessions), per)
    signal = rng.normal(size=s.size)
    y = 0.5 * signal + rng.normal(size=s.size)
    return rng, s, signal, y


def test_r2_oos_reference_mean():
    y = np.array([1.0, 2.0, 3.0])
    assert r2_oos(y, y) == 1.0
    assert r2_oos(y, np.full(3, 2.0)) == 0.0
    # forecasting the training mean itself scores exactly zero against that reference
    assert r2_oos(y, np.full(3, 0.0), mean_ref=0.0) == 0.0


def test_dm_identical_forecasts_is_zero():
    _, s, signal, y = _sessions()
    r = diebold_mariano_sessions(y, signal, signal, s)
    assert r.mean_loss_diff == 0 and r.t_stat == 0 and r.p_value == 1.0 and r.n_sessions == 30


def test_dm_detects_a_better_model_and_its_direction():
    _, s, signal, y = _sessions()
    good, useless = 0.5 * signal, np.zeros_like(y)
    r = diebold_mariano_sessions(y, good, useless, s)
    assert r.mean_loss_diff > 0 and r.t_stat > 5 and r.p_value < 1e-6
    flipped = diebold_mariano_sessions(y, useless, good, s)
    assert flipped.t_stat == pytest.approx(-r.t_stat)


def test_dm_is_not_fooled_by_noise():
    rng, s, _, y = _sessions(seed=3)
    a, b = rng.normal(scale=0.01, size=y.size), rng.normal(scale=0.01, size=y.size)
    assert diebold_mariano_sessions(y, a, b, s).p_value > 0.01


def test_bootstrap_interval_brackets_a_real_difference_and_excludes_zero():
    _, s, signal, y = _sessions()
    point, lo, hi = session_bootstrap_r2_diff(y, 0.5 * signal, np.zeros_like(y), s, mean_ref=0.0, n_boot=500)
    assert lo < point < hi and lo > 0
    assert point == pytest.approx(r2_oos(y, 0.5 * signal, 0.0) - r2_oos(y, np.zeros_like(y), 0.0))


def test_information_coefficient_and_hit_rate():
    _, s, signal, y = _sessions()
    ic = information_coefficient(y, signal, s)
    assert 0.3 < ic.mean_ic < 0.5 and ic.p_value < 1e-6 and ic.n_sessions == 30
    assert hit_rate(y, signal) > 0.6
    assert hit_rate(y, -signal) < 0.4


def test_walk_forward_folds_expand_and_embargo():
    days = [f"d{i:02d}" for i in range(16)]
    folds = list(walk_forward_folds(days, min_train=9, embargo=1))
    assert [f[2] for f in folds] == days[10:]
    for train, emb, test in folds:
        assert emb == [days[days.index(test) - 1]]
        assert max(train) < min(emb) < test
        assert not set(train) & {test, *emb}
    assert [len(f[0]) for f in folds] == list(range(9, 15))
