"""Tests for the pipeline. These never compute a holdout metric; they check features, metrics and window invariants."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import catalogs as C
from src.stats import auc, recall_at_frac


# ---------- features ----------
def test_hardness_ratio_masks_upper_limits_and_nonpositive():
    nufnu = np.array([[1.0, 3.0, 3.0, 0.0]])
    is_ul = np.array([[False, False, True, False]])
    hr = C.hardness_ratios(nufnu, is_ul)
    assert hr[0, 0] == pytest.approx(0.5)          # (3-1)/(3+1)
    assert np.isnan(hr[0, 1]) and np.isnan(hr[0, 2])   # UL band and nuFnu<=0 band


def test_log_transforms():
    assert C.ln_sig([0.0])[0] == pytest.approx(np.log(1e-3))
    assert C.ln_sig([-1.0])[0] == pytest.approx(np.log(1e-3))
    out = C.ln_flux([0.0, -2.0, np.e])
    assert np.isnan(out[0]) and np.isnan(out[1]) and out[2] == pytest.approx(1.0)


# ---------- metrics ----------
def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 5, 40).astype(float)
    neg = rng.integers(0, 5, 300).astype(float)
    y = np.r_[np.ones(40), np.zeros(300)]
    assert auc(pos, neg) == pytest.approx(roc_auc_score(y, np.r_[pos, neg]))


def test_recall_at_frac_tie_breaking():
    scores = np.array([3, 2, 2, 2, 1, 0, 0, 0, 0, 0], float)
    pos = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1], bool)
    # top 20% = 2 slots: item 0 (score 3) plus 1 of the 3 tied at 2, so the expected hit is 1/3 of one positive
    assert recall_at_frac(scores, pos, 0.2) == pytest.approx((1 / 3) / 2)


# ---------- catalog / window invariants ----------
@pytest.fixture(scope='module')
def windows():
    from src.windows import build_w1, build_w2
    return build_w1(), build_w2()


def test_upper_limit_flag_matches_sqrt_ts():
    t = C.load('DR3')
    is_ul = np.isnan(np.asarray(t['Unc_Flux_Band'], float)[..., 0])
    ts = np.asarray(t['Sqrt_TS_Band'], float)
    assert np.all(ts[is_ul] <= 1.0 + 1e-6) and np.all(ts[~is_ul] >= 1.0 - 1e-6)


def test_window_sizes_match_census(windows):
    w1, w2 = windows
    assert (w1.meta['n_pool'], int(w1.pos.sum())) == (1010, 55)
    assert (w2.meta['n_pool'], int(w2.pos.sum())) == (2291, 23)


def test_no_lookahead_pool_disjoint_from_training(windows):
    # train ∪ pool ∪ (excluded-but-not-pool) must partition the training catalog exactly: no source is in both.
    for w, key in zip(windows, ('3FGL', 'DR3')):
        cls = C.classes(C.load(key))
        pool_cls = {''} if key == '3FGL' else {'', 'unk'}
        n_excl_not_pool = int(np.isin(cls, list({'glc', '', 'unk'} - pool_cls)).sum())
        assert w.meta['n_train'] + w.meta['n_pool'] + n_excl_not_pool == len(cls)
        assert w.train_X.shape[1] == w.pool_X.shape[1]
        assert set(np.unique(w.train_y)) == {0, 1}
    # Pool sources are unassociated in the training release, so none can be a training row.
    t3 = C.load('3FGL')
    cls = C.classes(t3)
    assert set(cls[np.isin(C.names(t3), windows[0].pool_names)]) == {''}


def test_training_labels_exclude_ambiguous(windows):
    t = C.load('DR3')
    cls = C.classes(t)
    n_expected = int((~np.isin(cls, ['glc', '', 'unk'])).sum())
    assert windows[1].meta['n_train'] == n_expected


def test_windows_disjoint_positives(windows):
    w1, w2 = windows
    tdr3 = C.load('DR3')
    xw = dict(zip(C._str(tdr3['ASSOC_FGL']), C.names(tdr3)))
    w1_dr3_names = {xw[n] for n in w1.pool_names[w1.pos] if n in xw}
    assert not (w1_dr3_names & set(w2.pool_names[w2.pos]))


def test_features_are_finite_or_nan():
    for key, fn in (('3FGL', C.features_3fgl), ('DR3', C.features_4fgl), ('DR4', C.features_4fgl)):
        X, _ = fn(C.load(key))
        assert not np.isinf(X.values).any(), key
