"""
Tests for cross-validation with groups (several rows per subject, e.g. longitudinal data
in long format): neureval.best_nclust_cv_confounds and neureval.param_selection_confounds.

Run from the repository root:  pytest tests/test_groups.py -v
"""
import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC

import neureval.best_nclust_cv_confounds as bn
from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds, _make_splits
from neureval.param_selection_confounds import ParamSelectionConfounds

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)


def make_longitudinal(seed, n_subjects=150, n_windows=4, n_states=0, subject_sd=3.0, noise=0.5,
                      missing=0.0):
    """
    Long-format data: one row per subject and time window.

    Each row = subject profile (constant over time, sd `subject_sd`) + state centre (if
    n_states > 0, a random state for each row) + noise. With n_states=0 there are no clusters:
    rows of the same subject are similar only because they belong to the same subject.
    A fraction `missing` of the rows is removed (subjects with missing windows).
    The window enters the confound correction as dummy covariates.
    Returns data, modalities, covariates, subject IDs, window and true state of each row.
    """
    rng = np.random.default_rng(seed)
    p = 6
    subj = np.repeat(np.arange(n_subjects), n_windows)
    window = np.tile(np.arange(n_windows), n_subjects)
    X = rng.normal(0, subject_sd, (n_subjects, p))[subj] + rng.normal(0, noise, (len(subj), p))
    state = np.zeros(len(subj), dtype=int)
    if n_states > 0:
        state = rng.integers(0, n_states, len(subj))
        centers = np.zeros((n_states, p))
        for k in range(n_states):
            centers[k, k] = 8.0
        X += centers[state]
    X += 0.5 * window[:, None]                     # age effect, removed by the window covariates
    keep = rng.random(len(subj)) >= missing
    X, subj, window, state = X[keep], subj[keep], window[keep], state[keep]
    data = pd.DataFrame(X, columns=[f'f{j}' for j in range(p)])
    cov = pd.DataFrame({f'w{w}': (window == w).astype(float) for w in range(1, n_windows)})
    return data, {'m1': data.iloc[:, :3], 'm2': data.iloc[:, 3:]}, {'m1': cov, 'm2': cov}, subj, window, state


def _findbest(**kw):
    return FindBestClustCVConfounds(KMeans(n_init=10, random_state=0), SVC(random_state=42),
                                    preprocessing=None, nrand=10, **kw)


def _assert_groups_not_split(splits, groups, n):
    for tr, val in splits:
        assert len(np.intersect1d(groups[tr], groups[val])) == 0
        assert len(tr) + len(val) == n and len(np.intersect1d(tr, val)) == 0


class _RecordSplits:
    """Wraps _make_splits to record the splits actually used by the cross-validation."""
    def __init__(self, monkeypatch):
        self.calls = []
        original = bn._make_splits

        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            self.calls.append(out)
            return out
        monkeypatch.setattr(bn, '_make_splits', wrapper)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('nfold', [2, 3, 5])
def test_folds_never_split_groups(nfold):
    _, _, _, subj, _, _ = make_longitudinal(0, n_subjects=60, missing=0.2)
    n = len(subj)
    splits = _make_splits(n, nfold, iter_cv=3, groups=subj)
    assert len(splits) == 3 * nfold
    _assert_groups_not_split(splits, subj, n)
    for r in range(3):
        # in each repetition every row is in exactly one validation fold
        val = np.concatenate([v for _, v in splits[r * nfold:(r + 1) * nfold]])
        assert sorted(val) == list(range(n))
        # folds of similar size (subjects with missing windows have fewer rows)
        sizes = [len(v) for _, v in splits[r * nfold:(r + 1) * nfold]]
        assert max(sizes) - min(sizes) <= 4
    # the repetitions are different splits
    assert not all(np.array_equal(splits[i][1], splits[nfold + i][1]) for i in range(nfold))
    # and the splits are reproducible
    again = _make_splits(n, nfold, iter_cv=3, groups=subj)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(splits, again))


def test_stratified_group_folds():
    _, _, _, subj, _, _ = make_longitudinal(1, n_subjects=120)
    strata = (subj % 3 == 0).astype(int)          # constant within subject, 1/3 of subjects
    splits = _make_splits(len(subj), 3, iter_cv=2, strat_vect=strata, groups=subj)
    _assert_groups_not_split(splits, subj, len(subj))
    for _, val in splits:
        assert abs(strata[val].mean() - strata.mean()) < 0.1


def test_without_groups_splits_unchanged():
    """Without groups the splits are the same as before (rows split independently)."""
    old = list(RepeatedKFold(n_splits=3, n_repeats=2, random_state=42).split(np.zeros(50)))
    new = _make_splits(50, 3, iter_cv=2)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(old, new))


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
def test_groups_remove_leakage_between_rows_of_same_subject():
    """
    No clusters, but rows of the same subject are similar. Splitting rows of the same
    subject between training and validation makes the solution look stable; keeping the
    subjects together gives a stability close to chance.
    """
    data, mods, covs, subj, _, _ = make_longitudinal(0)
    k_range = [2, 3, 4]
    m_rows, _, _ = _findbest(nclust_range=k_range).best_nclust_confounds(data, mods, covs, iter_cv=2)
    m_grp, _, _ = _findbest(nclust_range=k_range).best_nclust_confounds(data, mods, covs, iter_cv=2,
                                                                       groups=subj)
    best_rows = min(v[0] for v in m_rows['val'].values())
    best_grp = min(v[0] for v in m_grp['val'].values())
    # seed 0: about 0.45 splitting rows, 0.78 keeping subjects together
    # (seeds 0-4: differences between 0.17 and 0.38)
    assert best_grp > 0.6
    assert best_grp - best_rows > 0.15


def test_groups_find_true_states():
    """With real states (that change over time within a subject) group CV finds them."""
    data, mods, covs, subj, _, state = make_longitudinal(2, n_subjects=100, n_states=3, subject_sd=1.0,
                                                         missing=0.1)
    _, k, lab = _findbest(nclust_range=[2, 3, 4, 5]).best_nclust_confounds(data, mods, covs, iter_cv=2,
                                                                         groups=subj)
    assert k == 3
    assert len(lab) == len(data)
    assert adjusted_rand_score(state, lab) > 0.9


def test_findbest_uses_group_splits(monkeypatch):
    rec = _RecordSplits(monkeypatch)
    data, mods, covs, subj, _, _ = make_longitudinal(3, n_subjects=40)
    _findbest(nclust_range=[2, 3], nfold=3).best_nclust_confounds(data, mods, covs, iter_cv=2, groups=subj)
    assert len(rec.calls) == 1
    _assert_groups_not_split(rec.calls[0], subj, len(subj))


@pytest.mark.parametrize('with_preprocessing', [False, True])
def test_param_selection_uses_group_splits(monkeypatch, with_preprocessing):
    from sklearn.decomposition import PCA
    rec = _RecordSplits(monkeypatch)
    data, mods, covs, subj, _, _ = make_longitudinal(4, n_subjects=40)
    params = {'c': {'n_init': [5]}, 's': {'C': [1.0, 10.0]}}
    preproc = None
    if with_preprocessing:
        params['preprocessing'] = {'n_components': [2, 3]}
        preproc = PCA(random_state=0)
    ps = ParamSelectionConfounds(params, cv=2, c=KMeans(random_state=0), s=SVC(random_state=42),
                                 preprocessing=preproc, nrand=5, iter_cv=2, clust_range=[2, 3])
    ps.fit(data, mods, covs, groups=subj)
    assert len(rec.calls) == 1
    _assert_groups_not_split(rec.calls[0], subj, len(subj))
    assert len(ps.best_param_) >= 1


def test_groups_with_site_harmonization(monkeypatch):
    """Site (constant within subject) and groups together: subjects kept together, sites stratified."""
    rec = _RecordSplits(monkeypatch)
    data, mods, covs, subj, _, state = make_longitudinal(5, n_subjects=90, n_states=3, subject_sd=1.0)
    site = np.array(['A', 'B', 'C'])[subj % 3]
    shift = {'A': 0.0, 'B': 3.0, 'C': -3.0}
    data = data + np.array([shift[s] for s in site])[:, None]
    mods = {'m1': data.iloc[:, :3], 'm2': data.iloc[:, 3:]}
    _, k, lab = _findbest(nclust_range=[2, 3, 4]).best_nclust_confounds(data, mods, covs, iter_cv=2,
                                                                      site=site, groups=subj)
    splits = rec.calls[0]
    _assert_groups_not_split(splits, subj, len(subj))
    for tr, _ in splits:
        assert set(site[tr]) == {'A', 'B', 'C'}
    assert k == 3 and adjusted_rand_score(state, lab) > 0.9


# ---------------------------------------------------------------------------
# External validation
# ---------------------------------------------------------------------------
def test_evaluate_rejects_subject_in_both_sets():
    data, mods, covs, subj, _, _ = make_longitudinal(6, n_subjects=60, n_states=3, subject_sd=1.0)
    rows = np.arange(len(data))
    tr, val = rows[:150], rows[150:]                   # row 150 splits subject 37 between the sets
    with pytest.raises(ValueError, match="1 group\\(s\\) have rows in both the training and the validation"):
        _findbest().evaluate_confounds(data, mods, covs, tr, val, nclust=3, groups=subj)
    # subjects split correctly: same result as without groups (groups only checks the sets)
    tr, val = rows[subj < 40], rows[subj >= 40]
    out = _findbest().evaluate_confounds(data, mods, covs, tr, val, nclust=3, groups=subj)
    ref = _findbest().evaluate_confounds(data, mods, covs, tr, val, nclust=3)
    assert out.test_stability == ref.test_stability
    np.testing.assert_array_equal(out.test_classlab, ref.test_classlab)


def test_evaluate_rejects_subject_in_training_and_calibration():
    data, mods, covs, subj, _, _ = make_longitudinal(7, n_subjects=60)
    site = np.where(subj < 40, 'A', 'B')
    site[subj < 20] = 'C'
    rows = np.arange(len(data))
    tr = rows[subj < 40]
    cal = rows[(subj >= 40) & (subj < 50)]
    val = rows[subj >= 50]
    groups = subj.copy()
    groups[cal[0]] = 0                                  # a calibration row of a training subject
    with pytest.raises(ValueError, match="training and the calibration set"):
        _findbest().evaluate_confounds(data, mods, covs, tr, val, nclust=2, site=site,
                                       calibration_idx=cal, groups=groups)


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------
def test_input_checks():
    data, mods, covs, subj, _, _ = make_longitudinal(8, n_subjects=30)
    fb = _findbest(nclust_range=[2, 3])
    with pytest.raises(ValueError, match="one value per row"):
        fb.best_nclust_confounds(data, mods, covs, groups=subj[:-1])
    bad = subj.astype(float)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        fb.best_nclust_confounds(data, mods, covs, groups=bad)
    with pytest.raises(ValueError, match="fewer than the 2 cross-validation folds"):
        fb.best_nclust_confounds(data, mods, covs, groups=np.zeros(len(data)))
    with pytest.raises(ValueError, match="one value per row"):
        fb.evaluate_confounds(data, mods, covs, np.arange(60), np.arange(60, 120), nclust=2,
                              groups=subj[:10])
