"""
Tests for the external validation (FindBestClustCVConfounds.evaluate_confounds)
and for error handling in the cross-validation.

Run from the repository root:  pytest tests/test_evaluate.py -v
"""
import logging
import os
import sys
import warnings

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_cases import make_synthetic_data  # noqa: E402

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

TR, VAL = np.arange(80), np.arange(80, 120)


def _fb(c=None):
    c = c if c is not None else GaussianMixture(random_state=42)
    return FindBestClustCVConfounds(c, SVC(random_state=42), preprocessing=None, nrand=10)


# ---------------------------------------------------------------------------
# New outputs
# ---------------------------------------------------------------------------
def test_new_outputs_well_separated_clusters():
    """Clear clusters: perfect replication, normalized stability 0."""
    data, mods, covs = make_synthetic_data(cluster_std=1.0)
    out = _fb().evaluate_confounds(data, mods, covs, TR, VAL, nclust=3)
    assert out.test_acc == 1.0
    assert out.test_stability == 0.0
    assert len(out.test_classlab) == len(VAL)
    assert set(out.test_classlab) <= set(out.train_cllab)


def test_normalized_stability_detects_noise():
    """
    No real clusters: the raw accuracy can look acceptable, but the normalized
    stability is close to or above 1 (no better than chance).
    """
    data, mods, covs = make_synthetic_data(cluster_std=60.0)
    out = _fb().evaluate_confounds(data, mods, covs, TR, VAL, nclust=3)
    assert out.test_stability > 0.8


def test_boolean_mask_same_as_positions():
    data, mods, covs = make_synthetic_data(cluster_std=10.0)
    mask = np.zeros(len(data), dtype=bool)
    mask[TR] = True
    a = _fb().evaluate_confounds(data, mods, covs, TR, VAL, nclust=3)
    b = _fb().evaluate_confounds(data, mods, covs, mask, ~mask, nclust=3)
    assert np.array_equal(a.test_cllab, b.test_cllab)
    assert a.test_acc == b.test_acc and a.test_stability == b.test_stability


def test_user_estimator_not_modified():
    data, mods, covs = make_synthetic_data()
    km = KMeans(n_clusters=8, n_init=10, random_state=42)
    _fb(km).evaluate_confounds(data, mods, covs, TR, VAL, nclust=3)
    assert km.n_clusters == 8


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------
def test_overlapping_sets_raise():
    data, mods, covs = make_synthetic_data()
    with pytest.raises(ValueError, match='both the training and the validation set'):
        _fb().evaluate_confounds(data, mods, covs, np.arange(0, 60), np.arange(59, 120), nclust=3)


@pytest.mark.parametrize('tr_idx, message', [
    (np.arange(0, 130), 'outside the range'),
    ([0, 1, 1, 2], 'duplicated'),
    ([], 'empty'),
    (np.ones(50, dtype=bool), 'boolean mask of length'),
    ([0.5, 1.5], 'integer row positions'),
])
def test_invalid_indices_raise(tr_idx, message):
    data, mods, covs = make_synthetic_data()
    with pytest.raises(ValueError, match=message):
        _fb().evaluate_confounds(data, mods, covs, tr_idx, VAL, nclust=3)


def test_missing_nclust_raises():
    data, mods, covs = make_synthetic_data()
    with pytest.raises(ValueError, match='nclust is required'):
        _fb(KMeans(n_init=10, random_state=42)).evaluate_confounds(data, mods, covs, TR, VAL)


def test_tr_lab_wrong_length_raises():
    data, mods, covs = make_synthetic_data()
    with pytest.raises(ValueError, match='tr_lab has'):
        _fb().evaluate_confounds(data, mods, covs, TR, VAL, nclust=3, tr_lab=[0, 1, 2])


def test_modalities_with_different_rows_raise():
    data, mods, covs = make_synthetic_data()
    mods = dict(mods, mod_02=mods['mod_02'].iloc[:100])
    with pytest.raises(ValueError, match='same number of subjects'):
        _fb().evaluate_confounds(data, mods, covs, TR, VAL, nclust=3)


# ---------------------------------------------------------------------------
# Errors are reported, not hidden
# ---------------------------------------------------------------------------
def test_no_clusters_returns_none():
    """HDBSCAN labels everything as noise (no real clusters): evaluation returns None (no crash)."""
    data, mods, covs = make_synthetic_data(cluster_std=60.0)
    hdb = HDBSCAN(min_cluster_size=40, min_samples=40, copy=True)
    assert _fb(hdb).evaluate_confounds(data, mods, covs, TR, VAL) is None


class _BrokenClassifier(SVC):
    """A classifier with a bug: raises TypeError when fitted."""
    def fit(self, X, y, sample_weight=None):
        raise TypeError('bug in the classifier')


def test_real_errors_are_not_hidden_in_cross_validation():
    """Before 1.2.0 any TypeError was silently treated as 'no clusters found'."""
    data, mods, covs = make_synthetic_data()
    fb = FindBestClustCVConfounds(GaussianMixture(random_state=42), _BrokenClassifier(),
                                  preprocessing=None, nrand=5, nclust_range=[2, 3])
    with pytest.raises(TypeError, match='bug in the classifier'):
        fb.best_nclust_confounds(data, mods, covs)


def test_real_errors_are_not_hidden_in_evaluation():
    data, mods, covs = make_synthetic_data()
    fb = FindBestClustCVConfounds(GaussianMixture(random_state=42), _BrokenClassifier(),
                                  preprocessing=None, nrand=5)
    with pytest.raises(TypeError, match='bug in the classifier'):
        fb.evaluate_confounds(data, mods, covs, TR, VAL, nclust=3)
