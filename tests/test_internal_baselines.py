"""
Tests for the internal measures (neureval.internal_baselines_confounds).
Their results without harmonization are also checked in tests/test_reference.py.

Run from the repository root:  pytest tests/test_internal_baselines.py -v
"""
import logging
import os
import sys
import warnings

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture

from neureval.harmonization import ComBat
from neureval.internal_baselines_confounds import select_best, select_best_bic_aic, evaluate_best
from neureval.relative_validation_confounds import RelativeValidationConfounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_cases import make_synthetic_data  # noqa: E402
from test_harmonization import make_multisite  # noqa: E402

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)


@pytest.fixture(scope='module')
def multisite():
    return make_multisite(1, {'A': 100, 'B': 100, 'C': 100})


def test_silhouette_with_combat_finds_true_clusters_not_sites(multisite):
    data, mods, covs, site, cov, y = multisite
    _, k_raw, lab_raw = select_best(data, mods, covs, GaussianMixture(n_init=10, random_state=42),
                                    silhouette_score, select='max', nclust_range=[2, 3, 4, 5])
    _, k_cb, lab_cb = select_best(data, mods, covs, GaussianMixture(n_init=10, random_state=42),
                                  silhouette_score, select='max', nclust_range=[2, 3, 4, 5],
                                  site=site, harmonize_covariates=cov)
    # without harmonization the clusters follow the sites, with ComBat the true clusters
    assert adjusted_rand_score(site, lab_raw) > 0.4 and adjusted_rand_score(y, lab_raw) < 0.2
    assert k_cb == 3
    assert adjusted_rand_score(y, lab_cb) > 0.8 and abs(adjusted_rand_score(site, lab_cb)) < 0.1


def test_bic_aic_with_combat_does_not_follow_sites(multisite):
    data, mods, covs, site, cov, y = multisite
    for score in ('bic', 'aic'):
        _, _, lab = select_best_bic_aic(data, mods, covs, GaussianMixture(n_init=10, random_state=42), score=score,
                                        nclust_range=[2, 3, 4], site=site, harmonize_covariates=cov)
        assert abs(adjusted_rand_score(site, lab)) < 0.1


@pytest.mark.parametrize('c, measure, select', [
    (GaussianMixture(n_init=5, random_state=42), silhouette_score, 'max'),
    (KMeans(n_init=10, random_state=42), davies_bouldin_score, 'min'),
])
def test_evaluate_best_matches_select_best(multisite, c, measure, select):
    """evaluate_best gives the same score as select_best for the selected number of clusters."""
    data, mods, covs, site, cov, y = multisite
    kw = dict(site=site, harmonize_covariates=cov)
    score, k, _ = select_best(data, mods, covs, c, measure, select=select, nclust_range=[2, 3, 4], **kw)
    assert evaluate_best(data, mods, covs, c, measure, ncl=k, **kw) == pytest.approx(score)


def test_same_confound_correction_as_neureval():
    """The internal measures use the same confound correction as the rest of NeuReval."""
    data, mods, covs = make_synthetic_data()
    all_idx = np.arange(len(data))
    corrected, _ = RelativeValidationConfounds(None, None).GLMcorrection_confounds(mods, covs, all_idx, all_idx)
    expected = np.concatenate([corrected[m] for m in corrected], axis=1)
    label = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(expected)
    score = evaluate_best(data, mods, covs, KMeans(n_init=10, random_state=42), silhouette_score, ncl=3)
    assert score == pytest.approx(silhouette_score(expected, label))


def test_user_clustering_object_not_modified():
    data, mods, covs = make_synthetic_data()
    km = KMeans(n_clusters=8, n_init=10, random_state=42)
    select_best(data, mods, covs, km, silhouette_score, nclust_range=[2, 3])
    evaluate_best(data, mods, covs, km, silhouette_score, ncl=3)
    gmm = GaussianMixture(n_components=7, random_state=42)
    select_best_bic_aic(data, mods, covs, gmm, nclust_range=[2, 3])
    assert km.n_clusters == 8 and gmm.n_components == 7


def test_input_checks():
    data, mods, covs = make_synthetic_data()
    with pytest.raises(ValueError, match="nclust_range is required"):
        select_best_bic_aic(data, mods, covs, GaussianMixture(random_state=42))
    with pytest.raises(ValueError, match="score must be"):
        select_best_bic_aic(data, mods, covs, GaussianMixture(random_state=42), score='hqic', nclust_range=[2, 3])
    with pytest.raises(ValueError, match="ncl is required"):
        evaluate_best(data, mods, covs, KMeans(n_init=10), silhouette_score)
    with pytest.raises(ValueError, match="site is missing"):
        select_best(data, mods, covs, KMeans(n_init=10), silhouette_score, nclust_range=[2, 3],
                    harmonization=ComBat())
    nan_covs = {m: c.copy() for m, c in covs.items()}
    nan_covs['mod_01'].iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        select_best(data, mods, nan_covs, KMeans(n_init=10), silhouette_score, nclust_range=[2, 3])
