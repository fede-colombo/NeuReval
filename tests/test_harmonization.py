"""
Tests for ComBat harmonization (neureval.harmonization).

Run from the repository root:  pytest tests/test_harmonization.py -v
The comparison with the reference implementation needs neuroCombat
(pip install neuroCombat); it is skipped if neuroCombat is not installed.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from neureval.harmonization import ComBat

warnings.filterwarnings('ignore')

P = 20
SLOPE = np.random.default_rng(100).normal(0.05, 0.01, P)   # true age effect on each feature
SHIFT = {'A': 0.0, 'B': 2.0, 'C': -1.5, 'D': 3.0}
SCALE = {'A': 1.0, 'B': 2.0, 'C': 0.5, 'D': 1.5}
AGE_MEAN = {'A': 40, 'B': 55, 'C': 65, 'D': 45}             # age differs between sites


def simulate(sizes, seed):
    """Data with known site effects (shift and scale) and a known age effect."""
    rng = np.random.default_rng(seed)
    site = np.repeat(list(sizes), list(sizes.values()))
    age = np.concatenate([rng.normal(AGE_MEAN[s], 10, n) for s, n in sizes.items()])
    noise = rng.normal(0, 1, (len(site), P))
    X = np.empty_like(noise)
    for s in sizes:
        m = site == s
        X[m] = noise[m] * SCALE[s] + SHIFT[s] + np.outer(age[m] - 50, SLOPE)
    return X, site, pd.DataFrame({'age': age})


def residual(X, cov):
    """Data without the true age effect: what is left are site effects and noise."""
    return X - np.outer(cov['age'] - 50, SLOPE)


def site_gap(X, site, cov):
    R = residual(X, cov)
    means = np.array([R[site == s].mean(axis=0) for s in np.unique(site)])
    return np.abs(means - means.mean(axis=0)).mean()


def age_slope_error(X, cov):
    A = np.c_[np.ones(len(cov)), cov['age']]
    return np.abs(np.linalg.lstsq(A, X, rcond=None)[0][1] - SLOPE).mean()


@pytest.fixture(scope='module')
def training():
    return simulate({'A': 150, 'B': 150, 'C': 150}, seed=1)


# ---------------------------------------------------------------------------
# Same results as the reference implementation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('eb, mean_only', [(True, False), (False, False), (True, True)])
def test_same_results_as_neurocombat(eb, mean_only):
    nc = pytest.importorskip('neuroCombat')
    rng = np.random.default_rng(0)
    site = np.repeat(['A', 'B', 'C'], [40, 25, 35])
    n = len(site)
    # neuroCombat converts continuous covariates to float32: use values exact in float32
    age = rng.normal(50, 12, n).astype(np.float32).astype(float)
    sex = rng.integers(0, 2, n).astype(float)
    X = rng.normal(0, 1, (n, 30)) + np.outer(age - 50, rng.normal(0, .05, 30)) + np.outer(sex, rng.normal(0, .5, 30))
    for s, shift, scale in [('B', 1.0, 1.8), ('C', -0.7, 0.6)]:
        X[site == s] = X[site == s] * scale + shift
    cov = pd.DataFrame({'site': site, 'age': age, 'sex': sex})
    ref = nc.neuroCombat(dat=X.T.copy(), covars=cov, batch_col='site', continuous_cols=['age', 'sex'],
                         eb=eb, mean_only=mean_only)['data'].T
    ours = ComBat(eb=eb, mean_only=mean_only).fit_transform(X, site, cov[['age', 'sex']])
    np.testing.assert_allclose(ours, ref, rtol=1e-9, atol=1e-10)


# ---------------------------------------------------------------------------
# Behaviour on data with known effects
# ---------------------------------------------------------------------------
def test_removes_site_effects_and_preserves_age(training):
    X, site, cov = training
    H = ComBat().fit_transform(X, site, cov)
    assert site_gap(X, site, cov) > 1.0
    assert site_gap(H, site, cov) < 0.1
    assert age_slope_error(H, cov) < 0.01


def test_age_not_preserved_if_not_specified(training):
    """Age differs between sites: without preserving it, ComBat removes part of its effect."""
    X, site, cov = training
    with_age = age_slope_error(ComBat().fit_transform(X, site, cov), cov)
    without_age = age_slope_error(ComBat().fit_transform(X, site), cov)
    assert without_age > 3 * with_age


def test_transform_new_subjects_same_sites(training):
    """Hold-out validation: model estimated on training, applied to new subjects of the same sites."""
    X, site, cov = training
    cb = ComBat().fit(X, site, cov)
    Xn, sn, cn = simulate({'A': 60, 'B': 60, 'C': 60}, seed=2)
    assert site_gap(cb.transform(Xn, sn, cn), sn, cn) < 0.25 < site_gap(Xn, sn, cn)


def test_transform_training_data_equals_fit_transform(training):
    X, site, cov = training
    cb = ComBat().fit(X, site, cov)
    np.testing.assert_allclose(cb.transform(X, site, cov), ComBat().fit_transform(X, site, cov))


def test_transform_does_not_depend_on_other_subjects(training):
    """Each subject is harmonized with the training model only: no leakage between subjects."""
    X, site, cov = training
    cb = ComBat().fit(X, site, cov)
    Xn, sn, cn = simulate({'A': 20, 'B': 20}, seed=3)
    full = cb.transform(Xn, sn, cn)
    np.testing.assert_allclose(cb.transform(Xn[:5], sn[:5], cn.iloc[:5]), full[:5])


# ---------------------------------------------------------------------------
# External validation: new site calibrated on a separate sample
# ---------------------------------------------------------------------------
def test_unseen_site_raises_without_calibration(training):
    X, site, cov = training
    cb = ComBat().fit(X, site, cov)
    Xd, sd, cd = simulate({'D': 30}, seed=4)
    with pytest.raises(ValueError, match="were not in the training data"):
        cb.transform(Xd, sd, cd)


def test_calibrated_site_is_harmonized(training):
    X, site, cov = training
    cb = ComBat().fit(X, site, cov)
    H = cb.transform(X, site, cov)
    Xd, sd, cd = simulate({'D': 150}, seed=5)
    cal, ev = np.arange(50), np.arange(50, 150)
    params_before = (cb.grand_mean_.copy(), cb.beta_covariates_.copy(), cb.var_pooled_.copy())
    cb.calibrate(Xd[cal], sd[cal], cd.iloc[cal])
    Hd = cb.transform(Xd[ev], sd[ev], cd.iloc[ev])
    ref = residual(H, cov).mean(axis=0)
    gap_before = np.abs(residual(Xd[ev], cd.iloc[ev]).mean(axis=0) - ref).mean()
    gap_after = np.abs(residual(Hd, cd.iloc[ev]).mean(axis=0) - ref).mean()
    assert gap_before > 2.0 and gap_after < 0.3
    # the training model is unchanged, training data are harmonized as before
    for before, after in zip(params_before, (cb.grand_mean_, cb.beta_covariates_, cb.var_pooled_)):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_allclose(cb.transform(X, site, cov), H)
    assert cb.calibrated_sites_.tolist() == ['D']


def test_calibrating_a_training_site_raises(training):
    X, site, cov = training
    cb = ComBat().fit(X, site, cov)
    with pytest.raises(ValueError, match="already in the model"):
        cb.calibrate(X[site == 'A'], site[site == 'A'], cov[site == 'A'])


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------
def test_input_checks(training):
    X, site, cov = training
    with pytest.raises(ValueError, match="at least 2 sites"):
        ComBat().fit(X, np.repeat('A', len(X)), cov)
    one_subject = site.copy()
    one_subject[0] = 'Z'
    with pytest.raises(ValueError, match="at least 2 subjects"):
        ComBat().fit(X, one_subject, cov)
    with pytest.raises(ValueError, match="missing or infinite"):
        Xnan = X.copy()
        Xnan[3, 2] = np.nan
        ComBat().fit(Xnan, site, cov)
    with pytest.raises(ValueError, match="collinear with the sites"):
        ComBat().fit(X, site, pd.DataFrame({'is_B': (site == 'B').astype(float)}))
    with pytest.raises(ValueError, match="must be numerical"):
        ComBat().fit(X, site, pd.DataFrame({'sex': np.where(np.arange(len(X)) % 2, 'F', 'M')}))
    cb = ComBat().fit(X, site, cov)
    with pytest.raises(ValueError, match="same columns as in fit"):
        cb.transform(X, site, cov.rename(columns={'age': 'AGE'}))
    with pytest.raises(ValueError, match="features"):
        cb.transform(X[:, :5], site, cov)


# ===========================================================================
# ComBat within the NeuReval pipeline
# ===========================================================================
import logging  # noqa: E402

from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds  # noqa: E402
from neureval.param_selection_confounds import ParamSelectionConfounds  # noqa: E402
from neureval.relative_validation_confounds import RelativeValidationConfounds  # noqa: E402

logging.disable(logging.CRITICAL)


def make_multisite(seed, sizes):
    """
    3 true clusters, independent of site, and strong site effects (shift and scale)
    that are larger than the cluster differences. Age differs between sites.
    Returns data, modalities, covariates (GLM), site, covariates to preserve, true clusters.
    """
    rng = np.random.default_rng(seed)
    shift = {'A': 0.0, 'B': 4.0, 'C': -4.0, 'D': 3.0}
    scale = {'A': 1.0, 'B': 1.6, 'C': 0.7, 'D': 1.3}
    site = np.repeat(list(sizes), list(sizes.values()))
    n = len(site)
    y = rng.integers(0, 3, n)
    centers = np.zeros((3, 6))
    for k in range(3):
        centers[k, k] = 6 / np.sqrt(2)
    X = np.c_[centers[y] + rng.normal(0, 1, (n, 6)), rng.normal(0, 1, (n, 4))]
    age = np.array([rng.normal({'A': 40, 'B': 55, 'C': 65, 'D': 45}[s], 8) for s in site])
    sex = rng.integers(0, 2, n).astype(float)
    direction = rng.normal(0, 1, X.shape[1])
    direction /= np.linalg.norm(direction) / np.sqrt(X.shape[1])
    for s in sizes:
        X[site == s] = X[site == s] * scale[s] + shift[s] * direction
    X += np.outer((age - 50) / 10, rng.normal(0, .5, X.shape[1])) + np.outer(sex, rng.normal(0, .3, X.shape[1]))
    data = pd.DataFrame(X)
    cov = pd.DataFrame({'age': age, 'sex': sex})
    return data, {'m1': data.iloc[:, :5], 'm2': data.iloc[:, 5:]}, {'m1': cov, 'm2': cov}, site, cov, y


def _findbest(**kw):
    return FindBestClustCVConfounds(GaussianMixture(random_state=42), SVC(random_state=42),
                                    preprocessing=None, nrand=10, **kw)


def test_cross_validation_finds_true_clusters_not_sites():
    data, mods, covs, site, cov, y = make_multisite(2, {'A': 100, 'B': 100, 'C': 100})
    _, k_raw, lab_raw = _findbest(nclust_range=[2, 3, 4, 5]).best_nclust_confounds(data, mods, covs, iter_cv=2)
    _, k_cb, lab_cb = _findbest(nclust_range=[2, 3, 4, 5]).best_nclust_confounds(
        data, mods, covs, iter_cv=2, site=site, harmonize_covariates=cov)
    # without harmonization the clusters follow the sites
    assert adjusted_rand_score(site, lab_raw) > 0.4 and adjusted_rand_score(y, lab_raw) < 0.2
    # with ComBat they follow the true clusters
    assert k_cb == 3
    assert adjusted_rand_score(y, lab_cb) > 0.8 and abs(adjusted_rand_score(site, lab_cb)) < 0.1


def test_harmonization_estimated_on_training_fold_only():
    """Changing the validation subjects does not change the prepared training data (no leakage)."""
    data, mods, covs, site, cov, y = make_multisite(0, {'A': 60, 'B': 60, 'C': 60})
    reval = RelativeValidationConfounds(None, None, None, 10)
    tr, val = np.arange(0, 180, 2), np.arange(1, 180, 2)
    Xtr, Xval = reval._prepare_fold(mods, covs, tr, val, site, cov)
    changed = {k: v.copy() for k, v in mods.items()}
    for v in changed.values():
        v.iloc[val] = v.iloc[val] * 3 + 10
    Xtr2, Xval2 = reval._prepare_fold(changed, covs, tr, val, site, cov)
    np.testing.assert_allclose(Xtr, Xtr2)
    assert not np.allclose(Xval, Xval2)


def test_parallel_same_results():
    data, mods, covs, site, cov, y = make_multisite(1, {'A': 60, 'B': 60, 'C': 60})
    res = []
    for n_jobs in (1, 2):
        fb = _findbest(nclust_range=[2, 3], n_jobs=n_jobs)
        m, k, lab = fb.best_nclust_confounds(data, mods, covs, site=site, harmonize_covariates=cov)
        res.append((m['val'], k, lab))
    assert res[0][1:] == res[1][1:]
    for k in res[0][0]:
        np.testing.assert_allclose(res[0][0][k][0], res[1][0][k][0])


def test_grid_search_with_harmonization():
    data, mods, covs, site, cov, y = make_multisite(2, {'A': 60, 'B': 60, 'C': 60})
    params = {'s': {'C': [1], 'kernel': ['rbf']}, 'c': {'covariance_type': ['full', 'diag']}}
    ps = ParamSelectionConfounds(params, cv=2, c=GaussianMixture(random_state=42), s=SVC(random_state=42),
                                 preprocessing=None, nrand=5, iter_cv=1, clust_range=[2, 3])
    ps.fit(data, mods, covs, site=site, harmonize_covariates=cov)
    assert len(ParamSelectionConfounds.cv_results_['best_nclust']) == 2


def test_dictionary_of_covariates_per_modality():
    data, mods, covs, site, cov, y = make_multisite(0, {'A': 60, 'B': 60, 'C': 60})
    fb = _findbest(nclust_range=[2, 3])
    fb.best_nclust_confounds(data, mods, covs, site=site, harmonize_covariates={'m1': cov, 'm2': cov[['age']]})
    with pytest.raises(ValueError, match="no entry for the modalities"):
        fb.best_nclust_confounds(data, mods, covs, site=site, harmonize_covariates={'m1': cov})


# ---------------------------------------------------------------------------
# Hold-out validation (same sites) and external validation (new site)
# ---------------------------------------------------------------------------
def test_holdout_same_sites():
    data, mods, covs, site, cov, y = make_multisite(2, {'A': 150, 'B': 150, 'C': 150})
    tr = np.zeros(len(y), dtype=bool)
    tr[np.random.default_rng(2).choice(len(y), 300, replace=False)] = True
    raw = _findbest().evaluate_confounds(data, mods, covs, tr, ~tr, nclust=3)
    cb = _findbest().evaluate_confounds(data, mods, covs, tr, ~tr, nclust=3, site=site, harmonize_covariates=cov)
    assert adjusted_rand_score(y[~tr], raw.test_classlab) < 0.2
    assert adjusted_rand_score(y[~tr], cb.test_classlab) > 0.8


def test_external_site_needs_calibration_and_calibration_works():
    data, mods, covs, site, cov, y = make_multisite(0, {'A': 150, 'B': 150, 'C': 150, 'D': 150})
    idx = np.arange(len(y))
    tr, cal, ev = idx[site != 'D'], idx[site == 'D'][:50], idx[site == 'D'][50:]
    with pytest.raises(ValueError, match="were not in the training data"):
        _findbest().evaluate_confounds(data, mods, covs, tr, ev, nclust=3, site=site, harmonize_covariates=cov)
    out = _findbest().evaluate_confounds(data, mods, covs, tr, ev, nclust=3, site=site,
                                         harmonize_covariates=cov, calibration_idx=cal)
    assert len(out.test_classlab) == len(ev)
    assert adjusted_rand_score(y[ev], out.test_classlab) > 0.8


def test_calibration_input_checks():
    data, mods, covs, site, cov, y = make_multisite(0, {'A': 60, 'B': 60, 'C': 60, 'D': 60})
    idx = np.arange(len(y))
    tr, d = idx[site != 'D'], idx[site == 'D']
    fb = _findbest()
    kw = dict(nclust=3, harmonize_covariates=cov)
    with pytest.raises(ValueError, match="also in the validation set"):
        fb.evaluate_confounds(data, mods, covs, tr, d, site=site, calibration_idx=d[:20], **kw)
    with pytest.raises(ValueError, match="which are in the training set"):
        fb.evaluate_confounds(data, mods, covs, tr[:-20], d, site=site, calibration_idx=tr[-20:], **kw)
    with pytest.raises(ValueError, match="needs the site"):
        fb.evaluate_confounds(data, mods, covs, tr, d[20:], calibration_idx=d[:20], nclust=3)
    with pytest.raises(ValueError, match="site is missing"):
        FindBestClustCVConfounds(GaussianMixture(random_state=42), SVC(random_state=42), None, 10,
                                 harmonization=ComBat()).evaluate_confounds(data, mods, covs, tr, d, nclust=3)
