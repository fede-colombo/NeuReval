"""
Tests for saving and loading fitted models and the whole pipeline (neureval.export).

Run from the repository root:  pytest tests/test_export.py -v
"""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from neureval.export import (GLMCorrection, export_model, load_model, load_models, model_from_dict,
                             model_to_dict, save_models)
from neureval.harmonization import ComBat
from neureval.relative_validation_confounds import RelativeValidationConfounds

warnings.filterwarnings('ignore')


def simulate(n_per_site, sites, n_features=15, seed=0):
    """Multisite data with site shifts and an age effect."""
    rng = np.random.default_rng(seed)
    site = np.repeat(sites, n_per_site)
    n = len(site)
    age = rng.normal(50, 10, n)
    sex = rng.integers(0, 2, n)
    shift = {s: rng.normal(0, 1.5, n_features) for s in sites}
    X = rng.normal(0, 1, (n, n_features)) + np.outer(age - 50, rng.normal(0.05, 0.01, n_features))
    X += np.vstack([shift[s] for s in site])
    cov = pd.DataFrame({'age': age, 'sex': sex})
    return X, site, cov


@pytest.fixture(scope='module')
def data():
    X, site, cov = simulate(60, ['A', 'B', 'C'], seed=1)
    rng = np.random.default_rng(2)
    idx = rng.permutation(len(site))
    return X, site, cov, idx[:120], idx[120:]


# ---------------------------------------------------------------------------
# GLMCorrection gives the same results as the pipeline
# ---------------------------------------------------------------------------
def test_glm_same_as_pipeline(data):
    X, _, cov, tr, val = data
    reval = RelativeValidationConfounds(c=KMeans(2, n_init=10), s=LogisticRegression())
    train_ref, test_ref = reval.GLMcorrection_confounds({'mri': X}, {'mri': cov}, tr, val)

    glm = GLMCorrection().fit(X[tr], cov.iloc[tr])
    np.testing.assert_array_equal(glm.transform(X[tr], cov.iloc[tr]), train_ref['mri'])
    np.testing.assert_array_equal(glm.transform(X[val], cov.iloc[val]), test_ref['mri'])


def test_glm_errors(data):
    X, _, cov, tr, _ = data
    with pytest.raises(ValueError, match="not fitted"):
        GLMCorrection().transform(X, cov)
    cov_nan = cov.copy()
    cov_nan.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="Covariates contain missing"):
        GLMCorrection().fit(X, cov_nan)
    glm = GLMCorrection().fit(X[tr], cov.iloc[tr])
    with pytest.raises(ValueError, match="features"):
        glm.transform(X[:, :5], cov)
    with pytest.raises(ValueError, match="covariates"):
        glm.transform(X, cov[['age']])


# ---------------------------------------------------------------------------
# Saved and loaded models give exactly the same results
# ---------------------------------------------------------------------------
def test_round_trip_exact(data, tmp_path):
    X, site, cov, tr, val = data
    combat = ComBat().fit(X[tr], site[tr], cov.iloc[tr][['age']])
    H = combat.transform(X, site, cov[['age']])
    glm = GLMCorrection().fit(H[tr], cov.iloc[tr])

    path = tmp_path / 'model.json'
    save_models(path, {'combat_mri': combat, 'glm_mri': glm}, metadata={'k': 3, 'features': ['f1', 'f2']})
    models, metadata = load_models(path)

    assert metadata == {'k': 3, 'features': ['f1', 'f2']}
    H2 = models['combat_mri'].transform(X, site, cov[['age']])
    np.testing.assert_array_equal(H2, H)
    np.testing.assert_array_equal(models['glm_mri'].transform(H2[val], cov.iloc[val]),
                                  glm.transform(H[val], cov.iloc[val]))


def test_calibrate_after_loading(tmp_path):
    """The external centre calibrates its own (new) site on the loaded model."""
    X, site, cov = simulate(60, ['A', 'B', 'C'], seed=3)
    X_new, site_new, cov_new = simulate(40, ['EXT'], seed=4)
    combat = ComBat().fit(X, site, cov[['age']])
    save_models(tmp_path / 'model.json', {'combat': combat})
    loaded = load_models(tmp_path / 'model.json')[0]['combat']

    calib, evaluation = np.arange(20), np.arange(20, 40)
    combat.calibrate(X_new[calib], site_new[calib], cov_new.iloc[calib][['age']])
    loaded.calibrate(X_new[calib], site_new[calib], cov_new.iloc[calib][['age']])
    np.testing.assert_array_equal(
        loaded.transform(X_new[evaluation], site_new[evaluation], cov_new.iloc[evaluation][['age']]),
        combat.transform(X_new[evaluation], site_new[evaluation], cov_new.iloc[evaluation][['age']]))


@pytest.mark.parametrize('sites', [['A', 'B'], [1, 2], ['site 10', 'site 2']])
def test_site_labels_kept(sites, tmp_path):
    X, site, _ = simulate(30, sites, seed=5)
    combat = ComBat(eb=False, mean_only=True).fit(X, site)
    loaded = model_from_dict(json.loads(json.dumps(model_to_dict(combat))))
    assert list(loaded.sites_) == list(combat.sites_)
    assert loaded.eb is False and loaded.mean_only is True
    np.testing.assert_array_equal(loaded.transform(X, site), combat.transform(X, site))


# ---------------------------------------------------------------------------
# The file is transparent and contains aggregated parameters only
# ---------------------------------------------------------------------------
def test_file_is_plain_json_without_individual_data(data, tmp_path):
    X, site, cov, tr, _ = data
    combat = ComBat().fit(X[tr], site[tr])
    glm = GLMCorrection().fit(X[tr], cov.iloc[tr])
    path = tmp_path / 'model.json'
    save_models(path, {'combat': combat, 'glm': glm})

    content = json.loads(path.read_text())   # standard JSON, no code
    assert content['format'] == 'neureval-model'
    n_train = len(tr)
    for model in content['models'].values():
        for value in model['attributes'].values():
            if isinstance(value, dict) and 'shape' in value:
                # no array has one row per training subject
                assert n_train not in value['shape']


def test_unsupported_models_refused():
    X, site, _ = simulate(30, ['A', 'B'], seed=6)
    knn = KNeighborsClassifier().fit(X, site)   # keeps the training data inside
    with pytest.raises(TypeError, match="cannot be saved"):
        model_to_dict(knn)
    with pytest.raises(ValueError, match="not fitted"):
        model_to_dict(ComBat())


def test_wrong_file(tmp_path):
    path = tmp_path / 'other.json'
    path.write_text(json.dumps({'something': 1}))
    with pytest.raises(ValueError, match="not a NeuReval model file"):
        load_models(path)
    path.write_text(json.dumps({'format': 'neureval-model', 'format_version': 99, 'models': {}}))
    with pytest.raises(ValueError, match="newer version"):
        load_models(path)


# ---------------------------------------------------------------------------
# Whole pipeline: export_model (training centre) and load_model (external centre)
# ---------------------------------------------------------------------------
def clustered_data(sites, n_per_site, shift_size=3.0, seed=0):
    """3 clusters independent of site, site shifts, an age effect; two modalities as DataFrames."""
    rng = np.random.default_rng(seed)
    site = np.repeat(sites, n_per_site)
    n = len(site)
    y = rng.choice(3, size=n, p=[0.4, 0.35, 0.25])
    centers = np.zeros((3, 8))
    for k in range(3):
        centers[k, k] = 5
    X = centers[y] + rng.normal(0, 1, (n, 8))
    age = rng.normal(50, 10, n)
    sex = rng.integers(0, 2, n).astype(float)
    X += np.outer(age - 50, rng.normal(0.05, 0.02, 8))
    for i, s in enumerate(sites):
        X[site == s] += shift_size * (i % 3 - 1)
    data = pd.DataFrame(X, columns=[f'f{i}' for i in range(8)])
    cov = pd.DataFrame({'age': age, 'sex': sex})
    modalities = {'mri': data.iloc[:, :5], 'dti': data.iloc[:, 5:]}
    covariates = {'mri': cov, 'dti': cov[['age']]}
    return data, modalities, covariates, site, cov, y


def subset(modalities, covariates, idx):
    return ({k: v.iloc[idx].reset_index(drop=True) for k, v in modalities.items()},
            {k: v.iloc[idx].reset_index(drop=True) for k, v in covariates.items()})


def assert_same_evaluation(out, ref, exported):
    np.testing.assert_array_equal(exported.train_cllab, ref.train_cllab)
    assert out.train_cllab is None                     # training labels are not shared
    assert out.train_acc == ref.train_acc
    np.testing.assert_array_equal(out.test_cllab, ref.test_cllab)
    np.testing.assert_array_equal(out.test_classlab, ref.test_classlab)
    assert out.test_acc == ref.test_acc
    assert out.test_stability == ref.test_stability


@pytest.mark.parametrize('classifier, preproc', [
    (LogisticRegression(max_iter=1000), None),
    (QuadraticDiscriminantAnalysis(reg_param=0.01), PCA(n_components=4)),
    (LinearDiscriminantAnalysis(), None),
])
def test_export_same_as_evaluate_confounds(classifier, preproc, tmp_path):
    data, modalities, covariates, _, _, _ = clustered_data(['A'], 300, seed=10)
    idx = np.random.default_rng(11).permutation(300)
    tr, val = idx[:200], idx[200:]
    fb = FindBestClustCVConfounds(GaussianMixture(n_init=3, random_state=42), classifier,
                                  preprocessing=preproc, nrand=10)
    ref = fb.evaluate_confounds(data, modalities, covariates, tr, val, nclust=3)

    m_tr, c_tr = subset(modalities, covariates, tr)
    exported = export_model(tmp_path / 'model.json', fb, m_tr, c_tr, nclust=3)
    m_val, c_val = subset(modalities, covariates, val)
    model = load_model(tmp_path / 'model.json')
    out = model.evaluate(m_val, c_val)
    assert_same_evaluation(out, ref, exported)
    np.testing.assert_array_equal(model.predict(m_val, c_val), ref.test_classlab)
    assert model.info['nclust'] == 3 and model.info['n_subjects'] == 200


def test_export_with_combat_and_calibration(tmp_path):
    """External site D, calibrated on part of its subjects: same results as evaluate_confounds."""
    data, modalities, covariates, site, cov, _ = clustered_data(['A', 'B', 'C', 'D'], 120, seed=12)
    hcov = cov[['age']]
    tr = np.flatnonzero(site != 'D')
    d = np.flatnonzero(site == 'D')
    cal, ext = d[:40], d[40:]
    fb = FindBestClustCVConfounds(GaussianMixture(n_init=3, random_state=42),
                                  QuadraticDiscriminantAnalysis(reg_param=0.01), preprocessing=None, nrand=10)
    ref = fb.evaluate_confounds(data, modalities, covariates, tr, ext, nclust=3, site=site,
                                harmonize_covariates=hcov, calibration_idx=cal)

    m_tr, c_tr = subset(modalities, covariates, tr)
    exported = export_model(tmp_path / 'model.json', fb, m_tr, c_tr, nclust=3, site=site[tr],
                            harmonize_covariates=hcov.iloc[tr].reset_index(drop=True))
    # External centre: only site D, calibration subjects first
    local = np.concatenate([cal, ext])
    m_d, c_d = subset(modalities, covariates, local)
    h_d = hcov.iloc[local].reset_index(drop=True)
    model = load_model(tmp_path / 'model.json')
    with pytest.raises(ValueError, match="calibrat"):
        model.evaluate(m_d, c_d, site=site[local], harmonize_covariates=h_d)      # new site, not calibrated
    out = model.evaluate(m_d, c_d, site=site[local], harmonize_covariates=h_d,
                         calibration_idx=np.arange(len(cal)))
    assert_same_evaluation(out, ref, exported)
    # The loaded model is not changed by the calibration: a second evaluation gives the same results
    out2 = model.evaluate(m_d, c_d, site=site[local], harmonize_covariates=h_d,
                          calibration_idx=np.arange(len(cal)))
    assert out2.test_stability == out.test_stability
    assert model.info['training_sites'] == ['A', 'B', 'C']


def test_export_file_has_no_individual_data(tmp_path):
    data, modalities, covariates, _, _, _ = clustered_data(['A'], 237, seed=13)
    fb = FindBestClustCVConfounds(KMeans(n_init=10, random_state=0), LogisticRegression(max_iter=1000),
                                  preprocessing=None, nrand=5)
    export_model(tmp_path / 'model.json', fb, modalities, covariates, nclust=3, metadata={'cohort': 'test'})
    content = json.loads((tmp_path / 'model.json').read_text())

    def shapes(value):
        if isinstance(value, dict) and 'shape' in value:
            yield value['shape']
        elif isinstance(value, list):
            for v in value:
                yield from shapes(v)

    for model in content['models'].values():
        for value in model['attributes'].values():
            for shape in shapes(value):
                assert 237 not in shape          # no array with one row per training subject
    assert content['metadata']['user'] == {'cohort': 'test'}
    assert sum(content['metadata']['pipeline']['cluster_sizes'].values()) == 237


def test_export_refuses_methods_with_individual_data(tmp_path):
    data, modalities, covariates, _, _, _ = clustered_data(['A'], 100, seed=14)
    fb = FindBestClustCVConfounds(GaussianMixture(), SVC(), preprocessing=None, nrand=5)
    with pytest.raises(TypeError, match="SVC cannot be exported as classifier"):
        export_model(tmp_path / 'model.json', fb, modalities, covariates, nclust=3)
    assert not (tmp_path / 'model.json').exists()
    fb = FindBestClustCVConfounds(GaussianMixture(), LogisticRegression(),
                                  preprocessing=KNeighborsClassifier(), nrand=5)
    with pytest.raises(TypeError, match="preprocessing"):
        export_model(tmp_path / 'model.json', fb, modalities, covariates, nclust=3)


def test_external_data_checked(tmp_path):
    data, modalities, covariates, site, _, _ = clustered_data(['A'], 150, seed=15)
    fb = FindBestClustCVConfounds(GaussianMixture(n_init=2, random_state=0), LogisticRegression(),
                                  preprocessing=None, nrand=5)
    export_model(tmp_path / 'model.json', fb, modalities, covariates, nclust=3)
    model = load_model(tmp_path / 'model.json')

    swapped = dict(modalities, mri=modalities['mri'][['f1', 'f0', 'f2', 'f3', 'f4']])
    with pytest.raises(ValueError, match="different order"):
        model.evaluate(swapped, covariates)
    renamed = dict(modalities, mri=modalities['mri'].rename(columns={'f4': 'other'}))
    with pytest.raises(ValueError, match="missing \\['f4'\\]"):
        model.evaluate(renamed, covariates)
    with pytest.raises(ValueError, match="modalities"):
        model.evaluate({'mri': modalities['mri']}, {'mri': covariates['mri']})
    with pytest.raises(ValueError, match="without site harmonization"):
        model.evaluate(modalities, covariates, site=site)
    # Arrays instead of DataFrames: only the number of columns can be checked
    arrays = {k: v.to_numpy() for k, v in modalities.items()}
    assert model.evaluate(arrays, covariates) is not None
    with pytest.raises(ValueError, match="columns"):
        model.evaluate(dict(arrays, mri=arrays['mri'][:, :4]), covariates)


def test_load_model_needs_a_pipeline(tmp_path):
    X, site, _ = simulate(30, ['A', 'B'], seed=16)
    save_models(tmp_path / 'combat.json', {'combat': ComBat().fit(X, site)})
    with pytest.raises(ValueError, match="export_model"):
        load_model(tmp_path / 'combat.json')

# ---------------------------------------------------------------------------
# General checks for scikit-learn estimators (no fixed list of classifiers)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('classifier, clustering', [
    (LinearSVC(), GaussianMixture(n_init=3, random_state=42)),
    (GaussianNB(), GaussianMixture(n_init=3, random_state=42)),
    (LogisticRegression(max_iter=1000), KMeans(n_init=10, random_state=0)),
    (SGDClassifier(random_state=0), BayesianGaussianMixture(n_init=3, random_state=0)),
])
def test_other_classifiers_exported(classifier, clustering, tmp_path):
    """Classifiers with aggregated parameters pass the checks and give the same results."""
    data, modalities, covariates, _, _, _ = clustered_data(['A'], 300, seed=20)
    idx = np.random.default_rng(21).permutation(300)
    tr, val = idx[:200], idx[200:]
    fb = FindBestClustCVConfounds(clustering, classifier, preprocessing=None, nrand=10)
    ref = fb.evaluate_confounds(data, modalities, covariates, tr, val, nclust=3)
    m_tr, c_tr = subset(modalities, covariates, tr)
    exported = export_model(tmp_path / 'model.json', fb, m_tr, c_tr, nclust=3)
    m_val, c_val = subset(modalities, covariates, val)
    assert_same_evaluation(load_model(tmp_path / 'model.json').evaluate(m_val, c_val), ref, exported)


@pytest.mark.parametrize('classifier, reason', [
    (SVC(kernel='linear'), "data of training subjects"),       # support vectors
    (SVC(kernel='rbf'), "data of training subjects"),
    (KNeighborsClassifier(), "cannot be rebuilt"),             # training data in a private attribute
    (RidgeClassifier(), "cannot be rebuilt"),                  # predict needs a private object
    (RandomForestClassifier(n_estimators=5, random_state=0), "object and cannot be saved"),   # trees
])
def test_classifiers_with_individual_data_refused(classifier, reason, tmp_path):
    data, modalities, covariates, _, _, _ = clustered_data(['A'], 150, seed=22)
    fb = FindBestClustCVConfounds(GaussianMixture(n_init=2, random_state=0), classifier,
                                  preprocessing=None, nrand=3)
    with pytest.raises(TypeError, match=reason):
        export_model(tmp_path / 'model.json', fb, modalities, covariates, nclust=3)
    assert not (tmp_path / 'model.json').exists()


def test_only_sklearn_classes_loaded(tmp_path):
    """A model file cannot make NeuReval build an object that is not a scikit-learn estimator."""
    X, site, _ = simulate(30, ['A', 'B'], seed=23)
    d = model_to_dict(LogisticRegression().fit(X, site))
    for module, name in (('os', 'system'), ('subprocess', 'Popen'), ('sklearn.utils', 'Bunch'),
                         ('sklearn.model_selection', 'KFold')):
        with pytest.raises(ValueError, match="refused"):
            model_from_dict(dict(d, module=module, **{'class': name}))


def test_fingerprint_detects_a_different_model(tmp_path):
    """If the rebuilt model gives other results (e.g. another scikit-learn version), loading fails."""
    X, site, _ = simulate(40, ['A', 'B'], seed=24)
    d = model_to_dict(LogisticRegression().fit(X, site))
    assert set(d['fingerprint']['outputs']) == {'predict'}
    model_from_dict(json.loads(json.dumps(d)))                 # unchanged: loads
    d['attributes']['coef_']['data'] = [-v for v in d['attributes']['coef_']['data']]   # as if rebuilt differently
    with pytest.raises(ValueError, match="same results as when it was saved"):
        model_from_dict(d)


def test_min_cluster_size(tmp_path):
    data, modalities, covariates, _, _, _ = clustered_data(['A'], 120, seed=25)
    labels = np.zeros(120, dtype=int)
    labels[:3] = 1                                             # a cluster of 3 subjects
    labels[3:60] = 2
    fb = FindBestClustCVConfounds(HDBSCAN(), LogisticRegression(), preprocessing=None, nrand=3)
    with pytest.raises(ValueError, match="fewer than 10"):
        export_model(tmp_path / 'model.json', fb, modalities, covariates, tr_lab=labels)
    export_model(tmp_path / 'model.json', fb, modalities, covariates, tr_lab=labels, min_cluster_size=0)
    assert load_model(tmp_path / 'model.json').info['cluster_sizes'] == {'0': 60, '1': 3, '2': 57}