"""
Tests for saving and loading fitted models (neureval.export).

Run from the repository root:  pytest tests/test_export.py -v
"""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from neureval.export import (GLMCorrection, load_models, model_from_dict, model_to_dict,
                             save_models)
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
