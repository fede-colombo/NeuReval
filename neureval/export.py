"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience,
         IRCCS San Raffaele Scientific Institute, Milan, Italy

Save and load fitted models in a transparent format, for external validation when the
training data cannot be shared.

In this scenario the training centre fits the model and sends only the fitted parameters
to the external centre, which evaluates the model on its own data. The file must
therefore contain aggregated parameters only (means, variances, coefficients), never
individual data, and it must be possible to check this by opening it.

For this reason the models are saved as plain JSON, not with pickle:

- the file can be opened and read by anyone (e.g. a data protection officer);
- loading it does not run any code, so a file from another centre is safe to open;
- it does not depend on the versions of scikit-learn or numpy used to save it;
- numbers are written with full precision, so a loaded model gives exactly the same
  results as the original one.

Only models whose fitted parameters are aggregated can be saved. Models that keep
individual training data inside (e.g. k-nearest neighbours, SVM with support vectors,
UMAP) are refused.

This module contains the building blocks (harmonization and confound correction);
saving the whole NeuReval pipeline (export_model / load_model) builds on them.
"""
import json
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from sklearn.preprocessing import StandardScaler

from neureval.harmonization import ComBat

FORMAT_VERSION = 1


class GLMCorrection:
    """
    Confound correction with a linear model, with separate estimation (`fit`) and
    application (`transform`).

    It gives the same results as `RelativeValidationConfounds.GLMcorrection_confounds`
    for one modality: features and covariates are z-scored with the means and standard
    deviations of the training subjects, then the effects of the covariates, estimated on
    the training subjects, are removed.

    Attributes after fitting (all aggregated over the training subjects):

    - x_mean_, x_scale_: mean and standard deviation of each feature, (n_features,);
    - cov_mean_, cov_scale_: mean and standard deviation of each covariate, (n_covariates,);
    - beta_: effect of each covariate on each feature, (n_covariates, n_features).
    """

    def fit(self, X, covariates):
        """
        Estimate the correction on the training subjects.

        :param X: training data, (n_subjects, n_features).
        :type X: array-like
        :param covariates: covariates to remove, (n_subjects, n_covariates).
        :type covariates: array-like
        :return: self
        """
        X, C = self._check_inputs(X, covariates)
        x_scaler = StandardScaler().fit(X)
        cov_scaler = StandardScaler().fit(C)
        self.x_mean_, self.x_scale_ = x_scaler.mean_, x_scaler.scale_
        self.cov_mean_, self.cov_scale_ = cov_scaler.mean_, cov_scaler.scale_
        Xs, Cs = self._scale(X, C)
        self.beta_ = np.linalg.lstsq(Cs, Xs, rcond=None)[0]
        return self

    def transform(self, X, covariates):
        """
        Remove the covariate effects estimated in `fit` from any subjects.

        :param X: data, (n_subjects, n_features), same features as in `fit`.
        :type X: array-like
        :param covariates: covariates of these subjects, same columns as in `fit`.
        :type covariates: array-like
        :return: corrected data, (n_subjects, n_features).
        :rtype: ndarray
        """
        if not hasattr(self, 'beta_'):
            raise ValueError("GLMCorrection is not fitted yet: call fit first.")
        X, C = self._check_inputs(X, covariates)
        if X.shape[1] != len(self.x_mean_):
            raise ValueError(f"Expected {len(self.x_mean_)} features (as in fit), got {X.shape[1]}.")
        if C.shape[1] != len(self.cov_mean_):
            raise ValueError(f"Expected {len(self.cov_mean_)} covariates (as in fit), got {C.shape[1]}.")
        Xs, Cs = self._scale(X, C)
        # Same expression as GLMcorrection_confounds, so that the results are identical
        return (Xs.T - self.beta_.T @ Cs.T).T

    def fit_transform(self, X, covariates):
        """Fit the correction on the data and correct the same data."""
        return self.fit(X, covariates).transform(X, covariates)

    def _scale(self, X, C):
        return (X - self.x_mean_) / self.x_scale_, (C - self.cov_mean_) / self.cov_scale_

    @staticmethod
    def _check_inputs(X, covariates):
        # C-ordered arrays, as in GLMcorrection_confounds: the memory layout can change the
        # last digits of the linear algebra results
        X = np.ascontiguousarray(X, dtype=float)
        C = np.ascontiguousarray(covariates, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        if X.shape[0] != C.shape[0]:
            raise ValueError(f"Data have {X.shape[0]} rows but covariates have {C.shape[0]}.")
        if not np.all(np.isfinite(X)):
            raise ValueError("Data contain missing or infinite values.")
        if not np.all(np.isfinite(C)):
            raise ValueError("Covariates contain missing or infinite values.")
        return X, C


# ----------------------------------------------------------------------
# Models that can be saved, with the attributes that describe them
# ----------------------------------------------------------------------
# For each class: the constructor parameters and the fitted attributes to save.
# Everything else (and any class not listed here) is never written to the file.
_SUPPORTED = {
    'ComBat': {
        'cls': ComBat,
        'params': ['eb', 'mean_only'],
        'attributes': ['sites_', 'grand_mean_', 'beta_covariates_', 'var_pooled_', 'gamma_star_',
                       'delta_star_', 'calibrated_sites_', 'n_features_in_', 'covariate_names_',
                       'n_covariates_'],
    },
    'GLMCorrection': {
        'cls': GLMCorrection,
        'params': [],
        'attributes': ['x_mean_', 'x_scale_', 'cov_mean_', 'cov_scale_', 'beta_'],
    },
}


def model_to_dict(model):
    """
    Describe a fitted model as a dictionary of plain values (numbers, strings, lists),
    ready to be written as JSON.

    :param model: fitted model, of one of the supported classes (ComBat, GLMCorrection).
    :return: dictionary with the class name, constructor parameters and fitted attributes.
    :rtype: dict
    """
    name = type(model).__name__
    if name not in _SUPPORTED or not isinstance(model, _SUPPORTED[name]['cls']):
        raise TypeError(f"{name} cannot be saved. Only models whose fitted parameters are aggregated "
                        f"over subjects can be saved: {sorted(_SUPPORTED)}.")
    spec = _SUPPORTED[name]
    missing = [a for a in spec['attributes'] if not hasattr(model, a)]
    if missing:
        raise ValueError(f"{name} is not fitted yet (missing {missing}): call fit before saving it.")
    return {
        'class': name,
        'params': {p: getattr(model, p) for p in spec['params']},
        'attributes': {a: _encode(getattr(model, a)) for a in spec['attributes']},
    }


def model_from_dict(d):
    """
    Rebuild a fitted model from the dictionary written by `model_to_dict`.

    :param d: dictionary with the class name, constructor parameters and fitted attributes.
    :type d: dict
    :return: fitted model.
    """
    name = d.get('class')
    if name not in _SUPPORTED:
        raise ValueError(f"Unknown model class {name!r}. Known classes: {sorted(_SUPPORTED)}.")
    spec = _SUPPORTED[name]
    model = spec['cls'](**d['params'])
    missing = [a for a in spec['attributes'] if a not in d['attributes']]
    if missing:
        raise ValueError(f"The saved {name} is incomplete: missing {missing}.")
    for a in spec['attributes']:
        setattr(model, a, _decode(d['attributes'][a]))
    return model


def save_models(path, models, metadata=None):
    """
    Save fitted models to a JSON file.

    :param path: path of the file to write (e.g. 'model.json').
    :type path: str
    :param models: fitted models to save, e.g. {'combat_mri': ..., 'glm_mri': ...}.
    :type models: dict
    :param metadata: other information to store (e.g. feature names, number of clusters).
        Must contain only numbers, strings, booleans, lists and dictionaries.
    :type metadata: dict
    """
    content = {
        'format': 'neureval-model',
        'format_version': FORMAT_VERSION,
        'neureval_version': _neureval_version(),
        'metadata': metadata or {},
        'models': {key: model_to_dict(m) for key, m in models.items()},
    }
    with open(path, 'w', encoding='utf-8') as f:
        # allow_nan=False: fitted parameters must be finite, and the file must be standard JSON
        json.dump(content, f, indent=1, allow_nan=False)


def load_models(path):
    """
    Load the models saved with `save_models`. No code is run while loading.

    :param path: path of the JSON file.
    :type path: str
    :return: the fitted models (same keys as when saved) and the metadata.
    :rtype: dict, dict
    """
    with open(path, encoding='utf-8') as f:
        content = json.load(f)
    if content.get('format') != 'neureval-model':
        raise ValueError(f"{path} is not a NeuReval model file.")
    if content.get('format_version', 0) > FORMAT_VERSION:
        raise ValueError(f"{path} was written by a newer version of NeuReval "
                         f"({content.get('neureval_version')}): update NeuReval to read it.")
    models = {key: model_from_dict(d) for key, d in content['models'].items()}
    return models, content.get('metadata', {})


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _encode(value):
    """Arrays become {'dtype', 'shape', 'data'}; other values are kept as they are."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'USO':
            return {'dtype': 'str', 'shape': list(value.shape), 'data': [str(v) for v in value.ravel()]}
        if value.dtype.kind not in 'biuf':
            raise TypeError(f"Arrays of type {value.dtype} cannot be saved.")
        # Python floats are written with full precision by json (exact round trip)
        return {'dtype': value.dtype.str, 'shape': list(value.shape), 'data': value.ravel().tolist()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value):
    if isinstance(value, dict) and set(value) == {'dtype', 'shape', 'data'}:
        dtype = str if value['dtype'] == 'str' else np.dtype(value['dtype'])
        return np.array(value['data'], dtype=dtype).reshape(value['shape'])
    return value


def _neureval_version():
    try:
        return version('neureval')
    except PackageNotFoundError:
        return 'unknown'
