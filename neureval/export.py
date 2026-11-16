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

Main functions:

- `export_model` (training centre): fits the final model on the training subjects and saves it;
- `load_model` (external centre): loads it, then `evaluate` tests whether the training clusters
  replicate in the local data (same results as `evaluate_confounds` with both datasets in one
  place) and `predict` assigns local subjects to the training clusters.

`save_models` and `load_models` save and load single fitted models (e.g. ComBat).
"""
import copy
import json
import logging
from collections import namedtuple
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from neureval.best_nclust_cv_confounds import _check_groups, _check_groups_disjoint, _check_indices, _n_subjects
from neureval.harmonization import ComBat, _check_harmonization_inputs, _resolve_harmonizer, _rows
from neureval.utils import kuhn_munkres_algorithm

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
# For each class: the fitted attributes to save (constructor parameters are always saved).
# Everything else (and any class not listed here) is never written to the file.
# - fitted models: only aggregated parameters (means, variances, coefficients);
# - clustering methods: only their settings (e.g. number of clusters), as the external centre
#   runs the clustering again on its own data.
_SUPPORTED = {
    'ComBat': {
        'cls': ComBat, 'fitted': True,
        'attributes': ['sites_', 'grand_mean_', 'beta_covariates_', 'var_pooled_', 'gamma_star_',
                       'delta_star_', 'calibrated_sites_', 'n_features_in_', 'covariate_names_',
                       'n_covariates_'],
    },
    'GLMCorrection': {
        'cls': GLMCorrection, 'fitted': True,
        'attributes': ['x_mean_', 'x_scale_', 'cov_mean_', 'cov_scale_', 'beta_'],
    },
    # Preprocessing
    'PCA': {
        'cls': PCA, 'fitted': True,
        'attributes': ['components_', 'mean_', 'explained_variance_', 'explained_variance_ratio_',
                       'n_components_', 'n_features_in_'],
    },
    # Classifiers
    'LogisticRegression': {
        'cls': LogisticRegression, 'fitted': True,
        'attributes': ['classes_', 'coef_', 'intercept_', 'n_features_in_'],
    },
    'LinearDiscriminantAnalysis': {
        'cls': LinearDiscriminantAnalysis, 'fitted': True,
        'attributes': ['classes_', 'coef_', 'intercept_', 'means_', 'priors_', 'n_features_in_'],
    },
    'QuadraticDiscriminantAnalysis': {
        'cls': QuadraticDiscriminantAnalysis, 'fitted': True,
        'attributes': ['classes_', 'means_', 'priors_', 'rotations_', 'scalings_', 'n_features_in_'],
    },
    # Clustering methods (settings only)
    'GaussianMixture': {'cls': GaussianMixture, 'fitted': False, 'attributes': []},
    'KMeans': {'cls': KMeans, 'fitted': False, 'attributes': []},
    'AgglomerativeClustering': {'cls': AgglomerativeClustering, 'fitted': False, 'attributes': []},
    'HDBSCAN': {'cls': HDBSCAN, 'fitted': False, 'attributes': []},
}
_CLASSIFIERS = ['LogisticRegression', 'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis']
_PREPROCESSING = ['PCA']
_CLUSTERING = ['GaussianMixture', 'KMeans', 'AgglomerativeClustering', 'HDBSCAN']


def model_to_dict(model):
    """
    Describe a model as a dictionary of plain values (numbers, strings, lists), ready to be
    written as JSON.

    :param model: model of one of the supported classes: fitted (ComBat, GLMCorrection, PCA,
        LogisticRegression, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis) or a
        clustering method (GaussianMixture, KMeans, AgglomerativeClustering, HDBSCAN), of which
        only the settings are saved.
    :return: dictionary with the class name, constructor parameters and fitted attributes.
    :rtype: dict
    """
    name = _check_supported(model)
    spec = _SUPPORTED[name]
    missing = [a for a in spec['attributes'] if not hasattr(model, a)]
    if missing:
        raise ValueError(f"{name} is not fitted yet (missing {missing}): call fit before saving it.")
    params = model.get_params(deep=False) if hasattr(model, 'get_params') else {}
    for key, value in params.items():
        if not _is_plain(value):
            raise TypeError(f"Parameter {key}={value!r} of {name} cannot be saved: only numbers, strings, "
                            f"booleans, None, lists and dictionaries of these can be saved.")
    return {
        'class': name,
        'params': params,
        'attributes': {a: _encode(getattr(model, a)) for a in spec['attributes']},
    }


def model_from_dict(d):
    """
    Rebuild a model from the dictionary written by `model_to_dict`.

    :param d: dictionary with the class name, constructor parameters and fitted attributes.
    :type d: dict
    :return: the model (fitted, or not fitted for clustering methods).
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
        'sklearn_version': _package_version('scikit-learn'),
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
# Whole pipeline: training centre and external centre
# ----------------------------------------------------------------------
Export = namedtuple('Export', ['train_cllab', 'train_acc'])
Eval = namedtuple('Eval', ['train_cllab', 'train_acc', 'test_cllab', 'test_acc', 'test_classlab',
                           'test_stability'])


def export_model(path, findbestclust, modalities, covariates, nclust=None, tr_lab=None, site=None,
                 harmonize_covariates=None, metadata=None):
    """
    Training centre: fit the final model on all the subjects given and save it to a JSON file,
    which can be sent to an external centre instead of the data.

    The model is fitted exactly as in `FindBestClustCVConfounds.evaluate_confounds` on its
    training set: site harmonization (if `site` is given), confound correction, preprocessing
    (if any), clustering, classifier trained on the clusters, and `nrand` classifiers trained
    on random permutations of the cluster labels (needed for the normalized stability). The
    external centre then gets the same results as `evaluate_confounds` would give with both
    datasets in one place.

    Only aggregated parameters are saved, never individual data (not even the cluster labels of
    the training subjects). For this reason only some methods can be exported:

    - classifiers: LogisticRegression, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
      (SVM keeps support vectors, i.e. training subjects, and cannot be exported);
    - preprocessing: None or PCA (UMAP keeps the training data);
    - clustering: GaussianMixture, KMeans, AgglomerativeClustering, HDBSCAN (only the settings
      are saved: the external centre runs the clustering on its own data);
    - harmonization: ComBat.

    :param path: path of the file to write (e.g. 'model.json').
    :type path: str
    :param findbestclust: the FindBestClustCVConfounds object used in the analysis, with the
        optimized clustering, classifier and preprocessing methods.
    :type findbestclust: FindBestClustCVConfounds
    :param modalities: dictionary with the data of each modality, (n_subjects, n_features).
        All the subjects are used to fit the model. Use DataFrames: their column names are
        saved and checked in the external centre.
    :type modalities: dictionary
    :param covariates: dictionary with the covariates to remove for each modality.
    :type covariates: dictionary
    :param nclust: number of clusters (bestncl). Required for clustering methods with an
        `n_clusters` or `n_components` parameter.
    :type nclust: int
    :param tr_lab: cluster labels of the subjects, if the clustering is not run again (e.g. for
        methods without `n_clusters`). Default None.
    :type tr_lab: array-like
    :param site: site of each subject, for harmonization with ComBat. Default None.
    :type site: array-like
    :param harmonize_covariates: covariates whose effects ComBat must preserve: one table for all
        modalities or a dictionary with one table per modality. Default None.
    :type harmonize_covariates: DataFrame, dictionary or None
    :param metadata: other information to store (e.g. a description of the sample). Must contain
        only numbers, strings, booleans, lists and dictionaries, and no individual data.
    :type metadata: dict
    :return: namedtuple with the cluster labels of the subjects (train_cllab, kept by the training
        centre and not saved) and the accuracy of the classifier on them (train_acc).
    :rtype: namedtuple
    """
    # Check that every method can be exported before fitting anything
    clust_method = clone(findbestclust.clust_method)
    _check_supported(clust_method, _CLUSTERING, 'clustering method')
    _check_supported(findbestclust.class_method, _CLASSIFIERS, 'classifier',
                     hint=" SVM and random forests keep training subjects inside the model.")
    preproc = clone(findbestclust.preproc_method) if findbestclust.preproc_method is not None else None
    if preproc is not None:
        _check_supported(preproc, _PREPROCESSING, 'preprocessing method',
                         hint=" UMAP keeps the training data inside the model.")
    harmonizer = _resolve_harmonizer(findbestclust.harmonization, site, harmonize_covariates)
    if harmonizer is not None:
        _check_supported(harmonizer, ['ComBat'], 'harmonization method')

    n_subjects = _n_subjects(modalities, covariates)
    _check_harmonization_inputs(n_subjects, modalities, site, harmonize_covariates)
    clust_params = clust_method.get_params()
    for par in ('n_clusters', 'n_components'):
        if par in clust_params:
            if nclust is None:
                raise ValueError(f"nclust is required for {type(clust_method).__name__}, which needs the "
                                 f"number of clusters ({par}).")
            clust_method.set_params(**{par: nclust})
            break

    # Harmonization and confound correction of each modality
    models, prepared, info = {}, [], {}
    for mod in modalities:
        X = _as_2d(modalities[mod])
        if harmonizer is not None:
            hcov = harmonize_covariates[mod] if isinstance(harmonize_covariates, dict) else harmonize_covariates
            combat = clone(harmonizer).fit(X, np.asarray(site), hcov)
            X = combat.transform(X, np.asarray(site), hcov)
            models[f'harmonization/{mod}'] = combat
        glm = GLMCorrection().fit(X, covariates[mod])
        models[f'confounds/{mod}'] = glm
        prepared.append(glm.transform(X, covariates[mod]))
        info[mod] = {'features': _column_names(modalities[mod], X.shape[1]),
                     'covariates': _column_names(covariates[mod], len(glm.cov_mean_))}
    X_train = np.concatenate(prepared, axis=1)
    if preproc is not None:
        X_train = preproc.fit(X_train).transform(X_train)
        models['preprocessing'] = preproc

    # Clustering, classifier and classifiers trained on random labels (as in evaluate_confounds)
    if tr_lab is None:
        labels = clust_method.fit_predict(X_train)
    else:
        labels = np.asarray(tr_lab)
        if len(labels) != n_subjects:
            raise ValueError(f"tr_lab has {len(labels)} labels but there are {n_subjects} subjects.")
    if not np.any(labels >= 0):
        raise ValueError(f"No clusters were identified with {clust_method}: nothing to export.")
    classifier = clone(findbestclust.class_method).fit(X_train, labels)
    train_acc = 1 - zero_one_loss(labels, classifier.predict(X_train))
    np.random.seed(0)   # same random labels as in evaluate_confounds
    random_labels = [np.random.permutation(labels) for _ in range(findbestclust.nrand)]
    models['classifier'] = classifier
    for i, rnd in enumerate(random_labels):
        models[f'random_classifier/{i}'] = clone(findbestclust.class_method).fit(X_train, rnd)
    models['clustering'] = clone(clust_method)

    values, counts = np.unique(labels, return_counts=True)
    pipeline_info = {
        'modalities': info,
        'harmonization': harmonizer is not None,
        'nclust': None if nclust is None else int(nclust),
        'nrand': int(findbestclust.nrand),
        'n_subjects': int(n_subjects),
        'cluster_sizes': {str(v): int(c) for v, c in zip(values, counts)},
        'train_acc': float(train_acc),
    }
    if harmonizer is not None:
        pipeline_info['training_sites'] = [str(s) for s in np.unique(np.asarray(site))]
    save_models(path, models, metadata={'pipeline': pipeline_info, 'user': metadata or {}})
    logging.info(f"Model saved to {path}: {n_subjects} training subjects, {len(values)} clusters, "
                 f"training accuracy {train_acc:.3f}.")
    return Export(labels, train_acc)


def load_model(path):
    """
    External centre: load a model saved with `export_model`. No code is run while loading.

    :param path: path of the JSON file.
    :type path: str
    :return: the model, with the methods `evaluate` (replication of the clusters in the local
        data) and `predict` (training cluster of new subjects).
    :rtype: ExportedModel
    """
    models, metadata = load_models(path)
    if 'pipeline' not in metadata or 'classifier' not in models:
        raise ValueError(f"{path} contains models saved with save_models, not a whole pipeline saved "
                         f"with export_model: use load_models to read it.")
    return ExportedModel(models, metadata['pipeline'], metadata.get('user', {}))


class ExportedModel:
    """
    Model fitted in a training centre and loaded with `load_model`, to be evaluated on the
    data of an external centre.

    Attributes: `info` (description of the model: modalities with their features and
    covariates, number of clusters, size of the training clusters, training accuracy, ...) and
    `user_metadata` (information added by the training centre).
    """

    def __init__(self, models, info, user_metadata=None):
        self.models = models
        self.info = info
        self.user_metadata = user_metadata or {}

    def evaluate(self, modalities, covariates, site=None, harmonize_covariates=None, calibration_idx=None,
                 groups=None):
        """
        Test whether the training clusters replicate in the local data, as
        `FindBestClustCVConfounds.evaluate_confounds` does with both datasets in one place: the
        local subjects are clustered independently, assigned to the training clusters by the
        classifier, and the two labelings are compared.

        :param modalities: dictionary with the local data of each modality, with the same
            modalities and features (same columns, same order) as in the training centre.
        :type modalities: dictionary
        :param covariates: dictionary with the covariates of each modality, same columns as in
            the training centre.
        :type covariates: dictionary
        :param site: site of each subject, required if the model includes harmonization.
        :type site: array-like
        :param harmonize_covariates: covariates whose effects ComBat must preserve, same columns
            as in the training centre.
        :type harmonize_covariates: DataFrame, dictionary or None
        :param calibration_idx: subjects of the local sites (new for the model) used only to
            calibrate the harmonization, and excluded from the evaluation (integer row positions
            or boolean mask). They should be comparable to the training population (e.g. healthy
            controls in both). Required for sites that were not in the training centre.
        :type calibration_idx: array-like
        :param groups: group of each row (e.g. subject ID) for data with several rows per
            subject: an error is raised if a subject has rows in both the calibration and the
            evaluated set. Default None.
        :type groups: array-like
        :return: namedtuple with the same fields as `evaluate_confounds`: train_cllab (None, the
            training labels are not shared), train_acc, test_cllab, test_acc, test_classlab and
            test_stability, for the local subjects not used for calibration. None if the
            clustering finds no clusters in the local data.
        :rtype: namedtuple
        """
        X, _ = self._prepare(modalities, covariates, site, harmonize_covariates, calibration_idx, groups)
        clust_labels = self.models['clustering'].fit_predict(X)
        if not np.any(clust_labels >= 0):
            logging.warning("Evaluation not possible: no clusters were identified in the local data.")
            return None
        classlab = self.models['classifier'].predict(X)
        matched = kuhn_munkres_algorithm(np.int32(classlab), np.int32(clust_labels))
        misclass = zero_one_loss(classlab, matched)
        # Error with classifiers trained on random labels, as in cross-validation
        rnd_errors = []
        for i in range(self.info['nrand']):
            pred = self.models[f'random_classifier/{i}'].predict(X)
            rnd_errors.append(zero_one_loss(pred, kuhn_munkres_algorithm(np.int32(pred), np.int32(matched))))
        rnd_misc = np.mean(rnd_errors)
        test_stability = misclass / rnd_misc if rnd_misc > 0.0 else 1.0
        return Eval(None, self.info['train_acc'], matched, 1 - misclass, classlab, test_stability)

    def predict(self, modalities, covariates, site=None, harmonize_covariates=None, calibration_idx=None,
                groups=None):
        """
        Assign local subjects to the training clusters with the classifier, without evaluating
        the replication. Same inputs as `evaluate`.

        :return: training cluster of each local subject not used for calibration.
        :rtype: ndarray
        """
        X, _ = self._prepare(modalities, covariates, site, harmonize_covariates, calibration_idx, groups)
        return self.models['classifier'].predict(X)

    def _prepare(self, modalities, covariates, site, harmonize_covariates, calibration_idx, groups):
        """Harmonization, confound correction and preprocessing of the local subjects."""
        expected = list(self.info['modalities'])
        if list(modalities) != expected or list(covariates) != expected:
            raise ValueError(f"The model expects the modalities {expected} (same keys, same order) in both "
                             f"modalities and covariates, got {list(modalities)} and {list(covariates)}.")
        n_subjects = _n_subjects(modalities, covariates)
        harmonized = self.info['harmonization']
        if harmonized and site is None:
            raise ValueError("The model includes site harmonization: give the site of each subject (site=...).")
        if not harmonized and (site is not None or calibration_idx is not None):
            raise ValueError("The model was fitted without site harmonization: site and calibration_idx "
                             "cannot be used.")
        if harmonized:
            _check_harmonization_inputs(n_subjects, modalities, site, harmonize_covariates)
            site = np.asarray(site)

        cal_idx = np.array([], dtype=int)
        if calibration_idx is not None:
            cal_idx = _check_indices(calibration_idx, n_subjects, 'calibration_idx')
        eval_idx = np.setdiff1d(np.arange(n_subjects), cal_idx)
        if len(eval_idx) == 0:
            raise ValueError("All the subjects are calibration subjects: no subjects left to evaluate.")
        if groups is not None and len(cal_idx) > 0:
            groups = _check_groups(groups, n_subjects)
            _check_groups_disjoint(groups, cal_idx, eval_idx, 'evaluated')

        prepared = []
        for mod in expected:
            names = self.info['modalities'][mod]
            _check_columns(modalities[mod], names['features'], f"modalities[{mod!r}]")
            _check_columns(covariates[mod], names['covariates'], f"covariates[{mod!r}]")
            X = _as_2d(modalities[mod])
            if harmonized:
                hcov = harmonize_covariates[mod] if isinstance(harmonize_covariates, dict) else harmonize_covariates
                combat = copy.deepcopy(self.models[f'harmonization/{mod}'])   # the loaded model is not changed
                if len(cal_idx) > 0:
                    combat.calibrate(X[cal_idx], site[cal_idx], _rows(hcov, cal_idx))
                X_eval = combat.transform(X[eval_idx], site[eval_idx], _rows(hcov, eval_idx))
            else:
                X_eval = X[eval_idx]
            prepared.append(self.models[f'confounds/{mod}'].transform(X_eval, _rows(covariates[mod], eval_idx)))
        X = np.concatenate(prepared, axis=1)
        if 'preprocessing' in self.models:
            X = self.models['preprocessing'].transform(X)
        return X, eval_idx


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _check_supported(model, allowed=None, role='model', hint=''):
    """Name of the class of the model, if it can be saved (and has one of the allowed classes)."""
    name = type(model).__name__
    allowed = allowed if allowed is not None else list(_SUPPORTED)
    if name not in allowed or not isinstance(model, _SUPPORTED[name]['cls']):
        what = 'saved' if role == 'model' else f'exported as {role}'
        raise TypeError(f"{name} cannot be {what}: only methods whose fitted parameters are aggregated over "
                        f"subjects can be saved ({', '.join(allowed)}).{hint}")
    return name


def _is_plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_plain(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_plain(v) for k, v in value.items())
    return False


def _as_2d(data):
    X = np.array(data, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _column_names(table, n_columns):
    """Column names of a DataFrame (as strings), or None for arrays; with the number of columns."""
    if isinstance(table, pd.Series):
        table = table.to_frame()
    names = [str(c) for c in table.columns] if isinstance(table, pd.DataFrame) else None
    return {'names': names, 'n': int(n_columns)}


def _check_columns(table, expected, what):
    """Same columns as in the training centre (names and order if both have names, else number)."""
    if isinstance(table, pd.Series):
        table = table.to_frame()
    n = 1 if np.ndim(table) == 1 else np.shape(table)[1]
    if isinstance(table, pd.DataFrame) and expected['names'] is not None:
        names = [str(c) for c in table.columns]
        if names != expected['names']:
            missing = [c for c in expected['names'] if c not in names]
            extra = [c for c in names if c not in expected['names']]
            detail = (f"missing {missing}, unexpected {extra}" if missing or extra
                      else "same columns but in a different order")
            raise ValueError(f"{what} must have the same columns as in the training centre, in the same "
                             f"order: {detail}.")
    elif n != expected['n']:
        raise ValueError(f"{what} has {n} columns, but the model was fitted with {expected['n']}.")


def _encode(value):
    """Arrays become {'dtype', 'shape', 'data'}; lists are encoded element by element."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'USO':
            return {'dtype': 'str', 'shape': list(value.shape), 'data': [str(v) for v in value.ravel()]}
        if value.dtype.kind not in 'biuf':
            raise TypeError(f"Arrays of type {value.dtype} cannot be saved.")
        # Python floats are written with full precision by json (exact round trip)
        return {'dtype': value.dtype.str, 'shape': list(value.shape), 'data': value.ravel().tolist()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value):
    if isinstance(value, dict) and set(value) == {'dtype', 'shape', 'data'}:
        dtype = str if value['dtype'] == 'str' else np.dtype(value['dtype'])
        return np.array(value['data'], dtype=dtype).reshape(value['shape'])
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _neureval_version():
    return _package_version('neureval')


def _package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return 'unknown'