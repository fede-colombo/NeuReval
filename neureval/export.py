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
- numbers are written with full precision, so a loaded model gives exactly the same
  results as the original one;
- a check after loading (results on synthetic points saved in the file) detects a version
  of scikit-learn that rebuilds a model differently.

Only models whose fitted parameters are aggregated can be saved. There is no fixed list of
classifiers or preprocessing methods: any scikit-learn estimator is checked automatically
(see "Models that can be saved" below), and models that keep individual training data inside
(e.g. SVM with its support vectors, k-nearest neighbours) or that cannot be rebuilt from their
parameters (e.g. random forests) are refused.

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
from sklearn.metrics import zero_one_loss
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
# Models that can be saved
# ----------------------------------------------------------------------
# NeuReval classes are saved with an explicit list of fitted attributes.
_NEUREVAL = {
    'ComBat': {
        'cls': ComBat,
        'attributes': ['sites_', 'grand_mean_', 'beta_covariates_', 'var_pooled_', 'gamma_star_',
                       'delta_star_', 'calibrated_sites_', 'n_features_in_', 'covariate_names_',
                       'n_covariates_'],
    },
    'GLMCorrection': {
        'cls': GLMCorrection,
        'attributes': ['x_mean_', 'x_scale_', 'cov_mean_', 'cov_scale_', 'beta_'],
    },
}
# scikit-learn estimators are not listed one by one: any classifier, preprocessing (transformer) or
# clustering method of scikit-learn can be saved if it passes these checks.
# - When saving (model_to_dict): its settings and all its public fitted attributes (names ending
#   with '_') must be plain values (numbers, strings, arrays): models that keep other objects inside
#   (e.g. the trees of a random forest) are refused. The model is then rebuilt from what would be
#   written and must give the same results as the original on synthetic points: models that need
#   internal data that is not saved (e.g. k-nearest neighbours) are refused.
# - When exporting a pipeline (export_model), also: no fitted array may have one value per training
#   subject, and no row of a fitted array may be identical to the data of a training subject
#   (e.g. the support vectors of an SVM).
# - When loading: only scikit-learn classes can be rebuilt (never a class named in the file from
#   another package), and the rebuilt model must give the same results on the synthetic points as
#   when it was saved (the "fingerprint"), which detects a different version of scikit-learn
#   rebuilding the model differently.
# These checks catch the known ways in which a model keeps individual data, but they cannot prove
# that a model contains none (e.g. a very large neural network can memorize its training data):
# the responsibility for what is shared stays with the training centre.
_FINGERPRINT_POINTS = 20


def model_to_dict(model):
    """
    Describe a model as a dictionary of plain values (numbers, strings, lists), ready to be
    written as JSON.

    :param model: a fitted ComBat or GLMCorrection, or a scikit-learn classifier, transformer
        (e.g. PCA) or clustering method. Fitted scikit-learn models are saved with all their
        fitted attributes; models that are not fitted (e.g. a clustering method) only with their
        settings.
    :return: dictionary with the class name, constructor parameters and fitted attributes.
    :rtype: dict
    """
    name = type(model).__name__
    if name in _NEUREVAL and isinstance(model, _NEUREVAL[name]['cls']):
        attributes = _NEUREVAL[name]['attributes']
        missing = [a for a in attributes if not hasattr(model, a)]
        if missing:
            raise ValueError(f"{name} is not fitted yet (missing {missing}): call fit before saving it.")
        return {'class': name,
                'params': _plain_params(model),
                'attributes': {a: _encode(getattr(model, a), f"{name}.{a}") for a in attributes}}

    module = _sklearn_module(model)
    d = {'class': name, 'module': module, 'params': _plain_params(model),
         'attributes': {a: _encode(v, f"{name}.{a}") for a, v in _fitted_attributes(model).items()}}
    if d['attributes']:
        d['fingerprint'] = _fingerprint(model)
        # The model rebuilt from what would be saved must give the same results
        try:
            model_from_dict(json.loads(json.dumps(d, allow_nan=False)))
        except Exception as err:
            raise TypeError(f"{name} cannot be saved: it cannot be rebuilt from its saved parameters, as it "
                            f"keeps internal objects that are not saved (e.g. the training data). "
                            f"Details: {err}") from None
    return d


def model_from_dict(d, sklearn_version=None):
    """
    Rebuild a model from the dictionary written by `model_to_dict`.

    :param d: dictionary with the class name, constructor parameters and fitted attributes.
    :type d: dict
    :param sklearn_version: version of scikit-learn used to save the model, for error messages.
    :type sklearn_version: str
    :return: the model (fitted, or not fitted for clustering methods).
    """
    name = d.get('class')
    if 'module' not in d:
        if name not in _NEUREVAL:
            raise ValueError(f"Unknown model class {name!r}.")
        cls, attributes = _NEUREVAL[name]['cls'], _NEUREVAL[name]['attributes']
        missing = [a for a in attributes if a not in d['attributes']]
        if missing:
            raise ValueError(f"The saved {name} is incomplete: missing {missing}.")
    else:
        cls, attributes = _sklearn_class(d['module'], name), list(d['attributes'])
    model = cls(**d['params'])
    for a in attributes:
        setattr(model, a, _decode(d['attributes'][a]))
    if 'fingerprint' in d:
        _check_fingerprint(model, d['fingerprint'], sklearn_version)
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
    models = {key: model_from_dict(d, content.get('sklearn_version')) for key, d in content['models'].items()}
    return models, content.get('metadata', {})


# ----------------------------------------------------------------------
# Whole pipeline: training centre and external centre
# ----------------------------------------------------------------------
Export = namedtuple('Export', ['train_cllab', 'train_acc'])
Eval = namedtuple('Eval', ['train_cllab', 'train_acc', 'test_cllab', 'test_acc', 'test_classlab',
                           'test_stability'])


def export_model(path, findbestclust, modalities, covariates, nclust=None, tr_lab=None, site=None,
                 harmonize_covariates=None, metadata=None, min_cluster_size=10):
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
    the training subjects). For this reason only some methods can be exported. Any scikit-learn
    classifier, preprocessing method or clustering method is checked automatically:

    - classifier and preprocessing: all their fitted parameters are saved. They are refused if
      they keep other objects inside (e.g. the trees of a random forest), if a fitted array has
      one value per training subject, if a row of a fitted array is identical to the data of a
      training subject (e.g. the support vectors of an SVM), or if the model cannot be rebuilt
      from its saved parameters (e.g. k-nearest neighbours, UMAP). Linear models, discriminant
      analysis, naive Bayes and PCA pass these checks.
    - clustering: only its settings are saved (e.g. the number of clusters): the external centre
      runs the clustering on its own data.
    - harmonization: ComBat.

    These checks catch the known ways in which a model keeps individual data, but they cannot
    prove that a model contains none (e.g. a very large neural network can memorize its training
    data): the training centre remains responsible for what it shares.

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
    :param min_cluster_size: smallest number of training subjects allowed in a cluster, default 10.
        The parameters of a very small cluster (e.g. its mean) are close to the data of its
        subjects. Set it to 0 to skip the check.
    :type min_cluster_size: int
    :return: namedtuple with the cluster labels of the subjects (train_cllab, kept by the training
        centre and not saved) and the accuracy of the classifier on them (train_acc).
    :rtype: namedtuple
    """
    # Check that every method can be exported before fitting anything
    clust_method = clone(findbestclust.clust_method)
    _check_role(clust_method, 'clustering method')
    _check_role(findbestclust.class_method, 'classifier')
    preproc = clone(findbestclust.preproc_method) if findbestclust.preproc_method is not None else None
    if preproc is not None:
        _check_role(preproc, 'preprocessing method')
    harmonizer = _resolve_harmonizer(findbestclust.harmonization, site, harmonize_covariates)
    if harmonizer is not None and not isinstance(harmonizer, ComBat):
        raise TypeError(f"{type(harmonizer).__name__} cannot be exported as harmonization method: only ComBat.")

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
        X_corrected = X_train
        X_train = preproc.fit(X_corrected).transform(X_corrected)
        _check_aggregated(preproc, X_corrected, 'preprocessing method')
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
    sizes = pd.Series(labels[labels >= 0]).value_counts().sort_index()
    small = sizes[sizes < (min_cluster_size or 0)]
    if len(small) > 0:
        raise ValueError(f"Cluster(s) {small.index.tolist()} have fewer than {min_cluster_size} training "
                         f"subjects ({small.tolist()}): their parameters would be close to the data of "
                         f"individual subjects. Use fewer clusters, or lower min_cluster_size if this is "
                         f"acceptable.")
    classifier = clone(findbestclust.class_method).fit(X_train, labels)
    train_acc = 1 - zero_one_loss(labels, classifier.predict(X_train))
    np.random.seed(0)   # same random labels as in evaluate_confounds
    random_labels = [np.random.permutation(labels) for _ in range(findbestclust.nrand)]
    models['classifier'] = classifier
    for i, rnd in enumerate(random_labels):
        models[f'random_classifier/{i}'] = clone(findbestclust.class_method).fit(X_train, rnd)
    for key in ['classifier'] + [f'random_classifier/{i}' for i in range(findbestclust.nrand)]:
        _check_aggregated(models[key], X_train, 'classifier')
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
def _sklearn_module(model):
    """Module of a scikit-learn estimator (the only external classes that can be saved)."""
    from sklearn.base import BaseEstimator
    module = type(model).__module__
    if not (isinstance(model, BaseEstimator) and module.split('.')[0] == 'sklearn'):
        raise TypeError(f"{type(model).__name__} cannot be saved: only ComBat, GLMCorrection and "
                        f"scikit-learn estimators can be saved.")
    return module


def _sklearn_class(module, name):
    """
    scikit-learn class named in a model file. Only scikit-learn classifiers, transformers and
    clustering methods are accepted, so that a file cannot make NeuReval build another object.
    """
    import importlib
    from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, TransformerMixin
    from sklearn.mixture._base import BaseMixture
    if not isinstance(module, str) or module.split('.')[0] != 'sklearn':
        raise ValueError(f"Model class {name!r} from {module!r} refused: only scikit-learn classes can be "
                         f"loaded.")
    cls = getattr(importlib.import_module(module), str(name), None)
    if not (isinstance(cls, type) and issubclass(cls, BaseEstimator)
            and issubclass(cls, (ClassifierMixin, TransformerMixin, ClusterMixin, BaseMixture))):
        raise ValueError(f"Model class {name!r} from {module!r} refused: only scikit-learn classifiers, "
                         f"transformers and clustering methods can be loaded.")
    return cls


def _check_role(model, role):
    """The method must be a scikit-learn estimator of the right kind, with settings that can be saved."""
    from sklearn.base import is_classifier
    name = type(model).__name__
    _sklearn_module(model)
    needed = {'classifier': 'predict', 'preprocessing method': 'transform', 'clustering method': 'fit_predict'}
    if not hasattr(model, needed[role]) or (role == 'classifier' and not is_classifier(model)):
        raise TypeError(f"{name} cannot be exported as {role}: it has no {needed[role]} method.")
    _plain_params(model)


def _plain_params(model):
    params = model.get_params(deep=False) if hasattr(model, 'get_params') else {}
    for key, value in params.items():
        if not _is_plain(value):
            raise TypeError(f"Parameter {key}={value!r} of {type(model).__name__} cannot be saved: only "
                            f"numbers, strings, booleans, None, lists and dictionaries of these can be saved.")
    return params


def _fitted_attributes(model):
    """Public fitted attributes of a scikit-learn estimator (names ending with '_')."""
    return {a: v for a, v in vars(model).items() if a.endswith('_') and not a.startswith('_')}


def _arrays(value):
    """All the arrays inside an attribute (an array, or a list of arrays)."""
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _arrays(v)


def _check_aggregated(model, X, role):
    """
    Refuse a fitted model with a fitted array that has one value per training subject, or a row
    identical to the data of a training subject (e.g. the support vectors of an SVM).
    """
    name = type(model).__name__
    X = np.ascontiguousarray(X, dtype=float)
    n_subjects, n_features = X.shape
    rows = {r.tobytes() for r in X}
    for attr, value in _fitted_attributes(model).items():
        for arr in _arrays(value):
            if n_subjects in arr.shape:
                raise TypeError(f"{name} cannot be exported as {role}: its fitted attribute {attr} has one "
                                f"value per training subject (shape {arr.shape}).")
            if arr.ndim == 2 and arr.shape[1] == n_features and arr.dtype.kind == 'f':
                if any(np.ascontiguousarray(r, dtype=float).tobytes() in rows for r in arr):
                    raise TypeError(f"{name} cannot be exported as {role}: its fitted attribute {attr} "
                                    f"contains the data of training subjects (e.g. support vectors).")


def _output_methods(model):
    """Methods NeuReval uses after loading: predict for classifiers, transform for preprocessing."""
    from sklearn.base import is_classifier
    if is_classifier(model):
        return ['predict']
    return ['transform'] if hasattr(model, 'transform') else []


def _fingerprint(model):
    """
    Results of the model on synthetic points (random numbers, no real data), for the methods that
    NeuReval uses after loading (predict for classifiers, transform for preprocessing), to check
    the model after loading it.
    """
    n_features = getattr(model, 'n_features_in_', None)
    methods = _output_methods(model)
    if n_features is None or not methods:
        return None
    points = np.random.default_rng(0).normal(size=(_FINGERPRINT_POINTS, int(n_features)))
    outputs = {m: _encode(np.asarray(getattr(model, m)(points))) for m in methods}
    return {'points': _encode(points), 'outputs': outputs}


def _check_fingerprint(model, fingerprint, sklearn_version=None):
    if fingerprint is None:
        return
    points = _decode(fingerprint['points'])
    for method, saved in fingerprint['outputs'].items():
        expected = _decode(saved)
        try:
            output = np.asarray(getattr(model, method)(points))
            same = output.shape == expected.shape and (
                np.allclose(output, expected, rtol=1e-6, atol=1e-8) if expected.dtype.kind == 'f'
                else np.array_equal(output, expected))
        except Exception:
            same = False
        if not same:
            raise ValueError(f"The loaded {type(model).__name__} does not give the same results as when it was "
                             f"saved ({method}; scikit-learn {sklearn_version or 'unknown'} when saved, "
                             f"{_package_version('scikit-learn')} now). Install the scikit-learn version used "
                             f"to save the model.")


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


def _encode(value, name='value'):
    """Arrays become {'dtype', 'shape', 'data'}; lists are encoded element by element."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'USO':
            return {'dtype': 'str', 'shape': list(value.shape), 'data': [str(v) for v in value.ravel()]}
        if value.dtype.kind not in 'biuf':
            raise TypeError(f"{name}: arrays of type {value.dtype} cannot be saved.")
        if value.dtype.kind == 'f' and not np.all(np.isfinite(value)):
            raise TypeError(f"{name} contains missing or infinite values and cannot be saved.")
        # Python floats are written with full precision by json (exact round trip)
        return {'dtype': value.dtype.str, 'shape': list(value.shape), 'data': value.ravel().tolist()}
    if isinstance(value, (list, tuple)):
        return [_encode(v, name) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): _encode(v, name) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise TypeError(f"{name} is not finite ({value}) and cannot be saved.")
        return value
    raise TypeError(f"{name} is a {type(value).__name__} object and cannot be saved: only numbers, strings "
                    f"and arrays can be saved (models that keep other objects inside, e.g. the trees of a "
                    f"random forest, cannot be exported).")


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