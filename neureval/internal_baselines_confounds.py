
"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience,
         IRCCS San Raffaele Scientific Institute, Milan, Italy

Internal measures (e.g. silhouette, Davies-Bouldin, AIC/BIC) to select the number of
clusters, for comparison with NeuReval. Internal measures use the whole dataset (no
cross-validation): data are prepared once on all subjects, in the same way as in NeuReval
(optional site harmonization, confound correction, concatenation of the modalities and
optional preprocessing), and clustered for each number of clusters.
"""

import numpy as np
from sklearn.base import clone

from neureval.relative_validation_confounds import RelativeValidationConfounds
from neureval.harmonization import (_resolve_harmonizer, _check_harmonization_inputs,
                                    harmonize_modalities)


def _prepare_data(modalities, covariates, preprocessing=None, site=None, harmonize_covariates=None,
                  harmonization=None):
    """
    Prepare the whole dataset for the internal measures: site harmonization (if a site is
    given), confound correction, concatenation of the modalities and preprocessing (if
    specified). Harmonization, confound correction and preprocessing are estimated on all
    subjects, as the internal measures are computed on all subjects.

    :return: prepared data, (n_samples, n_features).
    :rtype: ndarray
    """
    n_subjects = {len(v) for v in modalities.values()} | {len(v) for v in covariates.values()}
    if len(n_subjects) != 1:
        raise ValueError("Modalities and covariates must have the same number of subjects (rows).")
    n = n_subjects.pop()
    all_idx = np.arange(n)
    _check_harmonization_inputs(n, modalities, site, harmonize_covariates)
    harmonizer = _resolve_harmonizer(harmonization, site, harmonize_covariates)
    if harmonizer is not None:
        modalities = harmonize_modalities(harmonizer, modalities, site, harmonize_covariates, all_idx, all_idx)
    for mod in modalities:
        if not np.all(np.isfinite(np.asarray(covariates[mod], dtype=float))):
            raise ValueError(f"Covariates of modality {mod!r} contain missing or infinite values.")
    # Same confound correction as in NeuReval, estimated on all subjects
    corrected, _ = RelativeValidationConfounds(None, None).GLMcorrection_confounds(
        modalities, covariates, all_idx, all_idx)
    corr_data = np.concatenate([np.array(corrected[mod]) for mod in corrected], axis=1)
    if preprocessing is not None:
        corr_data = clone(preprocessing).fit_transform(corr_data)
    return corr_data


def _set_nclust(c, ncl):
    """Set the number of clusters of a clustering method (n_clusters or n_components)."""
    if 'n_clusters' in c.get_params().keys():
        c.set_params(n_clusters=ncl)
    else:
        c.set_params(n_components=ncl)


def select_best(data, modalities, covariates, c, int_measure, preprocessing=None, select='max', nclust_range=None,
                site=None, harmonize_covariates=None, harmonization=None):
    """
    Select the best number of clusters that minimizes/maximizes
    the internal measure selected.

    :param data: dataset.
    :type data: array-like
    :param modalities: dictionary specifying the datasets for each type of input features
    :type modalities: dictionary
    :param covariates: dictionary specifying the covariates for each type of input features
    :type covariates: dictionary
    :param c: clustering algorithm class. Not modified: a copy is used.
    :type c: obj
    :param int_measure: internal measure function.
    :type int_measure: obj
    :param preprocessing: data reduction algorithm class, default None.
    :type preprocessing: obj
    :param select: it can be 'min', if the internal measure is to be minimized
        or 'max' if the internal measure should be maximized.
    :type select: str
    :param nclust_range: Range of clusters to consider, default None.
    :type nclust_range: list
    :param site: site of each subject. If given, the data are harmonized across sites (ComBat
        by default, estimated on all subjects) before the confound correction, as in NeuReval.
        Use the same setting as in the NeuReval analysis, so that the comparison is fair.
        Default None.
    :type site: array-like
    :param harmonize_covariates: covariates whose effects the harmonization must preserve:
        one table for all modalities, or a dictionary with one table per modality. Default None.
    :type harmonize_covariates: DataFrame, dictionary or None
    :param harmonization: harmonization method, e.g. neureval.harmonization.ComBat(); default
        ComBat with default settings if a site is given.
    :type harmonization: obj
    :return: internal score, best number of clusters and clustering labels.
    :rtype: float, int, ndarray
    """
    corr_data = _prepare_data(modalities, covariates, preprocessing, site, harmonize_covariates, harmonization)
    c = clone(c)

    if nclust_range is not None:
        scores = []
        label_vect = []
        for ncl in nclust_range:
            _set_nclust(c, ncl)
            label = c.fit_predict(corr_data)
            scores.append(int_measure(corr_data, label))
            label_vect.append(label)
    else:
        label = c.fit_predict(corr_data)
        best = int_measure(corr_data, label)
        return best, len([lab for lab in np.unique(label) if lab >= 0]), label

    if select == 'max':
        best = np.where(np.array(scores) == max(scores))[0]
    elif select == 'min':
        best = np.where(np.array(scores) == min(scores))[0]
    else:
        raise ValueError(f"select must be 'max' or 'min', got {select!r}.")
    if len(set(label_vect[int(max(best))])) == nclust_range[int(max(best))]:
        return scores[int(max(best))], nclust_range[int(max(best))], label_vect[int(max(best))]
    else:
        return scores[int(max(best))], len(set(label_vect[int(max(best))])), label_vect[int(max(best))]


def select_best_bic_aic(data, modalities, covariates, c, preprocessing=None, score='bic', nclust_range=None,
                        site=None, harmonize_covariates=None, harmonization=None):
    """
    Function that selects the best number of clusters that minimizes BIC and AIC
    in Gaussian Mixture Models.

    :param data: dataset.
    :type data: array-like
    :param modalities: dictionary specifying the datasets for each type of input features
    :type modalities: dictionary
    :param covariates: dictionary specifying the covariates for each type of input features
    :type covariates: dictionary
    :param c: Gaussian mixture model class. Not modified: a copy is used.
    :type c: obj
    :param preprocessing: data reduction algorithm class, default None.
    :type preprocessing: obj
    :param score: type of score to compute. It could be 'bic' for BIC score, 'aic' for AIC score
    :type score: str
    :param nclust_range: Range of clusters to consider (required).
    :type nclust_range: list
    :param site: site of each subject, for harmonization (see :func:`select_best`). Default None.
    :type site: array-like
    :param harmonize_covariates: covariates whose effects the harmonization must preserve.
    :type harmonize_covariates: DataFrame, dictionary or None
    :param harmonization: harmonization method, default ComBat if a site is given.
    :type harmonization: obj
    :return: BIC or AIC score, best number of clusters and clustering labels.
    :rtype: float, int, ndarray
    """
    if nclust_range is None:
        raise ValueError("nclust_range is required: list of numbers of components to compare.")
    if score not in ('bic', 'aic'):
        raise ValueError(f"score must be 'bic' or 'aic', got {score!r}.")
    corr_data = _prepare_data(modalities, covariates, preprocessing, site, harmonize_covariates, harmonization)
    c = clone(c)

    scores = []
    label_vect = []
    for components in nclust_range:
        c.set_params(n_components=components)
        label = c.fit_predict(corr_data)
        scores.append(c.bic(corr_data) if score == 'bic' else c.aic(corr_data))
        label_vect.append(label)

    best = np.where(np.array(scores) == min(scores))[0]
    return scores[int(max(best))], len(set(label_vect[int(max(best))])), label_vect[int(max(best))]


def evaluate_best(data, modalities, covariates, c, int_measure, ncl=None, preprocessing=None,
                  site=None, harmonize_covariates=None, harmonization=None):
    """
    Function that, given a number of clusters, returns the corresponding internal measure
    for a dataset.

    :param data: dataset.
    :type data: array-like
    :param modalities: dictionary specifying the datasets for each type of input features
    :type modalities: dictionary
    :param covariates: dictionary specifying the covariates for each type of input features
    :type covariates: dictionary
    :param c: clustering algorithm class. Not modified: a copy is used.
    :type c: obj
    :param int_measure: internal measure function.
    :type int_measure: obj
    :param ncl: number of clusters. Required for clustering methods with an `n_clusters`
        or `n_components` parameter.
    :type ncl: int
    :param preprocessing: dimensionality reduction algorithm class, default None.
    :type preprocessing: obj
    :param site: site of each subject, for harmonization (see :func:`select_best`). Default None.
    :type site: array-like
    :param harmonize_covariates: covariates whose effects the harmonization must preserve.
    :type harmonize_covariates: DataFrame, dictionary or None
    :param harmonization: harmonization method, default ComBat if a site is given.
    :type harmonization: obj
    :return: internal score.
    :rtype: float
    """
    corr_data = _prepare_data(modalities, covariates, preprocessing, site, harmonize_covariates, harmonization)
    c = clone(c)
    params = c.get_params().keys()
    if 'n_clusters' in params or 'n_components' in params:
        if ncl is None:
            raise ValueError(f"ncl is required for {type(c).__name__}, which needs the number of clusters.")
        _set_nclust(c, ncl)
    label = c.fit_predict(corr_data)
    return int_measure(corr_data, label)
