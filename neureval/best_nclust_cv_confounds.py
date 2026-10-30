"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, 
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from neureval.relative_validation_confounds import RelativeValidationConfounds
from collections import namedtuple
from scipy import stats
import numpy as np
import math
import logging
import multiprocessing as mp
import itertools
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import clone
import pandas as pd
import statistics

logging.basicConfig(format='%(asctime)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


def _prepare_folds(reval, data, modalities, covariates, nfold, iter_cv=1, strat_vect=None, n_jobs=1):
    """
    Generate the (repeated) cross-validation folds and prepare the data of each
    fold once (confound correction and, if specified, preprocessing) with
    :meth:`neureval.relative_validation_confounds.RelativeValidationConfounds._prepare_fold`.

    :param reval: class used to prepare the folds (its preprocessing method is used).
    :type reval: RelativeValidationConfounds
    :param data: dataset, used to define the folds.
    :type data: ndarray or DataFrame, (n_samples, n_features)
    :param modalities: dictionary specifying the datasets for each type of input features
    :type modalities: dictionary
    :param covariates: dictionary specifying the covariates for each type of input features
    :type covariates: dictionary
    :param nfold: number of cross-validation folds.
    :type nfold: int
    :param iter_cv: number of repeated cross-validation, default 1.
    :type iter_cv: int
    :param strat_vect: vector for stratification, default None.
    :type strat_vect: ndarray, (n_samples,)
    :param n_jobs: number of parallel processes, default 1.
    :type n_jobs: int
    :return: one tuple (train indices, validation indices, prepared training data,
        prepared validation data) for each fold.
    :rtype: list
    """
    return _prepare_folds_multi([reval], data, modalities, covariates, nfold,
                                iter_cv, strat_vect, n_jobs)[0]


def _prepare_folds_multi(revals, data, modalities, covariates, nfold, iter_cv=1,
                         strat_vect=None, n_jobs=1):
    """
    Same as :func:`_prepare_folds` for several `RelativeValidationConfounds` objects
    (e.g. one for each combination of preprocessing parameters), on the same folds.
    All (object, fold) pairs are independent and are prepared in a single parallel queue.

    :return: for each object in `revals`, the list of its prepared folds.
    :rtype: list of lists
    """
    if strat_vect is not None:
        kfold = RepeatedStratifiedKFold(n_splits=nfold, n_repeats=iter_cv, random_state=42)
    else:
        kfold = RepeatedKFold(n_splits=nfold, n_repeats=iter_cv, random_state=42)
    splits = list(kfold.split(data, strat_vect))
    jobs = [(reval, tr_idx, val_idx) for reval in revals for tr_idx, val_idx in splits]
    if n_jobs > 1 and len(jobs) > 1:
        prepared = Parallel(n_jobs=n_jobs)(
            delayed(_prepare_one)(reval, modalities, covariates, tr_idx, val_idx)
            for reval, tr_idx, val_idx in jobs)
    else:
        prepared = [_prepare_one(reval, modalities, covariates, tr_idx, val_idx)
                    for reval, tr_idx, val_idx in jobs]
    n = len(splits)
    return [prepared[i * n:(i + 1) * n] for i in range(len(revals))]


def _prepare_one(reval, modalities, covariates, tr_idx, val_idx):
    """Prepare one fold: (train indices, validation indices, prepared train, prepared validation)."""
    return (tr_idx, val_idx) + reval._prepare_fold(modalities, covariates, tr_idx, val_idx)


class FindBestClustCVConfounds(RelativeValidationConfounds):
    """
    Child class of :class:`reval.relative_validation.RelativeValidationConfounds`.
    It performs (repeated) k-fold cross validation on the training set to
    select the best number of clusters, i.e., the number that minimizes the
    normalized stability (i.e., average misclassification
    error/asymptotic misclassification rate).

    :param nfold: number of CV folds.
    :type nfold: int
    :param nclust_range: list with clusters to look for, default None.
    :type nclust_range: list of int
    :param s: classification object inherited
        from :class:`neureval.relative_validation.RelativeValidationConfounds`.
    :type s: class
    :param c: clustering object inherited
        from :class:`neureval.relative_validation.RelativeValidationConfounds`.
    :type c: class
    :param preprocessing: preprocessing method inherited
        from :class:`neureval.relative_validation.RelativeValidationConfounds`.
    :type preprocessing: class
    :param nrand: number of random labeling iterations
        to compute asymptotic misclassification rate, inherited from
        :class:`neureval.relative_validation.RelativeValidation` class.
    :type nrand: int
    :param n_jobs: number of processes to be run in parallel, default 1.
        Negative values follow the joblib convention: -1 uses all CPUs, -2 all CPUs but one.
    :type n_jobs: int

    :attribute: cv_results_ dataframe with cross validation results. Columns are
        ncl = number of clusters; ms_tr = misclassification training;
        ms_val = misclassification validation.
    """

    def __init__(self, c, s, preprocessing, nrand, nfold=2, n_jobs=1, nclust_range=None):
        """
        Construct method.
        """
        super().__init__(c, s, preprocessing, nrand)
        self.nfold = nfold
        self.n_jobs = _check_n_jobs(n_jobs)
        self.nclust_range = nclust_range

    def best_nclust_confounds(self, data, modalities, covariates, iter_cv=1, strat_vect=None,
                              prepared_folds=None):
        """
        This method takes as input the training dataset and the
        stratification vector (if available) and performs a
        (repeated) CV procedure to select the best number of clusters that minimizes
        normalized stability.

        :param data: training dataset.
        :type data: ndarray, (n_samples, n_features)
        :param modalities: dictionary specifying the datasets for each type of input features
        :type modalities: dictionary
        :param covariates: dictionary specifying the covariates for each type of input features
        :type covatìriates: dictionary
        :param iter_cv: number of iteration for repeated CV, default 1.
        :type iter_cv: integer
        :param strat_vect: vector for stratification, defaults to None.
        :type strat_vect: ndarray, (n_samples,)
        :param prepared_folds: internal use, default None. Folds already prepared with :func:`_prepare_folds`.
            Used by :class:`neureval.param_selection_confounds.ParamSelectionConfounds` to
            prepare the folds once for all parameter combinations. If None, folds are
            prepared here.
        :type prepared_folds: list
        :return: CV metrics for training and validation sets, best number of clusters,
            misclassification errors at each CV iteration.
        :rtype: dictionary, int, (list) if n_clusters parameter is not available
        """
        
        reval = RelativeValidationConfounds(self.clust_method, self.class_method, self.preproc_method, self.nrand)

        # Confound correction and preprocessing depend only on the fold:
        # compute them once per fold and reuse them for every number of clusters.
        if prepared_folds is None:
            prepared_folds = _prepare_folds(reval, data, modalities, covariates,
                                           self.nfold, iter_cv, strat_vect, self.n_jobs)
        params = self._make_tasks(reval, prepared_folds)
        out = _run_tasks(params, self.n_jobs)
        return self._aggregate_results(params, out)

    def _make_tasks(self, reval, prepared_folds):
        """
        Build the list of independent tasks: one for each fold and number of clusters
        (or one for each fold for clustering methods without a number of clusters).

        :param reval: `neureval.RelativeValidationConfounds` class used by the tasks.
        :type reval: RelativeValidationConfounds
        :param prepared_folds: folds prepared with :func:`_prepare_folds`.
        :type prepared_folds: list
        :return: list of tasks, each a tuple (reval, fold) or (reval, fold, number of clusters).
        :rtype: list
        """
        if self.nclust_range is not None and 'n_clusters' in self.clust_method.get_params().keys():
            return list(itertools.product([reval], prepared_folds, self.nclust_range))
        elif self.nclust_range is not None and 'n_components' in self.clust_method.get_params().keys():
            return list(itertools.product([reval], prepared_folds, self.nclust_range))
        else:
            return list(itertools.product([reval], prepared_folds))

    def _aggregate_results(self, params, out):
        """
        Combine the results of the tasks into CV metrics, best number of clusters and labels.

        :param params: tasks, as returned by :meth:`_make_tasks`.
        :type params: list
        :param out: results of the tasks, in the same order as `params`.
        :type out: list
        :return: CV metrics, best number of clusters, clustering labels; None if no
            clusters were found.
        :rtype: dictionary, int, list
        """
        # return dataframe attribute (cv_results_) with cv scores
        # If no point are labeled (e.g., all points assigned to -1 class by HDBSCAN)
        # the method returns
        cv_results_ = pd.DataFrame(out,
                                   columns=['ncl', 'ms_tr', 'ms_val', 'tr_labels', 'val_labels', 'val_classlab'])
        ctrl_rows = cv_results_.shape[0]
        cv_results_ = cv_results_.dropna(axis=0)
        if 0 < ctrl_rows - cv_results_.shape[0] < ctrl_rows:
            logging.info("Dropped results where clustering algorithm failed to identify clusters.")
            FindBestClustCVConfounds.cv_results_ = cv_results_
        elif ctrl_rows - cv_results_.shape[0] == 0:
            FindBestClustCVConfounds.cv_results_ = cv_results_
        else:
            logging.info(f"{self.clust_method} was not able to identify any cluster. Failed run with "
                         f"{self.class_method}.")
            return None

        
        metrics = {'train': {}, 'val': {}}
        for ncl in cv_results_.ncl.unique():
            norm_stab_tr = cv_results_.loc[cv_results_.ncl == ncl]['ms_tr']
            norm_stab_val = cv_results_.loc[cv_results_.ncl == ncl]['ms_val']
            metrics['train'][ncl] = (np.mean(norm_stab_tr), _confint(norm_stab_tr))
            metrics['val'][ncl] = (np.mean(norm_stab_val), _confint(norm_stab_val))

        val_score = np.array([val[0] for val in metrics['val'].values()])
        bestscore = min(val_score)
        # select the cluster with the minimum misclassification error
        # and the maximum number of clusters
        if self.nclust_range is not None and 'n_clusters' in self.clust_method.get_params().keys():
            bestncl = self.nclust_range[np.flatnonzero(val_score == bestscore)[-1]]
            # To comment
            best_idx = cv_results_.loc[cv_results_.ncl == bestncl].ms_val.idxmin()
            idx_vect = np.concatenate((params[best_idx][1][0], params[best_idx][1][1]))
            label_vect = np.concatenate((out[best_idx][3], out[best_idx][4]))
            tr_lab = [lab for _, lab in sorted(zip(idx_vect, label_vect))]
            return metrics, bestncl, tr_lab
        elif self.nclust_range is not None and 'n_components' in self.clust_method.get_params().keys():
            bestncl = self.nclust_range[np.flatnonzero(val_score == bestscore)[-1]]
            # To comment
            best_idx = cv_results_.loc[cv_results_.ncl == bestncl].ms_val.idxmin()
            idx_vect = np.concatenate((params[best_idx][1][0], params[best_idx][1][1]))
            label_vect = np.concatenate((out[best_idx][3], out[best_idx][4]))
            tr_lab = [lab for _, lab in sorted(zip(idx_vect, label_vect))]
            return metrics, bestncl, tr_lab
        else:
            bestncl = list(metrics['val'].keys())[np.flatnonzero(val_score == bestscore)[-1]]
            best_idx = cv_results_.loc[cv_results_.ncl == bestncl].ms_val.idxmin()
            idx_vect = np.concatenate((params[best_idx][1][0], params[best_idx][1][1]))
            label_vect = np.concatenate((out[best_idx][3], out[best_idx][4]))
            tr_lab = [lab for _, lab in sorted(zip(idx_vect, label_vect))]
            return metrics, bestncl, tr_lab
    
    def evaluate_confounds(self, data, modalities, covariates, tr_idx, val_idx, nclust=None, tr_lab=None):
        """
        External validation: tests whether the clustering solution found in the training
        set replicates in a validation (hold-out or external) set.

        Covariates are regressed out and preprocessing (if any) is fitted on the training
        set only, then applied to the validation set. A classifier is trained on the
        training clusters; the validation set is clustered independently and the two
        labelings are compared.

        :param data: dataset with both training and validation subjects.
        :type data: ndarray or DataFrame, (n_samples, n_features)
        :param modalities: dictionary specifying the datasets for each type of input feature
        :type modalities: dictionary
        :param covariates: dictionary specifying the covariates for each type of input feature
        :type covariates: dictionary
        :param tr_idx: training subjects, as integer row positions (in any order) or as a
            boolean mask with one value per subject.
        :type tr_idx: array-like
        :param val_idx: validation subjects, as integer row positions or as a boolean mask.
            Must not overlap with `tr_idx`.
        :type val_idx: array-like
        :param nclust: number of clusters. Required for clustering methods with an
            `n_clusters` or `n_components` parameter. Default None.
        :type nclust: int
        :param tr_lab: clustering labels for the training set. If not None the clustering
            algorithm is not run on the training set and the classifier is fitted on these
            labels (e.g. for methods without `n_clusters`, using the labels found during
            cross-validation). Default None.
        :type tr_lab: array-like
        :return: namedtuple with fields

            - train_cllab: clustering labels of the training subjects;
            - train_acc: accuracy of the classifier on the training set;
            - test_cllab: labels of the independent clustering of the validation set,
              matched to the training clusters;
            - test_acc: agreement between `test_cllab` and the classifier predictions
              (raw accuracy, its chance level depends on the number of clusters);
            - test_classlab: training cluster assigned by the classifier to each
              validation subject. Use these labels to assign new subjects to the
              training clusters;
            - test_stability: normalized stability on the validation set, i.e.
              (1 - test_acc) divided by the error obtained with random training labels,
              as in cross-validation: 0 means perfect replication, values close to 1
              mean no better than chance.

            Returns None if the clustering algorithm finds no clusters.
        :rtype: namedtuple
        """
        n_samples = _n_subjects(modalities, covariates)
        tr_idx = _check_indices(tr_idx, n_samples, 'tr_idx')
        val_idx = _check_indices(val_idx, n_samples, 'val_idx')
        overlap = np.intersect1d(tr_idx, val_idx)
        if len(overlap) > 0:
            raise ValueError(f"{len(overlap)} subject(s) are in both the training and the validation set "
                             f"(row positions {overlap[:10].tolist()}{'...' if len(overlap) > 10 else ''}). "
                             f"The two sets must not overlap.")
        if tr_lab is not None:
            tr_lab = np.asarray(tr_lab)
            if len(tr_lab) != len(tr_idx):
                raise ValueError(f"tr_lab has {len(tr_lab)} labels but the training set has "
                                 f"{len(tr_idx)} subjects.")

        # Work on copies of the estimators, so that the objects passed by the user are not modified
        clust_method = clone(self.clust_method)
        clust_params = clust_method.get_params()
        for par in ('n_clusters', 'n_components'):
            if par in clust_params:
                if nclust is None:
                    raise ValueError(f"nclust is required for {type(clust_method).__name__}, "
                                     f"which needs the number of clusters ({par}).")
                clust_method.set_params(**{par: nclust})
                break
        preproc = clone(self.preproc_method) if self.preproc_method is not None else None
        reval = RelativeValidationConfounds(clust_method, clone(self.class_method), preproc, self.nrand)

        X_train_prep, X_test_prep = reval._prepare_fold(modalities, covariates, tr_idx, val_idx)
        train_res = reval._cluster_and_classify_train(X_train_prep, tr_lab)
        if train_res is None:
            logging.warning(f"Evaluation not possible: no clusters were identified in the training set "
                            f"with {clust_method}.")
            return None
        tr_misc, modelfit, labels_tr, X_train_cor = train_res
        test_res = reval._cluster_and_classify_test(X_test_prep, modelfit)
        if test_res is None:
            logging.warning(f"Evaluation not possible: no clusters were identified in the validation set "
                            f"with {clust_method}.")
            return None
        ts_misc, labels_ts, X_test_cor = test_res
        classlab_ts = modelfit.predict(X_test_cor)

        # Normalize by the error obtained with random training labels, as in cross-validation
        rnd_misc = reval.rndlabels_traineval(X_train_cor, X_test_cor, labels_tr, labels_ts)
        test_stability = ts_misc / rnd_misc if rnd_misc > 0.0 else 1.0

        Eval = namedtuple('Eval', ['train_cllab', 'train_acc', 'test_cllab', 'test_acc',
                                   'test_classlab', 'test_stability'])
        return Eval(labels_tr, 1 - tr_misc, labels_ts, 1 - ts_misc, classlab_ts, test_stability)

    @staticmethod
    def _fit_confounds(reval, fold, ncl=None):
        """
        Function that calls training, test, and random labeling on one prepared fold.

        :param reval: `neureval.RelativeValidationConfounds` class.
        :type reval: RelativeValidationConfounds
        :param fold: training indices, validation indices and the corresponding
            prepared (confound corrected and preprocessed) datasets, as returned by
            :func:`_prepare_folds`.
        :type fold: tuple
        :param ncl: number of clusters, default None
        :type ncl: int
        :return: number of clusters, misclassification error on the training set, normalized
            stability on the validation set, clustering labels of the training and validation sets
            (the latter matched to the classifier labels), classifier predictions on the validation set.
        :rtype: tuple
        """

        tr_idx, val_idx, X_train_prep, X_test_prep = fold
        if 'n_clusters' in reval.clust_method.get_params().keys():
           reval.clust_method.n_clusters = ncl
        elif 'n_components' in reval.clust_method.get_params().keys():
           reval.clust_method.n_components = ncl
   
        
        # Case in which no clusters are identified (all points labeled as noise,
        # it happens particularly for SpectralClustering and HDBSCAN): no result for this fold.
        # Other errors are not hidden.
        train_res = reval._cluster_and_classify_train(X_train_prep)
        if train_res is None:
            return None, None, None, None, None, None
        miscl_tr, modelfit, tr_labels, X_train_cor = train_res
        test_res = reval._cluster_and_classify_test(X_test_prep, modelfit)
        if test_res is None:
            return None, None, None, None, None, None
        miscl_val, val_labels, X_test_cor = test_res
        # Classifier predictions on the validation set, needed for the agreement metrics
        # (computed before random labeling, which refits the same classifier on random labels)
        val_classlab = modelfit.predict(X_test_cor)
        rndmisc_mean_val = reval.rndlabels_traineval(X_train_cor, X_test_cor, tr_labels, val_labels)

        # If random labeling gives perfect prediction, substitute
        # misclassification for validation loop with 1.0
        if rndmisc_mean_val > 0.0:
            ms_val = miscl_val / rndmisc_mean_val
        else:
            ms_val = 1.0
        return (len([n for n in np.unique(val_labels) if n >= 0]), miscl_tr, ms_val, tr_labels, val_labels,
                val_classlab)

        
def _n_subjects(modalities, covariates):
    """
    Number of subjects, checking that all modalities and covariates have the same
    number of rows.
    """
    sizes = {f'modalities[{k!r}]': len(v) for k, v in modalities.items()}
    sizes.update({f'covariates[{k!r}]': len(v) for k, v in covariates.items()})
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Modalities and covariates must have the same number of subjects (rows), "
                         f"found: {sizes}")
    return next(iter(sizes.values()))


def _check_indices(idx, n_samples, name):
    """
    Convert training/validation subjects, given as integer row positions or as a
    boolean mask, to an array of row positions, checking that they are valid.
    """
    idx = np.asarray(idx)
    if idx.size == 0:
        raise ValueError(f"{name} is empty.")
    if idx.dtype == bool:
        if idx.ndim != 1 or len(idx) != n_samples:
            raise ValueError(f"{name} is a boolean mask of length {len(idx)}, but there are "
                             f"{n_samples} subjects.")
        return np.flatnonzero(idx)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"{name} must be a list of integer row positions or a boolean mask.")
    if idx.min() < 0 or idx.max() >= n_samples:
        raise ValueError(f"{name} contains row positions outside the range 0-{n_samples - 1}.")
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"{name} contains duplicated row positions.")
    return idx


def _check_n_jobs(n_jobs):
    """
    Number of parallel processes to use, following the joblib convention for
    negative values (-1: all CPUs, -2: all CPUs but one). Values larger than the
    number of CPUs are capped, 0 or None mean 1.
    """
    if n_jobs is None or n_jobs == 0:
        return 1
    return max(1, min(effective_n_jobs(n_jobs), mp.cpu_count()))


def _run_tasks(tasks, n_jobs):
    """
    Run :meth:`FindBestClustCVConfounds._fit_confounds` on each task, in parallel
    with joblib when n_jobs > 1, otherwise sequentially in the current process.

    joblib starts its worker processes in a way that is safe with libraries using
    OpenMP threads (e.g. UMAP, numba, scikit-learn), works the same way on Linux,
    macOS and Windows, raises an error if a worker dies instead of waiting forever,
    and limits the threads used by each worker to avoid oversubscribing the CPUs.

    :param tasks: tasks, as returned by :meth:`FindBestClustCVConfounds._make_tasks`.
    :type tasks: list
    :param n_jobs: number of parallel processes.
    :type n_jobs: int
    :return: results of the tasks, in the same order as `tasks`.
    :rtype: list
    """
    if n_jobs > 1 and len(tasks) > 1:
        return Parallel(n_jobs=n_jobs)(delayed(FindBestClustCVConfounds._fit_confounds)(*t)
                                       for t in tasks)
    return [FindBestClustCVConfounds._fit_confounds(*t) for t in tasks]


def _confint(vect):
    """
    Private function to compute confidence interval.

    :param vect: performance scores.
    :type vect: array-like
    :return: mean and error.
    :rtype: tuple
    """
    mean = np.mean(vect)
    # interval = 1.96 * math.sqrt((mean * (1 - mean)) / len(vect))
    interval = stats.t.ppf(1 - (0.05 / 2), len(vect) - 1) * (np.std(vect) / math.sqrt(len(vect)))
    return mean, interval