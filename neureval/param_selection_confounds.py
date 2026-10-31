"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, 
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

from neureval.relative_validation_confounds import RelativeValidationConfounds
from neureval.best_nclust_cv_confounds import (FindBestClustCVConfounds, _prepare_folds,
                                               _prepare_folds_multi, _run_tasks, _check_n_jobs,
                                               _n_subjects)
from neureval.harmonization import _check_harmonization_inputs
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
import multiprocessing as mp
import logging
import numpy as np
import itertools

from sklearn.cluster import AgglomerativeClustering

logging.basicConfig(format='%(asctime)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


class ParamSelectionConfounds(RelativeValidationConfounds):
    """
    Class that implements grid search cross-validation in parallel to select
    the best combinations of parameters for fixed classifier/clustering algorithms.
    If a preprocessing method (e.g. dimensionality reduction algorithm) is specified,
    also the specified parameters will be selected through grid-search cross-validation.

    :param params: dictionary of dictionaries of the form {'s': {classifier parameter grid},
        'c': {clustering parameter grid}, 'preprocessing': {preprocessing parameter grid}}. If one of the two dictionary of parameters is not
        available, initialize key but leave dictionary empty.
    :type params: dict
    :param cv: cross-validation folds.
    :type cv: int
    :param clust_range: list with number of clusters to investigate.
    :type clust_range: list
    :param n_jobs: number of jobs to run in parallel, default 1.
        Negative values follow the joblib convention: -1 uses all CPUs, -2 all CPUs but one.
    :type n_jobs: int
    :param iter_cv: number of repeated cv loops, default 1.
    :type iter_cv: int
    :param strat: stratification vector for cross-validation splits, default None
        (stratified by site if a site is given to `fit`).
    :type strat: numpy array
    :param harmonization: site harmonization method, e.g. neureval.harmonization.ComBat().
        Used when the site of each subject is given to `fit`; default None (ComBat with
        default settings if a site is given).
    :type harmonization: class
    :param combined_data: if True, only grey matter features will be adjusted for TIV.
        Otherwise, both grey and white matter features will be adjusted for the same set of covariates.
    :type combined_data: boolean value

    :attribute: cv_results_ cross-validation results that can be directly transformed to
        a dataframe. Key names: classifier parameters, clustering parameters,
        'best_nclust', 'mean_train_score', 'sd_train_score',
        'mean_val_score', 'sd_val_score', 'validation_meanerror'. Dictionary of lists.
    :attribute: best_param_ best solution(s) selected (minimum validation error). List.
    :attribute: best_index_ index/indices of the best solution(s). Values correspond to the
        rows of the `cv_results_` table. List.
    """

    def __init__(self, params, cv, c, s, preprocessing, nrand,
                n_jobs=1, iter_cv=1, strat=None, clust_range=None, harmonization=None):
        super().__init__(c, s, preprocessing, nrand, harmonization)
        self.params = params
        self.cv = cv
        self.iter_cv = iter_cv
        self.clust_range = clust_range
        self.n_jobs = _check_n_jobs(n_jobs)
        self.strat = strat

    def fit(self, data_tr, mod_tr, cov_tr, nclass=None, site=None, harmonize_covariates=None):
        """
        Class method that performs grid search cross-validation on training data. It
        deals with the error due to wrong parameter combinations (e.g., ward linkage
        with no euclidean affinity). If the true number of classes is know, the method
        selects both the best parameter combination that selects the true number of clusters
        (minimum stability) and the best parameter combination that minimizes
        overall stability.

        :param data_tr: training dataset.
        :type data_tr: numpy array
        :param mod_tr: dictionary specifying the dataset for each type of input feature
        :type mod_tr: dictionary
        :param cov_tr: dictionary specifying the covariates for each type of input feature
        :type cov_tr: dictionary
        :param nclass: number of true classes, default None.
        :type nclass: int
        :param site: site of each subject. If given, the data are harmonized across sites
            (ComBat by default), estimating the harmonization on the training set of each fold.
            Default None.
        :type site: array-like
        :param harmonize_covariates: covariates whose effects the harmonization must preserve
            (e.g. age, sex): one table for all modalities, or a dictionary with one table per
            modality. Default None.
        :type harmonize_covariates: DataFrame, dictionary or None
        """
        _check_harmonization_inputs(_n_subjects(mod_tr, cov_tr), mod_tr, site, harmonize_covariates)
        # Harmonization, confound correction and preprocessing do not depend on the clustering and
        # classifier parameters: prepare the cross-validation folds once (once for each
        # combination of preprocessing parameters) and share them across the grid.
        if self.preproc_method is not None:
            grid = {'c': ParameterGrid(self.params['c']), 's': ParameterGrid(self.params['s']), 'preprocessing': ParameterGrid(self.params['preprocessing'])}
            preproc_grid = list(grid['preprocessing'])
            revals = [RelativeValidationConfounds(self.clust_method, self.class_method,
                                                  clone(self.preproc_method).set_params(**param_preproc),
                                                  self.nrand, self.harmonization)
                      for param_preproc in preproc_grid]
            folds_by_preproc = _prepare_folds_multi(revals, data_tr, mod_tr, cov_tr, self.cv,
                                                    self.iter_cv, self.strat, self.n_jobs,
                                                    site, harmonize_covariates)
            new_grid = list(itertools.product(grid['c'], grid['s'], preproc_grid))
            new_params = [(ng[0], ng[1], ng[2], folds_by_preproc[preproc_grid.index(ng[2])])
                          for ng in new_grid if self._allowed_par(ng[0])]
        else:
            grid = {'c': ParameterGrid(self.params['c']), 's': ParameterGrid(self.params['s'])}
            reval = RelativeValidationConfounds(self.clust_method, self.class_method, None, self.nrand,
                                                self.harmonization)
            folds = _prepare_folds(reval, data_tr, mod_tr, cov_tr, self.cv, self.iter_cv, self.strat,
                                   self.n_jobs, site, harmonize_covariates)
            new_grid = list(itertools.product(grid['c'], grid['s']))
            new_params = [(ng[0], ng[1], None, folds) for ng in new_grid if self._allowed_par(ng[0])]

        if len(new_grid) != len(new_params):
            logging.info(f"Dropped {len(new_grid) - len(new_params)} out of {len(new_grid)} parameter "
                          f"combinations "
                          f"due to {self.clust_method} class requirements.")

        logging.info(f'Running {len(new_params)} combinations of '
                      f'parameters...\n')

        # Build the tasks of all parameter combinations (one task for each combination,
        # fold and number of clusters) and run them in a single parallel queue, so that
        # all the CPUs are kept busy even when there are few parameter combinations.
        # Each combination gets its own copy (clone) of the estimators.
        combinations, all_tasks = [], []
        for param_c, param_s, param_preproc, folds in new_params:
            c = clone(self.clust_method).set_params(**param_c)
            s = clone(self.class_method).set_params(**param_s)
            preproc = None
            if param_preproc is not None:
                preproc = clone(self.preproc_method).set_params(**param_preproc)
            findclust = FindBestClustCVConfounds(c, s, preprocessing=preproc, nrand=self.nrand,
                                                 nfold=self.cv, n_jobs=1,
                                                 nclust_range=self.clust_range)
            reval = RelativeValidationConfounds(c, s, preproc, self.nrand)
            tasks = findclust._make_tasks(reval, folds)
            combinations.append((param_c, param_s, param_preproc, findclust, tasks))
            all_tasks.extend(tasks)

        task_out = _run_tasks(all_tasks, self.n_jobs)

        perf_list, start = [], 0
        for param_c, param_s, param_preproc, findclust, tasks in combinations:
            result = findclust._aggregate_results(tasks, task_out[start:start + len(tasks)])
            start += len(tasks)
            perf_list.append(_combination_perf(param_c, param_s, param_preproc,
                                               result, findclust.cv_results_))
        out = list(zip(*perf_list))

        # cv_results_
        res_dict = _create_result_table(out)
        ParamSelectionConfounds.cv_results_ = res_dict

        # best_param_, best_index_
        val_scores = [vs for vs in res_dict['mean_val_score'] if vs is not None]
        val_idx = [idx for idx, vs in enumerate(res_dict['mean_val_score']) if vs is not None]
        if len(val_scores) > 0:
            idx_best = [val_idx[i] for i in _return_best(val_scores)]
        else:
            logging.info(f"No clustering solutions were found with any parameter combinations.")
            return self
        
    
        out_best = []
        if nclass is not None:
            logging.info(f'True number of clusters known: {nclass}\n')
            idx = np.where(np.array(res_dict['best_nclust']) == nclass)[0]
            idx_inter = set(idx).intersection(set(idx_best))
            if len(idx_inter) > 0:
                idx_best = list(idx_inter)
            else:
                if len(idx) > 0:
                    idx_true = _return_knownbest(res_dict['mean_val_score'], idx)
                    logging.info(f'Best solution(s) with true number of clusters:')
                    if self.preproc_method is not None:
                        for bidx in idx_true:
                            for k in self.params['c'].keys():
                                logging.info(f'Parameters clustering (C): {k}={res_dict[k][bidx]}')
                            for k in self.params['s'].keys():
                                logging.info(f'Parameters classifier (S): {k}={res_dict[k][bidx]}')
                            for k in self.params['preprocessing'].keys():
                                logging.info(f'Parameters preprocessing: {k}={res_dict[k][bidx]}')
                            logging.info(f'Validation performance: {res_dict["validation_meanerror"][bidx]}')
                            logging.info(f'N clusters: {res_dict["best_nclust"][bidx]}\n')
                            out_best.append([res_dict[k][bidx] for k in self.params['c'].keys()] +
                                            [res_dict[k][bidx] for k in self.params['s'].keys()] +
                                            [res_dict[k][bidx] for k in self.params['preprocessing'].keys()] +
                                            [res_dict["best_nclust"][bidx], res_dict["validation_meanerror"][bidx]])
                    else:
                        for bidx in idx_true:
                            for k in self.params['c'].keys():
                                logging.info(f'Parameters clustering (C): {k}={res_dict[k][bidx]}')
                            for k in self.params['s'].keys():
                                logging.info(f'Parameters classifier (S): {k}={res_dict[k][bidx]}')    
                            logging.info(f'Validation performance: {res_dict["validation_meanerror"][bidx]}')
                            logging.info(f'N clusters: {res_dict["best_nclust"][bidx]}\n')
                            out_best.append([res_dict[k][bidx] for k in self.params['c'].keys()] +
                                            [res_dict[k][bidx] for k in self.params['s'].keys()] +
                                            [res_dict["best_nclust"][bidx], res_dict["validation_meanerror"][bidx]])
        
        
        logging.info(f'Best solution(s):')
        if self.preproc_method is not None:
            for bidx in idx_best:
                for k in self.params['c'].keys():
                    logging.info(f'Parameters clustering (C): {k}={res_dict[k][bidx]}')
                for k in self.params['s'].keys():
                    logging.info(f'Parameters classifier (S): {k}={res_dict[k][bidx]}')
                for k in self.params['preprocessing'].keys():
                    logging.info(f'Parameters preprocessing: {k}={res_dict[k][bidx]}')
                logging.info(f'Validation performance: {res_dict["validation_meanerror"][bidx]}')
                logging.info(f'N clusters: {res_dict["best_nclust"][bidx]}\n')
                out_best.append([res_dict[k][bidx] for k in self.params['c'].keys()] +
                                [res_dict[k][bidx] for k in self.params['s'].keys()] +
                                [res_dict[k][bidx] for k in self.params['preprocessing'].keys()] +
                                [res_dict["best_nclust"][bidx], res_dict["validation_meanerror"][bidx]])
        else:
            for bidx in idx_best:
                for k in self.params['c'].keys():
                    logging.info(f'Parameters clustering (C): {k}={res_dict[k][bidx]}')
                for k in self.params['s'].keys():
                    logging.info(f'Parameters classifier (S): {k}={res_dict[k][bidx]}')
                logging.info(f'Validation performance: {res_dict["validation_meanerror"][bidx]}')
                logging.info(f'N clusters: {res_dict["best_nclust"][bidx]}\n')
                out_best.append([res_dict[k][bidx] for k in self.params['c'].keys()] +
                                [res_dict[k][bidx] for k in self.params['s'].keys()] +
                                [res_dict["best_nclust"][bidx], res_dict["validation_meanerror"][bidx]])    
            
        ParamSelectionConfounds.best_param_ = out_best
        ParamSelectionConfounds.best_index_ = idx_best

        return self
     

    def _allowed_par(self, par_dict):
        """
        Private method that controls the allowed parameter combinations
        for hierarchical clustering.
    
        :param par_dict: clustering parameter grid.
        :type par_dict: dict
        :return: whether the parameter combination can be allowed.
        :rtype: bool
        """
        if isinstance(self.clust_method, AgglomerativeClustering):
            try:
                if par_dict['linkage'] == 'ward':
                    return par_dict['affinity'] == 'euclidean'
                else:
                    return True
            except KeyError:
                try:
                    return par_dict['affinity'] == 'euclidean'
                except KeyError:
                    return True
        else:
            return True



"""
Private functions
"""


def _return_best(val_scores):
    """
    Private function that returns indices corresponding to the best solution,
    i.e., those that minimize the validation stability scores.

    :param val_scores: list of validation scores averaged over cross-validation loops.
    :type val_scores: list
    :return: list of indices.
    :rtype: list
    """
    bidx = list(np.where(np.array(val_scores) == min([vs for vs in val_scores]))[0])
    return bidx


def _return_knownbest(val_perf, idx):
    """
    Private function that, given a stability score list and indices, returns the indices corresponding
    to the best solution.

    :param val_perf: list of validation scores averaged over cross-validation loops.
    :type val_perf: list
    :param idx: list of indices.
    :type idx: list
    :return: list of indices.
    :rtype: list
    """
    bidx = _return_best([val_perf[i] for i in idx])
    return [idx[b] for b in bidx]


def _combination_perf(param_c, param_s, param_preproc, result, cv_results):
    """
    Private function that builds the performance list of one parameter combination.

    :param param_c: clustering parameters.
    :type param_c: dict
    :param param_s: classifier parameters.
    :type param_s: dict
    :param param_preproc: preprocessing parameters, None if no preprocessing.
    :type param_preproc: dict
    :param result: output of
        :meth:`neureval.best_nclust_cv_confounds.FindBestClustCVConfounds._aggregate_results`
        (metrics, best number of clusters, labels), or None if no clusters were found.
    :type result: tuple
    :param cv_results: cross-validation results table of this combination.
    :type cv_results: pandas DataFrame
    :return: performance list.
    :rtype: list
    """
    perf = [(key, val) for key, val in param_c.items()] + \
           [(key, val) for key, val in param_s.items()]
    if param_preproc is not None:
        perf += [(key, val) for key, val in param_preproc.items()]
    if result is None:
        return perf + [('best_nclust', None),
                       ('mean_train_score', None),
                       ('sd_train_score', None),
                       ('mean_val_score', None),
                       ('sd_val_score', None),
                       ('validation_meanerror', None),
                       ('tr_label', None)]
    metric, nclbest, tr_lab = result
    best_rows = cv_results.loc[cv_results.ncl == nclbest]
    return perf + [('best_nclust', nclbest),
                   ('mean_train_score', np.mean(best_rows['ms_tr'])),
                   ('sd_train_score', np.std(best_rows['ms_tr'])),
                   ('mean_val_score', np.mean(best_rows['ms_val'])),
                   ('sd_val_score', np.std(best_rows['ms_val'])),
                   ('validation_meanerror', metric['val'][nclbest]),
                   ('tr_label', tr_lab)]


def _create_result_table(out):
    """
    Private function that builds the performance result dictionary to be transformed to
    dataframe.

    :param out: grid search performance results.
    :type out: list
    :return: dictionary with results.
    :rtype: dict
    """
    dict_obj = {}
    for el in out:
        for key, val in el:
            if key in dict_obj:
                if not isinstance(dict_obj[key], list):
                    dict_obj[key] = [dict_obj[key]]
                dict_obj[key].append(val)
            else:
                dict_obj[key] = val
    return dict_obj