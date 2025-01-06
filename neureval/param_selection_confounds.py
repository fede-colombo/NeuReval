"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, 
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

from neureval.relative_validation_confounds import RelativeValidationConfounds
from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
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
    :param n_jobs: number of jobs to run in parallel, default (number of cpus - 1).
    :type n_jobs: int
    :param iter_cv: number of repeated cv loops, default 1.
    :type iter_cv: int
    :param strat: stratification vector for cross-validation splits, default None.
    :type strat: numpy array
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
                n_jobs=1, iter_cv=1, strat=None, clust_range=None):
        super().__init__(c, s, preprocessing, nrand)
        self.params = params
        self.cv = cv
        self.iter_cv = iter_cv
        self.clust_range = clust_range
        if abs(n_jobs) > mp.cpu_count():
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = abs(n_jobs)
        self.strat = strat

    def fit(self, data_tr, mod_tr, cov_tr, nclass=None):
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
        """
       	if self.preproc_method is not None:
            grid = {'c': ParameterGrid(self.params['c']), 's': ParameterGrid(self.params['s']), 'preprocessing': ParameterGrid(self.params['preprocessing'])}
            new_grid = list(itertools.product(grid['c'], grid['s'], grid['preprocessing']))
            new_params = [(data_tr, mod_tr, cov_tr, ng[0], ng[1], ng[2]) for ng in new_grid if self._allowed_par(ng[0])]
        else:
            grid = {'c': ParameterGrid(self.params['c']), 's': ParameterGrid(self.params['s'])}
            new_grid = list(itertools.product(grid['c'], grid['s']))
            new_params = [(data_tr, mod_tr, cov_tr, ng[0], ng[1]) for ng in new_grid if self._allowed_par(ng[0])]

        if len(new_grid) != len(new_params):
            logging.info(f"Dropped {len(new_grid) - len(new_params)} out of {len(new_grid)} parameter "
                          f"combinations "
                          f"due to {self.clust_method} class requirements.")

        logging.info(f'Running {len(new_params)} combinations of '
                      f'parameters...\n')

        p = mp.Pool(processes=self.n_jobs)
        
        if self.preproc_method is not None:
            out = list(zip(*p.starmap(self._run_gridsearchcv_preproc, new_params))) #new_params
            p.close()
            p.join()
        else:
            out = list(zip(*p.starmap(self._run_gridsearchcv, new_params))) #new_params
            p.close()
            p.join()

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
     

    def _run_gridsearchcv(self, data, modalities, covariates, param_c, param_s):
        """
        Private method that initializes classifier/clustering/preprocessing with different
        parameter combinations and :class:`neureval.best_nclust_cv_confounds.FindBestClustCVConfounds`.

        :param data: dataset.
        :type data: numpy array
        :param modalities: dictionary specifying the dataset for each type of input features
        :type modalities: dictionary
        :param covariates: dictionary specifying the covariates ofr each type of input features
        :type covariates: dictionary
        :param param_c: dictionary of clustering parameters.
        :type param_c: dictionary
        :param param_c: dictionary of clustering parameters.
        :type param_c: dictionary
        :return: performance list.
        :rtype: list
        """
        self.clust_method.set_params(**param_c)
        self.class_method.set_params(**param_s)
       
        findclust = FindBestClustCVConfounds(nfold=self.cv,
                                             c=self.clust_method,
                                             s=self.class_method,
                                             preprocessing=self.preproc_method,
                                             nrand=self.nrand,n_jobs=1,
                                             nclust_range=self.clust_range)
                                             
        if self.clust_range is not None:
                metric, nclbest, tr_lab = findclust.best_nclust_confounds(data, modalities, covariates, iter_cv=self.iter_cv, strat_vect=self.strat)
                #tr_lab = None
        else:
                try:
                    metric, nclbest, tr_lab = findclust.best_nclust_confounds(data, modalities, covariates, iter_cv=self.iter_cv, strat_vect=self.strat)
                except TypeError:
                    perf =  [(key, val) for key, val in param_c.items()] + \
                            [(key, val) for key, val in param_s.items()] + \
                            [('best_nclust', None),
                            ('mean_train_score', None),
                            ('sd_train_score', None),
                            ('mean_val_score', None),
                            ('sd_val_score', None),
                            ('validation_meanerror', None),
                            ('tr_label', None),
                            ]
                    return perf
                

        perf =  [(key, val) for key, val in param_c.items()] + \
                [(key, val) for key, val in param_s.items()] + \
                [('best_nclust', nclbest),
                    ('mean_train_score', np.mean(
                        findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_tr'])),
                    ('sd_train_score', np.std(
                        findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_tr'])),
                    ('mean_val_score', np.mean(
                        findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_val'])),
                    ('sd_val_score', np.std(
                        findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_val'])),
                    ('validation_meanerror', metric['val'][nclbest]),
                    ('tr_label', tr_lab),
                    ]
        return perf
        
    def _run_gridsearchcv_preproc(self, data, modalities, covariates, param_c, param_s, param_preproc):
        """
        Private method that initializes classifier/clustering/preprocessing with different
        parameter combinations and :class:`neureval.best_nclust_cv_confounds.FindBestClustCVConfounds`.

        :param data: dataset.
        :type data: numpy array
        :param modalities: dictionary specifying the dataset for each type of input features
        :type modalities: dictionary
        :param covariates: dictionary specifying the covariates ofr each type of input features
        :type covariates: dictionary
        :param param_c: dictionary of clustering parameters.
        :type param_c: dictionary
        :param param_c: dictionary of clustering parameters.
        :type param_c: dictionary
        :param param_preproc: dictionary of preprocessing parameters.
        :type param_preproc: dictionary
        :return: performance list.
        :rtype: list
        """
        self.clust_method.set_params(**param_c)
        self.class_method.set_params(**param_s)
        self.preproc_method.set_params(**param_preproc)
       
        findclust = FindBestClustCVConfounds(nfold=self.cv,
                                             c=self.clust_method,
                                             s=self.class_method,
                                             preprocessing=self.preproc_method,
                                             nrand=self.nrand,n_jobs=1,
                                             nclust_range=self.clust_range)
                                             
        if self.clust_range is not None:
                metric, nclbest, tr_lab = findclust.best_nclust_confounds(data, modalities, covariates, iter_cv=self.iter_cv, strat_vect=self.strat)
                #tr_lab = None
        else:
                try:
                    metric, nclbest, tr_lab = findclust.best_nclust_confounds(data, modalities, covariates, iter_cv=self.iter_cv, strat_vect=self.strat)
                except TypeError:
                    perf =  [(key, val) for key, val in param_c.items()] + \
                            [(key, val) for key, val in param_s.items()] + \
                            [(key, val) for key, val in param_preproc.items()] + \
                            [('best_nclust', None),
                            ('mean_train_score', None),
                            ('sd_train_score', None),
                            ('mean_val_score', None),
                            ('sd_val_score', None),
                            ('validation_meanerror', None),
                            ('tr_label', None),
                            ]
                    return perf
                

        perf =  [(key, val) for key, val in param_c.items()] + \
                [(key, val) for key, val in param_s.items()] + \
                [(key, val) for key, val in param_preproc.items()] + \
                [('best_nclust', nclbest),
                    ('mean_train_score', np.mean(
                        findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_tr'])),
                    ('sd_train_score', np.std(
                        findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_tr'])),
                    ('mean_val_score', np.mean(
                        findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_val'])),
                    ('sd_val_score', np.std(
                        findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_val'])),
                    ('validation_meanerror', metric['val'][nclbest]),
                    ('tr_label', tr_lab),
                    ]
        return perf    
        
        
        
    
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
