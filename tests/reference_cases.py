"""
Reference cases for NeuReval.

Each case runs part of the NeuReval pipeline on synthetic data with fixed
random seeds and returns its results as plain Python objects (lists, floats,
ints, strings), so they can be stored in a JSON file and compared later.

The reference results are created once with save_reference_results.py,
on the code BEFORE a refactoring. test_reference_results.py then checks that the
current code still produces the same results.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from neureval.param_selection_confounds import ParamSelectionConfounds


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def make_synthetic_data(n=120, seed=0, cluster_std=4.0):
    """
    3 known clusters, 2 modalities (20 + 15 features) and 3 covariates
    (age, sex, TIV) that have a real effect on the features, so that the
    confound correction actually changes the data.
    """
    rng = np.random.default_rng(seed)
    X, y = make_blobs(n_samples=n, n_features=35, centers=3,
                      cluster_std=cluster_std, random_state=seed)
    age = rng.normal(40, 10, n)
    sex = rng.integers(0, 2, n).astype(float)
    tiv = rng.normal(1500, 100, n)
    cov_values = np.column_stack([age, sex, tiv])
    cov_z = (cov_values - cov_values.mean(0)) / cov_values.std(0)
    X = X + cov_z @ rng.normal(0, 0.8, (3, X.shape[1]))

    data = pd.DataFrame(X, columns=[f'Feature_{i:02d}' for i in range(1, 36)])
    cov = pd.DataFrame(cov_values, columns=['Age', 'Sex', 'TIV'])
    modalities = {'mod_01': data.iloc[:, :20],
                  'mod_02': data.iloc[:, 20:]}
    covariates = {'mod_01': cov,
                  'mod_02': cov[['Age', 'Sex']]}
    return data, modalities, covariates


# ---------------------------------------------------------------------------
# Conversion to JSON-friendly objects
# ---------------------------------------------------------------------------
def to_plain(obj):
    """Recursively convert numpy/pandas objects to lists, floats, ints, strings."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_plain(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)  # e.g. estimator objects stored in result tables


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def _findbest(c, s, preprocessing, nclust_range):
    data, modalities, covariates = make_synthetic_data()
    fb = FindBestClustCVConfounds(c, s, preprocessing=preprocessing, nfold=2,
                                  nrand=5, n_jobs=1, nclust_range=nclust_range)
    metrics, bestncl, tr_lab = fb.best_nclust_confounds(data, modalities, covariates,
                                                        iter_cv=2, strat_vect=None)
    cv = FindBestClustCVConfounds.cv_results_
    return to_plain({'metrics': metrics,
                     'bestncl': bestncl,
                     'tr_lab': tr_lab,
                     'cv_results': cv.to_dict(orient='list')})


def case_findbest_gmm():
    """Clustering with n_components (GaussianMixture), no preprocessing."""
    return _findbest(GaussianMixture(random_state=42), SVC(random_state=42),
                     None, list(range(2, 6)))


def case_findbest_kmeans_pca():
    """Clustering with n_clusters (KMeans) and PCA preprocessing."""
    return _findbest(KMeans(n_init=10, random_state=42), SVC(random_state=42),
                     PCA(n_components=5, random_state=42), list(range(2, 6)))


def case_findbest_hdbscan():
    """Clustering without a number of clusters (HDBSCAN)."""
    return _findbest(HDBSCAN(min_cluster_size=10), SVC(random_state=42),
                     None, None)


def _paramselection(params, preprocessing):
    data, modalities, covariates = make_synthetic_data()
    ps = ParamSelectionConfounds(params, cv=2, s=SVC(random_state=42),
                                 c=GaussianMixture(random_state=42),
                                 preprocessing=preprocessing, nrand=5, n_jobs=1,
                                 iter_cv=2, clust_range=list(range(2, 5)), strat=None)
    ps.fit(data, modalities, covariates)
    return to_plain({'cv_results': ParamSelectionConfounds.cv_results_,
                     'best_param': ParamSelectionConfounds.best_param_})


def case_paramselection():
    """Grid search over classifier and clustering parameters."""
    params = {'s': {'C': [0.1, 10], 'kernel': ['linear', 'rbf']},
              'c': {'covariance_type': ['full', 'diag']}}
    return _paramselection(params, None)


def case_paramselection_pca():
    """Grid search including preprocessing parameters (PCA components)."""
    params = {'s': {'C': [1], 'kernel': ['rbf']},
              'c': {'covariance_type': ['full']},
              'preprocessing': {'n_components': [2, 5]}}
    return _paramselection(params, PCA(random_state=42))


def _evaluate(c, s, preprocessing, nclust):
    """
    External validation with evaluate_confounds: 80 training and 40 validation
    subjects, chosen at random (not the first/last rows) to also check that
    indices in any order are handled correctly.
    Only the outputs that existed before version 1.2.0 are compared.
    """
    data, modalities, covariates = make_synthetic_data(cluster_std=10.0)
    order = np.random.default_rng(7).permutation(len(data))
    tr_idx, val_idx = np.sort(order[:80]), np.sort(order[80:])
    fb = FindBestClustCVConfounds(c, s, preprocessing=preprocessing, nfold=2,
                                  nrand=5, n_jobs=1, nclust_range=list(range(2, 6)))
    out = fb.evaluate_confounds(data, modalities, covariates, tr_idx, val_idx, nclust=nclust)
    return to_plain({'train_cllab': out.train_cllab, 'train_acc': out.train_acc,
                     'test_cllab': out.test_cllab, 'test_acc': out.test_acc})


def case_evaluate_gmm():
    """External validation, clustering with n_components (GaussianMixture)."""
    return _evaluate(GaussianMixture(random_state=42), SVC(random_state=42), None, 3)


def case_evaluate_kmeans_pca():
    """External validation, clustering with n_clusters (KMeans) and PCA preprocessing."""
    return _evaluate(KMeans(n_init=10, random_state=42), SVC(random_state=42),
                     PCA(n_components=5, random_state=42), 3)


CASES = {
    'findbest_gmm': case_findbest_gmm,
    'findbest_kmeans_pca': case_findbest_kmeans_pca,
    'findbest_hdbscan': case_findbest_hdbscan,
    'paramselection': case_paramselection,
    'paramselection_pca': case_paramselection_pca,
    'evaluate_gmm': case_evaluate_gmm,
    'evaluate_kmeans_pca': case_evaluate_kmeans_pca,
}