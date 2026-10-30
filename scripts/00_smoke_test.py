"""
Quick end-to-end check of NeuReval on synthetic data.

Creates a small dataset with 3 known clusters (same structure as the
templates in example_data/), saves it as both CSV and Excel, reads it back
with load_table and runs FindBestClustCVConfounds. If everything works,
the selected number of clusters should be 3.

Run from the repository root:  python scripts/smoke_test.py
"""
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from neureval.param_selection_confounds import ParamSelectionConfounds
from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from neureval.utils import load_table

def main():
    # 1. Synthetic data: 150 subjects, 3 clusters, 2 modalities + covariates
    rng = np.random.default_rng(42)
    n = 150
    X, y = make_blobs(n_samples=n, n_features=10, centers=3, cluster_std=1.0, random_state=42)
    ids = [f'sub-{i:03d}' for i in range(n)]

    data = pd.DataFrame(X, columns=[f'Feature_{i:02d}' for i in range(1, 11)])
    data.insert(0, 'Diagnosis', y)
    data.insert(0, 'Subject_ID', ids)

    cov = pd.DataFrame({'Subject_ID': ids,
                        'Diagnosis': y,
                        'Age': rng.normal(40, 10, n).round(1),
                        'Sex': rng.integers(0, 2, n),
                        'TIV': rng.normal(1500, 100, n).round(1)})

    with tempfile.TemporaryDirectory() as tmp:
        # 2. Save in both formats and read back with load_table
        for ext in ('.csv', '.xlsx'):
            data_path = os.path.join(tmp, 'database' + ext)
            cov_path = os.path.join(tmp, 'covariates' + ext)
            if ext == '.csv':
                data.to_csv(data_path, index=False)
                cov.to_csv(cov_path, index=False)
            else:
                data.to_excel(data_path, index=False)
                cov.to_excel(cov_path, index=False)

            data_in = load_table(data_path)
            cov_in = load_table(cov_path)
            assert data_in.shape == data.shape, f'{ext}: data shape mismatch'
            assert cov_in.shape == cov.shape, f'{ext}: covariates shape mismatch'
            print(f'load_table {ext}: OK {data_in.shape}')

    # 3. Run the clustering validation on the data read back from file
    modalities = {'mod_01': data_in.iloc[:, 2:7],
                'mod_02': data_in.iloc[:, 7:]}
    covariates = {'mod_01': cov_in.iloc[:, 2:],
                'mod_02': cov_in.iloc[:, 2:4]}

    # Define clustering, classifier, and preprocessing parameters to be optimized.
    # params should be a dictionary of the form {‘s’: {classifier parameter grid}, ‘c’: {clustering parameter grid}} 
    # including the lists of classifiers and clustering methods to fit to the data.
    # In case you want to optimize also preprocessing parameters (e.g., PCA or UMAP components), 
    # specify {'preprocessing':{preprocessing parameter grid}} within the dictionary.
    params = {'s': {'C': [0.01, 10, 100],
                    'kernel':['linear', 'rbf']},
            'c': {'covariance_type':['full']},
            'preprocessing': {'n_components': list(range(2,5))}}

    # Define classifier and clustering methods
    c = GaussianMixture(random_state=42)
    s = SVC(random_state=42)

    # If you want to perform dimensionality reduction, uncomment the following line. Change the dimensionality reduction algorithm accordingly (e.g., UMAP, PCA)
    # preproc = umap.UMAP(random_state=42)
    preproc = None

    # Parameters selection
    best_model = ParamSelectionConfounds(params, 
                                        cv=2, 
                                        s=s, 
                                        c=c,
                                        preprocessing=preproc,
                                        nrand=10,
                                        n_jobs=1,
                                        iter_cv=3,
                                        clust_range=list(range(2,5)),
                                        strat=None)

    best_model.fit(data,modalities,covariates)

    # Run Findbestclust
    findbestclust = FindBestClustCVConfounds(c, s,
                                            preprocessing=None, nfold=2, nrand=10, n_jobs=1,
                                            nclust_range=list(range(2, 5)))
    metrics, bestncl, tr_lab = findbestclust.best_nclust_confounds(data_in, modalities, covariates,
                                                                    iter_cv=3, strat_vect=None)

    print(f'Best number of clusters: {bestncl} (expected 3)')
    print('SMOKE TEST PASSED' if bestncl == 3 else 'WARNING: unexpected number of clusters')

if __name__ == '__main__':
    main()