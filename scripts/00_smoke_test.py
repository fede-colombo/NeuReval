"""
Quick end-to-end check of NeuReval on synthetic data.

Creates a realistic synthetic dataset (same structure as the templates in
example_data/): 300 subjects in 3 partially overlapping clusters of different
sizes, 6 informative and 4 uninformative features, and covariates (age, sex, TIV)
that affect the features. It saves the data as both CSV and Excel, reads it back
with load_table, runs the grid search (ParamSelectionConfounds), selects the
number of clusters (FindBestClustCVConfounds) and runs the external validation
(evaluate_confounds) on a hold-out set. If everything works, the selected number
of clusters should be 3, the labels should agree with the true clusters, and the
clusters should replicate in the hold-out set better than chance (not perfectly:
the data are noisy on purpose).

Run from the repository root:  python scripts/00_smoke_test.py
"""
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from neureval.param_selection_confounds import ParamSelectionConfounds
from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from neureval.utils import load_table
from neureval.metrics import agreement_metrics, cv_agreement_metrics


def main():
    # 1. Synthetic data
    rng = np.random.default_rng(0)
    sizes = [120, 100, 80]                 # 3 clusters of different sizes
    n = sum(sizes)
    y = np.repeat([0, 1, 2], sizes)        # true cluster of each subject
    # 6 informative features: cluster centers at distance 6 (in units of the within-cluster SD)
    centers = np.zeros((3, 6))
    for k in range(3):
        centers[k, k] = 6 / np.sqrt(2)
    X = np.c_[centers[y] + rng.normal(0, 1, (n, 6)),
              rng.normal(0, 1, (n, 4))]    # 4 uninformative features (pure noise)
    # Covariates with a real effect on the features
    age = rng.normal(40, 10, n).round(1)
    sex = rng.integers(0, 2, n)
    tiv = rng.normal(1500, 100, n).round(1)
    cov_z = np.c_[(age - 40) / 10, sex - 0.5, (tiv - 1500) / 100]
    X = X + cov_z @ rng.normal(0, 0.5, (3, X.shape[1]))
    # Subjects in random order (clusters not in blocks)
    order = rng.permutation(n)
    X, y, age, sex, tiv = X[order], y[order], age[order], sex[order], tiv[order]
    ids = [f'sub-{i:03d}' for i in range(n)]

    data = pd.DataFrame(X, columns=[f'Feature_{i:02d}' for i in range(1, 11)])
    data.insert(0, 'Diagnosis', y)
    data.insert(0, 'Subject_ID', ids)

    cov = pd.DataFrame({'Subject_ID': ids,
                        'Diagnosis': y,
                        'Age': age,
                        'Sex': sex,
                        'TIV': tiv})

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

    best_model.fit(data_in,modalities,covariates)

    # Run Findbestclust
    findbestclust = FindBestClustCVConfounds(c, s,
                                             preprocessing=None, nfold=2, nrand=10, n_jobs=1,
                                             nclust_range=list(range(2, 5)))
    metrics, bestncl, tr_lab = findbestclust.best_nclust_confounds(data_in, modalities, covariates,
                                                                    iter_cv=3, strat_vect=None)

    ari = adjusted_rand_score(y, tr_lab)
    print(f'Best number of clusters: {bestncl} (expected 3), '
          f'normalized stability {metrics["val"][bestncl][0]:.3f}')
    print(f'Agreement with the true clusters (ARI): {ari:.3f} (1 = perfect, 0 = chance)')
    _, cv_summary = cv_agreement_metrics(findbestclust.cv_results_)
    best = cv_summary.loc[cv_summary.ncl == bestncl].iloc[0]
    print(f'Cross-validation metrics for {bestncl} clusters (mean across folds): '
          f'accuracy {best.accuracy_mean:.3f}, balanced accuracy {best.balanced_accuracy_mean:.3f}, '
          f'kappa {best.cohen_kappa_mean:.3f}, ARI {best.ARI_mean:.3f}, '
          f'majority class baseline {best.majority_class_baseline_mean:.3f}')

    # 4. External validation: 200 training and 100 hold-out subjects, identified by
    # their IDs as in section 3 of 02_run_findbestclustcv.py
    rng_split = np.random.default_rng(0)
    training_ids = pd.Series(rng_split.choice(data_in['Subject_ID'], size=200, replace=False))
    tr_idx = data_in['Subject_ID'].isin(training_ids).to_numpy()
    val_idx = ~tr_idx
    out = findbestclust.evaluate_confounds(data_in, modalities, covariates, tr_idx, val_idx,
                                           nclust=bestncl)
    print(f'External validation: hold-out ACC {out.test_acc:.3f}, '
          f'normalized stability {out.test_stability:.3f} (expected below 1, i.e. better than chance), '
          f'{len(out.test_classlab)} hold-out subjects assigned (expected 100)')

    val_metrics, _ = agreement_metrics(out.test_cllab, out.test_classlab)
    print(f'External validation metrics: balanced accuracy {val_metrics["balanced_accuracy"]:.3f}, '
          f'kappa {val_metrics["cohen_kappa"]:.3f}, ARI {val_metrics["ARI"]:.3f}, '
          f'majority class baseline {val_metrics["majority_class_baseline"]:.3f}')

    checks = {'number of clusters': bestncl == 3,
              'agreement with the true clusters': ari > 0.5,
              'replication in the hold-out set': out.test_stability < 1.0,
              'labels for every hold-out subject': len(out.test_classlab) == val_idx.sum(),
              'metrics consistent with NeuReval accuracy': abs(val_metrics['accuracy'] - out.test_acc) < 1e-9,
              'per-fold metrics for every number of clusters': len(cv_summary) == 3}
    failed = [name for name, ok in checks.items() if not ok]
    print('SMOKE TEST PASSED' if not failed else f'WARNING: unexpected {", ".join(failed)}')


if __name__ == '__main__':
    main()