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

It then checks site harmonization with ComBat on multi-site data (3 training sites and
1 external site), where the site effects are larger than the differences between the
clusters, following the same steps as scripts 01 and 02: grid search, cross-validation
with the optimized parameters, hold-out validation (same sites) and external validation
in the new site (calibrated on a separate calibration sample). With ComBat the clusters
should follow the true clusters, not the sites.

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


def make_multisite_data(rng):
    """
    Multi-site synthetic data with the same structure as the templates in example_data/.

    3 true clusters of different sizes, independent of site, and strong site effects:
    each site shifts and rescales the features (shift larger than the distance between
    clusters). Age differs between sites. Sites A, B, C: 120 subjects each (training);
    site D: 150 subjects (external site).
    """
    site_size = {'A': 120, 'B': 120, 'C': 120, 'D': 150}
    site_shift = {'A': 0.0, 'B': 4.0, 'C': -4.0, 'D': 3.0}
    site_scale = {'A': 1.0, 'B': 1.6, 'C': 0.7, 'D': 1.3}
    site_age = {'A': 40, 'B': 55, 'C': 65, 'D': 45}
    site = np.repeat(list(site_size), list(site_size.values()))
    n = len(site)
    y = rng.choice(3, size=n, p=[0.4, 0.33, 0.27])     # true clusters, independent of site
    # Cluster centers at distance 7 (in units of the within-cluster SD): clearly 3 clusters
    # once the site effects are removed (closer clusters make the choice between 2 and 3 ambiguous)
    centers = np.zeros((3, 6))
    for k in range(3):
        centers[k, k] = 7 / np.sqrt(2)
    X = np.c_[centers[y] + rng.normal(0, 1, (n, 6)), rng.normal(0, 1, (n, 4))]
    age = np.array([rng.normal(site_age[s], 8) for s in site]).round(1)
    sex = rng.integers(0, 2, n)
    tiv = rng.normal(1500, 100, n).round(1)
    # Site effects: shift along a fixed direction and change of scale
    direction = rng.normal(0, 1, X.shape[1])
    direction /= np.linalg.norm(direction) / np.sqrt(X.shape[1])
    for s in site_size:
        X[site == s] = X[site == s] * site_scale[s] + site_shift[s] * direction
    # Covariate effects
    cov_z = np.c_[(age - 50) / 10, sex - 0.5, (tiv - 1500) / 100]
    X = X + cov_z @ rng.normal(0, 0.5, (3, X.shape[1]))
    ids = [f'sub-{i:03d}' for i in range(n)]

    data = pd.DataFrame(X, columns=[f'Feature_{i:02d}' for i in range(1, 11)])
    data.insert(0, 'Diagnosis', y)
    data.insert(0, 'Subject_ID', ids)
    cov = pd.DataFrame({'Subject_ID': ids, 'Diagnosis': y, 'Site': site,
                        'Age': age, 'Sex': sex, 'TIV': tiv})
    return data, cov


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
    # n_init: several initializations of the Gaussian mixture, as in scripts 01 and 02 (with a single
    # one, the scikit-learn default, it can converge to a poor solution)
    c = GaussianMixture(n_init=10, random_state=42)
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

    # 5. Multi-site data and site harmonization with ComBat
    print('\nMulti-site data: sites A, B, C (training) and D (external)')
    ms_data, ms_cov = make_multisite_data(np.random.default_rng(1))
    y_ms = ms_cov['Diagnosis'].to_numpy()
    site = ms_cov['Site'].to_numpy()
    ms_modalities = {'mod_01': ms_data.iloc[:, 2:7], 'mod_02': ms_data.iloc[:, 7:]}
    ms_covariates = {'mod_01': ms_cov[['Age', 'Sex', 'TIV']], 'mod_02': ms_cov[['Age', 'Sex']]}
    harmonize_covariates = ms_cov[['Age', 'Sex']]     # effects preserved by ComBat
    abc = site != 'D'

    def subset(mask):
        """Data, modalities, covariates, site and covariates to preserve of a subset of subjects."""
        idx = np.flatnonzero(mask)
        return (ms_data.iloc[idx].reset_index(drop=True),
                {k: v.iloc[idx].reset_index(drop=True) for k, v in ms_modalities.items()},
                {k: v.iloc[idx].reset_index(drop=True) for k, v in ms_covariates.items()},
                site[idx], harmonize_covariates.iloc[idx].reset_index(drop=True))

    d_abc, m_abc, c_abc, s_abc, h_abc = subset(abc)
    harmonization_settings = [('no harmonization', {}),
                              ('ComBat', {'site': s_abc, 'harmonize_covariates': h_abc})]

    # 5a. Grid search on sites A, B, C (as in script 01), without and with ComBat
    ms_params = {'s': {'C': [0.1, 10], 'kernel': ['linear', 'rbf']},
                 'c': {'covariance_type': ['full']}}
    grid_best = {}
    for name, kw in harmonization_settings:
        gs = ParamSelectionConfounds(ms_params, cv=2, s=s, c=c, preprocessing=None, nrand=10, n_jobs=1,
                                     iter_cv=2, clust_range=list(range(2, 6)), strat=None)
        gs.fit(d_abc, m_abc, c_abc, **kw)
        grid_best[name] = gs.best_param_
        print(f'Grid search, {name}: best solution(s) with '
              f'{sorted(set(int(b[-2]) for b in gs.best_param_))} clusters, '
              f'parameters {[tuple(b[:-2]) for b in gs.best_param_]}')

    # Classifier and clustering methods with the parameters optimized with ComBat (as in script 02)
    cov_type, C_opt, kernel_opt = grid_best['ComBat'][0][:3]
    c_ms = GaussianMixture(covariance_type=cov_type, n_init=10, random_state=42)
    s_ms = SVC(C=C_opt, kernel=kernel_opt, random_state=42)

    # 5b. Cross-validation on sites A, B, C with the optimized parameters, without and with ComBat
    results = {}
    for name, kw in harmonization_settings:
        fb = FindBestClustCVConfounds(c_ms, s_ms, preprocessing=None, nfold=2, nrand=10, n_jobs=1,
                                      nclust_range=list(range(2, 6)))
        _, k_ms, lab_ms = fb.best_nclust_confounds(d_abc, m_abc, c_abc, iter_cv=2, **kw)
        results[name] = (k_ms, adjusted_rand_score(y_ms[abc], lab_ms), adjusted_rand_score(s_abc, lab_ms))
        print(f'Cross-validation, {name}: {k_ms} clusters, ARI with true clusters {results[name][1]:.3f}, '
              f'ARI with sites {results[name][2]:.3f}')
    print('(without harmonization the clusters tend to follow the sites, with ComBat the true clusters)')
    findbest_ms = FindBestClustCVConfounds(c_ms, s_ms, preprocessing=None, nrand=10)

    # 5c. Hold-out validation in the same sites: 2/3 training, 1/3 hold-out, identified by IDs
    training_ids = pd.Series(np.random.default_rng(2).choice(ms_cov['Subject_ID'][abc], size=240, replace=False))
    tr_ms = ms_cov['Subject_ID'].isin(training_ids).to_numpy()
    val_ms = abc & ~tr_ms
    out_ho = findbest_ms.evaluate_confounds(ms_data, ms_modalities, ms_covariates, tr_ms, val_ms, nclust=3,
                                            site=site, harmonize_covariates=harmonize_covariates)
    ari_ho = adjusted_rand_score(y_ms[val_ms], out_ho.test_classlab)
    print(f'Hold-out validation (same sites), ComBat: normalized stability {out_ho.test_stability:.3f}, '
          f'ARI between assigned and true clusters {ari_ho:.3f}')

    # 5d. External validation in site D: training on all subjects of A, B, C;
    # 50 subjects of D for calibration only, the other 100 evaluated
    d_ids = ms_cov['Subject_ID'][site == 'D']
    calibration_idx = ms_cov['Subject_ID'].isin(d_ids.iloc[:50]).to_numpy()
    ext_idx = (site == 'D') & ~calibration_idx
    try:
        findbest_ms.evaluate_confounds(ms_data, ms_modalities, ms_covariates, abc, ext_idx, nclust=3,
                                       site=site, harmonize_covariates=harmonize_covariates)
        unseen_site_error = False
    except ValueError:
        unseen_site_error = True
    print(f'External site without calibration: error raised as expected: {unseen_site_error}')
    out_ext = findbest_ms.evaluate_confounds(ms_data, ms_modalities, ms_covariates, abc, ext_idx, nclust=3,
                                             site=site, harmonize_covariates=harmonize_covariates,
                                             calibration_idx=calibration_idx)
    ari_ext = adjusted_rand_score(y_ms[ext_idx], out_ext.test_classlab)
    print(f'External validation (site D, calibrated on {calibration_idx.sum()} subjects): '
          f'normalized stability {out_ext.test_stability:.3f}, ARI between assigned and true clusters '
          f'{ari_ext:.3f}, {len(out_ext.test_classlab)} subjects evaluated (expected 100)')

    checks.update({
        'ComBat: grid search selects 3 clusters': all(int(b[-2]) == 3 for b in grid_best['ComBat']),
        'ComBat: 3 clusters in cross-validation': results['ComBat'][0] == 3,
        'ComBat: clusters follow the true clusters': results['ComBat'][1] > 0.5,
        'ComBat: clusters do not follow the sites': abs(results['ComBat'][2]) < 0.2,
        'ComBat: hold-out subjects assigned to the right clusters': ari_ho > 0.5,
        'ComBat: error for an external site without calibration': unseen_site_error,
        'ComBat: external subjects assigned to the right clusters': ari_ext > 0.5,
        'ComBat: calibration subjects excluded from the evaluation': len(out_ext.test_classlab) == 100,
    })
    failed = [name for name, ok in checks.items() if not ok]
    print('\nSMOKE TEST PASSED' if not failed else f'\nWARNING: unexpected {", ".join(failed)}')


if __name__ == '__main__':
    main()