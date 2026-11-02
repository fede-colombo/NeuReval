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
with the optimized parameters, internal measures for comparison (silhouette, BIC),
hold-out validation (same sites) and external validation in the new site (calibrated
on a separate calibration sample). With ComBat the clusters
should follow the true clusters, not the sites.

Finally, it checks longitudinal data in long format (one row per subject and time window,
with missing windows): grid search and cross-validation with groups (all the rows of a
subject in the same fold), hold-out validation of new subjects identified by their IDs,
and the overestimation of stability that occurs when the rows of a subject are split
between training and validation folds.

Last, it follows script 04_longitudinal_states.py on simulated counts of 3 types of events
in 5 time windows: composition of the counts, a reference state for windows without
counts, states found with NeuReval in the training subjects, sequences and transitions,
the table for the regression of the transitions, validation of the states in new
subjects, and clusters of the sequences of states. It then follows script
05_longitudinal_trajectories.py: subjects clustered by the shape of their trajectories over
time (early, middle or late peak), which differ much more in their overall level.

Run from the repository root:  python scripts/00_smoke_test.py
"""
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from neureval.param_selection_confounds import ParamSelectionConfounds
from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from neureval.utils import load_table
from neureval.metrics import agreement_metrics, cv_agreement_metrics
from neureval.internal_baselines_confounds import select_best, select_best_bic_aic
from neureval.longitudinal import (composition_features, select_rows, add_reference_state,
                                   state_sequences, transition_matrices, transition_table,
                                   sequence_features, sequence_profiles, to_wide, level_shape_features)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score


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


def make_longitudinal_data(rng, n_subjects=150, n_windows=4, states=True):
    """
    Longitudinal synthetic data in long format: one row per subject and time window
    (about 10% of the rows missing, i.e. subjects with missing windows).

    Each row = stable subject profile + state of the subject in that window (3 states,
    which change over time; if states=False there are no states) + age effect + noise.
    With states=False rows of the same subject are similar only because they belong to the
    same subject: there are no clusters.
    """
    subj = np.repeat(np.arange(n_subjects), n_windows)
    window = np.tile(np.arange(1, n_windows + 1), n_subjects)
    # Stable subject profile: SD 1 with states, SD 3 without states (rows of a subject very similar)
    X = rng.normal(0, 1.0 if states else 3.0, (n_subjects, 10))[subj]
    X = X + rng.normal(0, 1.0 if states else 0.5, X.shape)
    state = rng.choice(3, size=len(subj), p=[0.4, 0.33, 0.27]) if states else np.zeros(len(subj), int)
    if states:
        centers = np.zeros((3, 6))
        for k in range(3):
            centers[k, k] = 7 / np.sqrt(2)
        X[:, :6] += centers[state]
    sex = rng.integers(0, 2, n_subjects)[subj]
    X = X + np.outer(window - 1, rng.normal(0, 0.5, X.shape[1])) + np.outer(sex - 0.5, rng.normal(0, 0.3, X.shape[1]))
    keep = rng.random(len(subj)) >= 0.1
    ids = np.array([f'sub-{i:03d}' for i in subj])[keep]
    data = pd.DataFrame(X[keep], columns=[f'Feature_{i:02d}' for i in range(1, 11)])
    data.insert(0, 'State', state[keep])
    data.insert(0, 'Window', window[keep])
    data.insert(0, 'Subject_ID', ids)
    cov = pd.DataFrame({'Subject_ID': ids, 'Window': window[keep], 'Sex': sex[keep]})
    return data, cov


def make_count_data(rng, n_subjects=300, n_windows=5):
    """
    Longitudinal counts in long format, as expected by script 04: counts of 3 types of events
    (Respiratory, Ear, Gastro) in each time window, for each subject (about 10% of the windows
    missing). True states: 0 = no events, 1 = mostly respiratory, 2 = mostly ear, 3 = mostly
    gastro. States persist over time, and the probability of no events increases with age.
    The persistence of state 1 increases with PRS.
    """
    lam = {0: [0, 0, 0], 1: [6, 1, 0.5], 2: [1, 5, 0.5], 3: [0.5, 0.5, 5]}
    prs = rng.normal(0, 1, n_subjects)
    sex = rng.integers(0, 2, n_subjects)
    state = np.zeros((n_subjects, n_windows), dtype=int)
    state[:, 0] = rng.choice(4, n_subjects, p=[0.2, 0.3, 0.35, 0.15])
    for w in range(1, n_windows):
        for i in range(n_subjects):
            logit = np.zeros(4)
            logit[state[i, w - 1]] += 1.5 + (0.8 * prs[i] if state[i, w - 1] == 1 else 0)
            logit[0] += 0.3 * w
            state[i, w] = rng.choice(4, p=np.exp(logit) / np.exp(logit).sum())
    rows = []
    for i in range(n_subjects):
        for w in range(n_windows):
            if rng.random() < 0.1:
                continue                      # window without follow-up: no row
            rows.append([f'sub-{i:04d}', w + 1, *rng.poisson(lam[state[i, w]]), sex[i], state[i, w]])
    counts = pd.DataFrame(rows, columns=['Subject_ID', 'Window', 'Respiratory', 'Ear', 'Gastro', 'Sex', 'True_state'])
    subject_data = pd.DataFrame({'PRS': prs, 'Sex': sex}, index=[f'sub-{i:04d}' for i in range(n_subjects)])
    return counts, subject_data


def make_trajectory_data(rng, n_subjects=300, n_windows=5):
    """
    Counts in long format (Subject_ID, Window, Count; about 10% of the windows missing) with 3
    true shapes over time (peak in the first, middle or last windows) and a subject-specific
    level (multiplicative, SD of the log 0.7) that varies much more than the shapes.
    """
    shapes = np.array([[3, 2, 1, .5, .5], [.5, 1.5, 3, 1.5, .5], [.5, .5, 1, 2, 3]])
    shape = rng.integers(0, 3, n_subjects)
    mult = np.exp(rng.normal(0, 0.7, n_subjects))
    rows = [[f'sub-{i:04d}', w + 1, rng.poisson(4 * shapes[shape[i], w] * mult[i])]
            for i in range(n_subjects) for w in range(n_windows) if rng.random() >= 0.1]
    counts = pd.DataFrame(rows, columns=['Subject_ID', 'Window', 'Count'])
    subject_data = pd.DataFrame({'Sex': rng.integers(0, 2, n_subjects), 'True_shape': shape},
                                index=[f'sub-{i:04d}' for i in range(n_subjects)])
    return counts, subject_data


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
    c = GaussianMixture(n_init=3, random_state=42)
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
    c_ms = GaussianMixture(covariance_type=cov_type, n_init=3, random_state=42)
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

    # 5c. Internal measures for comparison (as in section 2 of script 02), without and with ComBat
    internal = {}
    for name, kw in harmonization_settings:
        _, k_sil, lab_sil = select_best(d_abc, m_abc, c_abc, c_ms, silhouette_score, select='max',
                                        nclust_range=list(range(2, 6)), **kw)
        _, k_bic, lab_bic = select_best_bic_aic(d_abc, m_abc, c_abc, c_ms, score='bic',
                                                nclust_range=list(range(2, 6)), **kw)
        internal[name] = {'silhouette': (k_sil, adjusted_rand_score(y_ms[abc], lab_sil),
                                         adjusted_rand_score(s_abc, lab_sil)),
                          'BIC': (k_bic, adjusted_rand_score(y_ms[abc], lab_bic),
                                  adjusted_rand_score(s_abc, lab_bic))}
        for measure, (k_m, ari_true, ari_site) in internal[name].items():
            print(f'Internal measure {measure}, {name}: {k_m} clusters, ARI with true clusters {ari_true:.3f}, '
                  f'ARI with sites {ari_site:.3f}')
    findbest_ms = FindBestClustCVConfounds(c_ms, s_ms, preprocessing=None, nrand=10)

    # 5d. Hold-out validation in the same sites: 2/3 training, 1/3 hold-out, identified by IDs
    training_ids = pd.Series(np.random.default_rng(2).choice(ms_cov['Subject_ID'][abc], size=240, replace=False))
    tr_ms = ms_cov['Subject_ID'].isin(training_ids).to_numpy()
    val_ms = abc & ~tr_ms
    out_ho = findbest_ms.evaluate_confounds(ms_data, ms_modalities, ms_covariates, tr_ms, val_ms, nclust=3,
                                            site=site, harmonize_covariates=harmonize_covariates)
    ari_ho = adjusted_rand_score(y_ms[val_ms], out_ho.test_classlab)
    print(f'Hold-out validation (same sites), ComBat: normalized stability {out_ho.test_stability:.3f}, '
          f'ARI between assigned and true clusters {ari_ho:.3f}')

    # 5e. External validation in site D: training on all subjects of A, B, C;
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
        'ComBat: silhouette clusters follow the true clusters': internal['ComBat']['silhouette'][1] > 0.5,
        'ComBat: internal measures do not follow the sites': all(abs(v[2]) < 0.2 for v in internal['ComBat'].values()),
        'ComBat: hold-out subjects assigned to the right clusters': ari_ho > 0.5,
        'ComBat: error for an external site without calibration': unseen_site_error,
        'ComBat: external subjects assigned to the right clusters': ari_ext > 0.5,
        'ComBat: calibration subjects excluded from the evaluation': len(out_ext.test_classlab) == 100,
    })
    # 6. Longitudinal data in long format: several rows per subject
    print('\nLongitudinal data: one row per subject and time window')
    lg_data, lg_cov = make_longitudinal_data(np.random.default_rng(3), n_subjects=210)
    state = lg_data['State'].to_numpy()
    groups = lg_data['Subject_ID'].to_numpy()       # group_col = 'Subject_ID' in scripts 01 and 02
    # Time window as dummy covariates, so that the confound correction removes the age differences
    window_dummies = pd.get_dummies(lg_cov['Window'], prefix='Window', drop_first=True, dtype=float)
    lg_covariates = pd.concat([lg_cov[['Sex']], window_dummies], axis=1)
    lg_modalities = {'mod_01': lg_data.iloc[:, 3:8], 'mod_02': lg_data.iloc[:, 8:]}
    lg_covs = {'mod_01': lg_covariates, 'mod_02': lg_covariates}
    print(f'{len(lg_data)} rows, {len(np.unique(groups))} subjects')

    # 6a. Grid search with groups (as in script 01)
    gs = ParamSelectionConfounds({'s': {'C': [0.1, 10], 'kernel': ['linear', 'rbf']},
                                  'c': {'covariance_type': ['full']}},
                                 cv=2, s=s, c=c, preprocessing=None, nrand=10, n_jobs=1, iter_cv=2,
                                 clust_range=list(range(2, 6)), strat=None)
    gs.fit(lg_data, lg_modalities, lg_covs, groups=groups)
    print(f'Grid search with groups: best solution(s) with '
          f'{sorted(set(int(b[-2]) for b in gs.best_param_))} clusters')

    # 6b. Cross-validation with groups (as in section 1 of script 02)
    fb_lg = FindBestClustCVConfounds(c, s, preprocessing=None, nfold=2, nrand=10, n_jobs=1,
                                     nclust_range=list(range(2, 6)))
    m_lg, k_lg, lab_lg = fb_lg.best_nclust_confounds(lg_data, lg_modalities, lg_covs, iter_cv=2, groups=groups)
    ari_lg = adjusted_rand_score(state, lab_lg)
    print(f'Cross-validation with groups: {k_lg} clusters (expected 3), normalized stability '
          f'{m_lg["val"][k_lg][0]:.3f}, ARI with the true states {ari_lg:.3f}')

    # 6c. Without clusters, splitting the rows of a subject between folds overestimates the stability
    # (lower normalized stability). The size of the bias depends on how similar the rows of a subject are
    # and on the clustering algorithm: in these simulations it is clear with k-means (0.1-0.3) and small
    # with Gaussian mixtures. A stricter check is in tests/test_groups.py.
    null_data, null_cov = make_longitudinal_data(np.random.default_rng(4), states=False)
    null_dummies = pd.get_dummies(null_cov['Window'], prefix='Window', drop_first=True, dtype=float)
    null_covariates = pd.concat([null_cov[['Sex']], null_dummies], axis=1)
    null_mod = {'mod_01': null_data.iloc[:, 3:8], 'mod_02': null_data.iloc[:, 8:]}
    null_covs = {'mod_01': null_covariates, 'mod_02': null_covariates}
    null_stab = {}
    for name, g in (('rows split', None), ('subjects together', null_data['Subject_ID'].to_numpy())):
        fb_null = FindBestClustCVConfounds(KMeans(n_init=10, random_state=0), s, preprocessing=None, nfold=2,
                                           nrand=10, n_jobs=1, nclust_range=list(range(2, 5)))
        m_null, _, _ = fb_null.best_nclust_confounds(null_data, null_mod, null_covs, iter_cv=2, groups=g)
        null_stab[name] = min(v[0] for v in m_null['val'].values())
        print(f'No clusters (k-means), {name}: best normalized stability {null_stab[name]:.3f}')
    print('(with the rows of a subject split between folds the stability is overestimated, '
          'i.e. lower: it looks as if there were clusters)')

    # 6d. Hold-out validation of new subjects (as in section 3 of script 02 with long_format = True):
    # 140 training subjects, the other 70 in the hold-out set, with all their rows
    subjects = lg_data['Subject_ID'].drop_duplicates()
    training_ids = pd.Series(np.random.default_rng(5).choice(subjects, size=140, replace=False))
    tr_lg = lg_data['Subject_ID'].isin(training_ids).to_numpy()
    val_lg = ~tr_lg
    out_lg = fb_lg.evaluate_confounds(lg_data, lg_modalities, lg_covs, tr_lg, val_lg, nclust=3, groups=groups)
    ari_lg_ho = adjusted_rand_score(state[val_lg], out_lg.test_classlab)
    print(f'Hold-out validation (new subjects): normalized stability {out_lg.test_stability:.3f}, '
          f'ARI between assigned and true states {ari_lg_ho:.3f}')
    split_subject = tr_lg.copy()
    split_subject[np.flatnonzero(val_lg)[0]] = True        # one row of a hold-out subject in training
    try:
        fb_lg.evaluate_confounds(lg_data, lg_modalities, lg_covs, split_subject, ~split_subject, nclust=3,
                                 groups=groups)
        split_error = False
    except ValueError:
        split_error = True
    print(f'Subject with rows in both training and hold-out set: error raised as expected: {split_error}')

    checks.update({
        'Groups: grid search selects 3 clusters': all(int(b[-2]) == 3 for b in gs.best_param_),
        'Groups: 3 clusters in cross-validation': k_lg == 3,
        'Groups: clusters follow the true states': ari_lg > 0.5,
        'Groups: stability overestimated when rows are split': null_stab['rows split'] < null_stab['subjects together'],
        'Groups: hold-out rows assigned to the right states': ari_lg_ho > 0.5,
        'Groups: error for a subject in both sets': split_error,
    })
    # 7. Longitudinal states and transitions, as in script 04_longitudinal_states.py
    print('\nLongitudinal counts: states and transitions (script 04)')
    counts, subject_data = make_count_data(np.random.default_rng(6))
    ids, window, true_state = counts['Subject_ID'].to_numpy(), counts['Window'].to_numpy(), counts['True_state'].to_numpy()
    training_ids = pd.Series(np.random.default_rng(7).choice(subject_data.index, size=200, replace=False))
    training = np.isin(ids, training_ids)

    # 7a. Composition of the counts and reference state for windows without counts (section 1)
    features = composition_features(counts[['Respiratory', 'Ear', 'Gastro']])
    informative = features.notna().all(axis=1).to_numpy()
    cov_long = pd.concat([counts[['Sex']].astype(float),
                          pd.get_dummies(counts['Window'], prefix='Window', drop_first=True, dtype=float)], axis=1)
    feat_tr, cov_tr, ids_tr = select_rows(informative & training, features, cov_long, ids)
    print(f'{len(counts)} rows ({(~informative).sum()} without counts), {len(subject_data)} subjects, '
          f'{len(np.unique(ids[training]))} for training')

    # 7b. States with NeuReval in the training subjects (section 2)
    fb_st = FindBestClustCVConfounds(GaussianMixture(n_init=3, random_state=42), LogisticRegression(max_iter=1000),
                                     preprocessing=None, nfold=2, nrand=10, n_jobs=1, nclust_range=list(range(2, 6)))
    m_st, k_st, lab_st = fb_st.best_nclust_confounds(feat_tr, {'counts': feat_tr}, {'counts': cov_tr}, iter_cv=2,
                                                     groups=ids_tr)
    states_tr = add_reference_state(lab_st, informative[training])
    ari_st = adjusted_rand_score(true_state[training], states_tr)
    print(f'States: {k_st} (expected 3, plus state 0 for no counts), normalized stability '
          f'{m_st["val"][k_st][0]:.3f}, ARI with the true states {ari_st:.3f}')

    # 7c. Sequences, transitions and table for the regression (sections 4 and 5)
    seq = state_sequences(ids[training], window[training], states_tr)
    trans_n = transition_matrices(ids[training], window[training], states_tr, normalize=False)
    trans_p = transition_matrices(ids[training], window[training], states_tr)
    table = transition_table(ids[training], window[training], states_tr, subject_data=subject_data)
    persistence = np.diag(trans_p['all'].to_numpy())
    print(f'Sequences: {seq.shape[0]} subjects x {seq.shape[1]} windows; transitions: {len(table)}; '
          f'persistence of each state: {np.round(persistence, 2).tolist()}')

    # 7d. Validation of the states in new subjects, with the state numbers of the training (section 6)
    feat_all, cov_all, ids_all = select_rows(informative, features, cov_long, ids)
    tr_st = training[informative]
    out_st = fb_st.evaluate_confounds(feat_all, {'counts': feat_all}, {'counts': cov_all}, tr_st, ~tr_st,
                                      nclust=k_st, tr_lab=lab_st, groups=ids_all)
    states_ho = add_reference_state(out_st.test_classlab, informative[~training])
    ari_ho_st = adjusted_rand_score(true_state[~training], states_ho)
    persistence_ho = np.diag(transition_matrices(ids[~training], window[~training], states_ho)['all'].to_numpy())
    print(f'Hold-out (new subjects): normalized stability {out_st.test_stability:.3f}, ARI with the true states '
          f'{ari_ho_st:.3f}, persistence of each state: {np.round(persistence_ho, 2).tolist()}')

    # 7e. Clusters of sequences of states (section 7). In these data the states follow a Markov chain, without
    # distinct types of sequences: the check is that the analysis runs and describes every cluster.
    seq_feat = sequence_features(seq, encoding='onehot', max_missing=1)
    X_seq = seq_feat.reset_index(drop=True)
    seq_cov = subject_data.loc[seq_feat.index, ['Sex']].astype(float).reset_index(drop=True)
    fb_seq = FindBestClustCVConfounds(KMeans(n_init=10, random_state=42), LogisticRegression(max_iter=1000),
                                      preprocessing=None, nfold=2, nrand=10, n_jobs=1, nclust_range=list(range(2, 5)))
    m_seq, k_seq, lab_seq = fb_seq.best_nclust_confounds(X_seq, {'sequences': X_seq}, {'sequences': seq_cov}, iter_cv=2)
    profiles = sequence_profiles(seq, pd.Series(lab_seq, index=seq_feat.index))
    print(f'Clusters of sequences: {len(seq_feat)} subjects, {k_seq} clusters, normalized stability '
          f'{m_seq["val"][k_seq][0]:.3f}')

    # Each true state gets the same state number in the training and in the hold-out subjects
    same_numbers = all(pd.Series(states_tr[true_state[training] == t]).mode()[0] ==
                       pd.Series(states_ho[true_state[~training] == t]).mode()[0] for t in range(4))
    checks.update({
        'Longitudinal: 3 states with counts': k_st == 3,
        'Longitudinal: states follow the true states': ari_st > 0.8,
        'Longitudinal: transition table consistent with the matrices': len(table) == trans_n['all'].to_numpy().sum(),
        'Longitudinal: hold-out subjects assigned to the right states': ari_ho_st > 0.8,
        'Longitudinal: same state numbers in training and hold-out': same_numbers,
        'Longitudinal: profiles of every cluster of sequences in every window':
            len(profiles) == k_seq * seq.shape[1],
    })
    # 8. Trajectories: level and shape (script 05_longitudinal_trajectories.py)
    print('\nTrajectories: clusters of the shape over time (script 05)')
    tr_counts, tr_subj = make_trajectory_data(np.random.default_rng(8))
    wide = np.log1p(to_wide(tr_counts['Subject_ID'], tr_counts['Window'], tr_counts['Count']))
    ls = level_shape_features(wide, degree=2).dropna()
    true_shape = tr_subj.loc[ls.index, 'True_shape'].to_numpy()
    sex_traj = tr_subj.loc[ls.index, ['Sex']].astype(float).reset_index(drop=True)
    traj_results = {}
    # Without decomposition: log(1 + count) in each window (missing windows: mean of the subject)
    raw = wide.loc[ls.index].apply(lambda row: row.fillna(row.mean()), axis=1).reset_index(drop=True)
    for name, X in (('windows', raw), ('shape', ls[['linear', 'quadratic']].reset_index(drop=True))):
        fb_traj = FindBestClustCVConfounds(GaussianMixture(n_init=3, random_state=42), LogisticRegression(max_iter=1000),
                                           preprocessing=None, nfold=2, nrand=10, n_jobs=1,
                                           nclust_range=list(range(2, 6)))
        m_traj, k_traj, lab_traj = fb_traj.best_nclust_confounds(X, {'traj': X}, {'traj': sex_traj}, iter_cv=2)
        ari_shape_true = adjusted_rand_score(true_shape, lab_traj)
        ari_level = adjusted_rand_score(pd.qcut(ls['level'], k_traj, labels=False), lab_traj)
        traj_results[name] = (k_traj, ari_shape_true, ari_level)
        print(f'Clustering of the {name}: {k_traj} clusters, normalized stability {m_traj["val"][k_traj][0]:.3f}, '
              f'ARI with the true shapes {ari_shape_true:.3f}, ARI with groups of overall level {ari_level:.3f}')
    print('(the windows mix shape and level; the shape finds the 3 shapes, independently of the level)')

    checks.update({
        'Trajectories: 3 clusters of the shape': traj_results['shape'][0] == 3,
        'Trajectories: shape clusters follow the true shapes': traj_results['shape'][1] > 0.8,
        'Trajectories: shape clusters independent of the level': abs(traj_results['shape'][2]) < 0.1,
        'Trajectories: shape closer to the true shapes than the windows': traj_results['shape'][1] > traj_results['windows'][1],
    })
    failed = [name for name, ok in checks.items() if not ok]
    print('\nSMOKE TEST PASSED' if not failed else f'\nWARNING: unexpected {", ".join(failed)}')


if __name__ == '__main__':
    main()