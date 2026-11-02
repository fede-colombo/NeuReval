"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience,
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

### LONGITUDINAL STATES AND TRANSITIONS
# Example of NeuReval application to longitudinal count data in long format (one row per subject and
# time window), e.g. number of infections of each type in each age window:
#   0. Import data
#   1. Composition of the counts in each row, and rows without counts
#   2. Clustering of the states with NeuReval (all the rows of a subject in the same fold)
#   3. States of all rows and description of the states
#   4. Sequences of states and transition matrices
#   5. Regression model of the transitions (e.g. PRS and SES)
#   6. Validation of the states in new subjects (hold-out set, optional)
#   7. Clusters of sequences of states: subjects with similar courses over time (optional)
# Run the sections in order. If a hold-out set is used, sections 2-5 use only the training subjects,
# so that the hold-out subjects are not used to choose the number of states. To choose the clustering and classifier parameters, run the grid search of
# 01_grid_search.py on the same features (section 1 below), with group_col = 'Subject_ID'.

# Make the required imports
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from neureval.longitudinal import (composition_features, select_rows, add_reference_state,
                                   state_sequences, transition_matrices, transition_table,
                                   sequence_features, sequence_profiles)
from sklearn.cluster import KMeans
from neureval.metrics import cv_agreement_metrics, agreement_metrics
from neureval.visualization import plot_metrics
from neureval.utils import load_table


### 0. IMPORT DATA

# Define working directories and specify the folder 'models'
main_path = 'path/to/working/directory'
data_path = os.path.join(main_path, 'models')

# Define output folder called 'results', with a subfolder for this analysis
out_dir = os.path.join(data_path, 'results', 'longitudinal_states')
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

# Counts file: one row per subject and time window, with the counts of each type already aggregated
# in the window (rows of windows in which the subject was not followed up must be removed, not filled
# with zeros: zero means "followed up, no counts").
# Example:  Subject_ID | Window | Respiratory | Ear | Gastro | Skin
counts_file = 'counts.csv'
# Covariates file: same rows, in the same order, e.g. Subject_ID | Window | Sex
cov_file = 'covariates.csv'
# Subject-level variables for the regression of the transitions (section 5): one row per subject,
# e.g. Subject_ID | PRS | SES | Sex
subject_file = 'subject_data.csv'
# For Excel files, add sheet_name='sheet_name' to load_table if needed (default: first sheet)
counts_data = load_table(os.path.join(data_path, counts_file))
cov = load_table(os.path.join(data_path, cov_file))
subject_data = load_table(os.path.join(data_path, subject_file))

id_col = 'Subject_ID'                                     # subject ID (same name in all files)
time_col = 'Window'                                       # time window: use numbers (1, 2, 3, ...)
count_cols = ['Respiratory', 'Ear', 'Gastro', 'Skin']     # one column for each type of count
# Types with very few counts (e.g. counts in a small fraction of the rows) can form a state of their own,
# because a single count changes the composition of a row a lot. Merge rare types (e.g. into a column
# 'Other') or check that the states are similar without them.
# If the windows are labels (e.g. '6-18m', '18m-3y'), give their order from the first to the last:
time_order = None
# time_order = ['6-18m', '18m-3y', '3-6y', '6-10y', '10-13y']

if not (counts_data[id_col].equals(cov[id_col]) and counts_data[time_col].equals(cov[time_col])):
    raise Exception('Counts and covariates files must contain the same rows (subject and window) in the same order')
groups = counts_data[id_col].to_numpy()
window = counts_data[time_col].to_numpy()
print(f"{len(counts_data)} rows, {len(np.unique(groups))} subjects, windows {sorted(pd.unique(window))}")

# Hold-out set (optional, section 6): a file with the IDs of the training subjects (column id_col;
# repeated IDs are allowed). All the rows of these subjects are used in sections 2-5, all the rows of the
# other subjects only in section 6. Leave training_ids_file = None to use all the subjects in sections 2-5
# without a hold-out set.
training_ids_file = None
# training_ids_file = 'training_ids.csv'
if training_ids_file is not None:
    training_ids = load_table(os.path.join(data_path, training_ids_file))[id_col].drop_duplicates()
    training = np.isin(groups, training_ids)
else:
    training = np.ones(len(groups), dtype=bool)
print(f"Training subjects: {len(np.unique(groups[training]))}; hold-out subjects: {len(np.unique(groups[~training]))}")


### 1. COMPOSITION OF THE COUNTS AND ROWS WITHOUT COUNTS

# Each row is described by the composition of its counts (which types prevail: ILR coordinates
# ilr_1, ilr_2, ...) and by its total (log_total = log(1 + total)), so that the states are not
# defined only by the total number of counts.
# pseudocount: value added to each count (zero counts are allowed). With very small counts, check
# that the results are similar with e.g. 0.5 and 1.
features = composition_features(counts_data[count_cols], pseudocount=0.5, min_total=1, include_total=True)

# Rows with at least min_total counts are clustered; the others get the reference state 0 ("no counts")
informative = features.notna().all(axis=1).to_numpy()
print(f"Rows with counts (clustered): {informative.sum()}; rows without counts (state 0): {(~informative).sum()}")

# Covariates for the confound correction.
# adjust_for_window = True: the window is added as dummy variables, so that the confound correction
#   removes the average composition and total of each window. The states then describe each row
#   relative to the subjects of the same age (e.g. "more ear infections than usual at that age") and
#   are comparable across ages.
# adjust_for_window = False: the states describe the composition as it is; some states may then occur
#   only at some ages (e.g. ear infections in the first years). Report which choice was made.
adjust_for_window = True
covariates_all = cov[['Sex']].astype(float)
if adjust_for_window:
    window_dummies = pd.get_dummies(cov[time_col], prefix=time_col, drop_first=True, dtype=float)
    covariates_all = pd.concat([covariates_all, window_dummies], axis=1)

# Rows to cluster (rows with counts of the training subjects): same rows of features, covariates and IDs
to_cluster = informative & training
feat_tr, cov_tr, groups_tr = select_rows(to_cluster, features, covariates_all, groups)


### 2. CLUSTERING OF THE STATES WITH NEUREVAL

# Clustering and classifier algorithms (use the parameters selected with the grid search).
# With many rows (e.g. tens of thousands) prefer a fast classifier (e.g. logistic regression):
# an SVM with rbf kernel becomes very slow.
c = GaussianMixture(covariance_type='full', n_init=10, random_state=42)
s = LogisticRegression(max_iter=1000)
findbestclust = FindBestClustCVConfounds(c, s, preprocessing=None, nfold=2, nrand=10, n_jobs=1,
                                         nclust_range=list(range(2, 7)))

# groups: all the rows of a subject are kept in the same fold
metrics, bestncl, tr_lab = findbestclust.best_nclust_confounds(feat_tr, {'counts': feat_tr}, {'counts': cov_tr},
                                                               iter_cv=10, groups=groups_tr)
print(f"Best number of states (rows with counts): {bestncl}")
stab_mean, (_, stab_ci) = metrics['val'][bestncl]
print(f"Validation set normalized stability: {stab_mean:.3f} +/- {stab_ci:.3f} (mean +/- half-width of the 95% CI)")

cv_per_fold, cv_summary = cv_agreement_metrics(findbestclust.cv_results_)
print(cv_summary.loc[cv_summary.ncl == bestncl].T.to_string(header=False))
cv_per_fold.to_csv('CV_metrics_per_fold_states.csv', index=False)
cv_summary.to_csv('CV_metrics_summary_states.csv', index=False)
plot_metrics(metrics, color=('black', 'black'), save_fig='plot_states.png')


### 3. STATES OF ALL ROWS AND DESCRIPTION OF THE STATES

# State of every row of the training subjects: 0 = no counts, 1..k = states found by NeuReval
states = add_reference_state(tr_lab, informative[training])
groups_all, window_all = groups, window                  # all subjects (used in section 6)
groups, window = groups[training], window[training]      # training subjects (sections 3-5)
states_rows = pd.DataFrame({id_col: groups, time_col: window, 'State': states})
states_rows.to_csv('States_per_row.csv', index=False)

# Description of the states with the original counts: proportion of each type (over all the counts of
# the rows in the state; not defined for state 0), mean total and number of rows
counts_tr = counts_data.loc[training, count_cols].reset_index(drop=True)
by_state = counts_tr.groupby(states)
description = by_state.sum().div(by_state.sum().sum(axis=1), axis=0).add_prefix('prop_')
description['mean_total'] = counts_tr.sum(axis=1).groupby(states).mean()
description['n_rows'] = by_state.size()
description.index.name = 'State'
print("Description of the states (proportion of each type, mean total, number of rows):")
print(description.round(3))
description.to_csv('States_description.csv')

# Prevalence of each state in each window (proportion of the rows of the window)
prevalence = pd.crosstab(states_rows[time_col], states_rows['State'], normalize='index')
if time_order is not None:
    prevalence = prevalence.reindex(time_order)
print("Prevalence of each state in each window:")
print(prevalence.round(3))
prevalence.to_csv('States_prevalence_per_window.csv')


### 4. SEQUENCES OF STATES AND TRANSITION MATRICES

# Sequence of states of each subject: one row per subject, one column per window (<NA> = missing window)
sequences = state_sequences(groups, window, states, time_order=time_order)
sequences.to_csv('States_sequences.csv', index_label=id_col)

# Transitions between consecutive windows: number of subjects and probabilities (rows: state at t,
# columns: state at t+1; the diagonal is the persistence). 'all' sums all pairs of consecutive windows.
trans_counts = transition_matrices(groups, window, states, time_order=time_order, normalize=False)
trans_prob = transition_matrices(groups, window, states, time_order=time_order)
for key in trans_prob:
    name = 'all' if key == 'all' else f'{key[0]}_to_{key[1]}'
    print(f"Transitions {name} (number of subjects: {int(trans_counts[key].to_numpy().sum())})")
    print(trans_prob[key].round(3))
    trans_counts[key].to_csv(f'Transitions_counts_{name}.csv')
    trans_prob[key].to_csv(f'Transitions_prob_{name}.csv')


### 5. REGRESSION MODEL OF THE TRANSITIONS

# Multinomial logistic regression of the state at t+1 on the state at t and on subject-level variables.
# - C(state_from) * PRS: effect of PRS on entering each state and, through the interaction, on staying in it
# - C(time_from): transitions may differ between ages
# The reference outcome is state 0 (no counts). Standard errors are clustered by subject, as each
# subject contributes several transitions. Requires statsmodels (pip install statsmodels).
import statsmodels.formula.api as smf

subject_vars = subject_data.set_index(id_col)[['PRS', 'SES']]
table = transition_table(groups, window, states, time_order=time_order, subject_data=subject_vars)
print(f"Transitions in the regression: {len(table)} from {table['subject'].nunique()} subjects")
model = smf.mnlogit('state_to ~ C(state_from) * PRS + SES + C(time_from)', data=table)
result = model.fit(cov_type='cluster', cov_kwds={'groups': table['subject']}, disp=0)
print(result.summary())
with open('Transitions_regression.txt', 'w') as f:
    f.write(result.summary().as_text())


### 6. VALIDATION OF THE STATES IN NEW SUBJECTS (HOLD-OUT SET)

# Requires training_ids_file (section 0) and sections 1-3. The states found in the training subjects
# (section 2) are learned by the classifier and compared with an independent clustering of the rows of
# the hold-out subjects.
if training_ids_file is None:
    raise Exception('Section 6 requires a hold-out set: set training_ids_file in section 0 and run all sections again')

# Rows with counts of all subjects; training and hold-out rows as boolean masks on these rows
feat_inf, cov_inf, groups_inf = select_rows(informative, features, covariates_all, groups_all)
tr_idx = training[informative]
val_idx = ~tr_idx
print(f"Rows with counts: {tr_idx.sum()} of training subjects, {val_idx.sum()} of hold-out subjects")

# Same algorithms as in section 2. tr_lab: the classifier learns the states of section 2, so that the
# hold-out subjects get the same state numbers as in sections 3-5. groups checks that no subject is in
# both sets.
out = findbestclust.evaluate_confounds(feat_inf, {'counts': feat_inf}, {'counts': cov_inf}, tr_idx, val_idx,
                                       nclust=bestncl, tr_lab=tr_lab, groups=groups_inf)
print(f"Hold-out normalized stability: {out.test_stability:.3f} "
      f"(0 = perfect replication, close to 1 = no better than chance)")
val_metrics, val_cm = agreement_metrics(out.test_cllab, out.test_classlab)
for name, value in val_metrics.items():
    print(f"{name:>24} = {value:.3f}" if isinstance(value, float) else f"{name:>24} = {value}")
pd.DataFrame.from_dict({'normalized_stability': out.test_stability, **val_metrics}, orient='index',
                       columns=['value']).to_csv('Val_metrics_states.csv')
val_cm.to_csv('Val_confusion_matrix_states.csv')

# Agreement in each window (do the states replicate at every age?)
window_val = window_all[informative][val_idx]
per_window = {}
for w in pd.unique(window_val):
    sel = window_val == w
    m, _ = agreement_metrics(out.test_cllab[sel], out.test_classlab[sel])
    per_window[w] = {'n_rows': int(sel.sum()), 'balanced_accuracy': m['balanced_accuracy'],
                     'cohen_kappa': m['cohen_kappa']}
per_window = pd.DataFrame(per_window).T.sort_index()
print("Agreement in each window (hold-out set):")
print(per_window.round(3))
per_window.to_csv('Val_metrics_per_window_states.csv', index_label=time_col)

# States of the hold-out subjects, assigned by the classifier trained on the training subjects
# (0 = no counts; same numbers as in sections 3-5), and their transitions, to compare with those of the
# training subjects (section 4)
holdout = ~training
holdout_states = add_reference_state(out.test_classlab, informative[holdout])
holdout_trans = transition_matrices(groups_all[holdout], window_all[holdout], holdout_states, time_order=time_order)
print("Transitions in the hold-out set (all pairs of consecutive windows), to compare with section 4:")
print(holdout_trans['all'].round(3))
holdout_trans['all'].to_csv('Val_transitions_prob_all.csv')
pd.DataFrame({id_col: groups_all[holdout], time_col: window_all[holdout], 'Assigned_state': holdout_states}) \
    .to_csv('Val_states_per_row.csv', index=False)


### 7. CLUSTERS OF SEQUENCES OF STATES (OPTIONAL)

# Subjects with similar sequences of states (e.g. "ear infections in the first years, then no infections")
# are grouped with NeuReval. Each subject is one row (no groups). Run sections 0-4 first and, for the
# validation in the hold-out set, section 6.
# encoding='onehot': subjects are similar if they are in the same states at the same times (timing matters);
# encoding='duration': proportion of time spent in each state and number of changes (timing ignored).
# max_missing: subjects with more missing windows are excluded; missing windows of the others are filled
# with the mean of the nearest observed windows of the same subject.
seq_feat = sequence_features(sequences, encoding='onehot', max_missing=1)
print(f"Subjects with sequences: {len(seq_feat)} (excluded for missing windows: {len(sequences) - len(seq_feat)})")

# Covariates of each subject for the confound correction (e.g. sex), from the subject-level file
seq_cov = subject_data.set_index(id_col).loc[seq_feat.index, ['Sex']].astype(float).reset_index(drop=True)
X_seq = seq_feat.reset_index(drop=True)

# Clustering and classifier algorithms: k-means works well with one-hot sequences
c_seq = KMeans(n_init=10, random_state=42)
s_seq = LogisticRegression(max_iter=1000)
findbest_seq = FindBestClustCVConfounds(c_seq, s_seq, preprocessing=None, nfold=2, nrand=10, n_jobs=1,
                                        nclust_range=list(range(2, 7)))
metrics_seq, bestncl_seq, seq_lab = findbest_seq.best_nclust_confounds(X_seq, {'sequences': X_seq},
                                                                       {'sequences': seq_cov}, iter_cv=10)
print(f"Best number of clusters of sequences: {bestncl_seq}")
stab_mean, (_, stab_ci) = metrics_seq['val'][bestncl_seq]
print(f"Validation set normalized stability: {stab_mean:.3f} +/- {stab_ci:.3f}")
plot_metrics(metrics_seq, color=('black', 'black'), save_fig='plot_sequence_clusters.png')

# Description of the clusters: proportion of their subjects in each state in each window (typical course).
# Clusters may reflect mainly the time spent in the most frequent state (e.g. state 0): check the profiles.
seq_labels = pd.Series(seq_lab, index=seq_feat.index, name='Sequence_cluster')
profiles = sequence_profiles(sequences, seq_labels)
print("Profiles of the clusters of sequences (proportion of subjects in each state, by window):")
print(profiles.round(3))
profiles.to_csv('Sequence_clusters_profiles.csv')
seq_labels.to_csv('Sequence_clusters_per_subject.csv', index_label=id_col)

# Validation in the hold-out subjects (requires section 6): sequences of the states assigned to them
if training_ids_file is not None:
    holdout_seq = state_sequences(groups_all[holdout], window_all[holdout], holdout_states, time_order=time_order)
    # same columns as in training (a state may not occur in the hold-out subjects)
    seq_feat_ho = sequence_features(holdout_seq, encoding='onehot', max_missing=1) \
        .reindex(columns=seq_feat.columns, fill_value=0.0)
    X_all = pd.concat([seq_feat, seq_feat_ho]).reset_index(drop=True)
    cov_all_seq = subject_data.set_index(id_col).loc[list(seq_feat.index) + list(seq_feat_ho.index), ['Sex']] \
        .astype(float).reset_index(drop=True)
    tr_seq = np.r_[np.ones(len(seq_feat), dtype=bool), np.zeros(len(seq_feat_ho), dtype=bool)]
    out_seq = findbest_seq.evaluate_confounds(X_all, {'sequences': X_all}, {'sequences': cov_all_seq},
                                              tr_seq, ~tr_seq, nclust=bestncl_seq, tr_lab=seq_lab)
    print(f"Clusters of sequences, hold-out normalized stability: {out_seq.test_stability:.3f}")
    seq_val_metrics, _ = agreement_metrics(out_seq.test_cllab, out_seq.test_classlab)
    print(f"Balanced accuracy {seq_val_metrics['balanced_accuracy']:.3f}, kappa {seq_val_metrics['cohen_kappa']:.3f}")
    pd.Series(out_seq.test_classlab, index=seq_feat_ho.index, name='Assigned_sequence_cluster') \
        .to_csv('Val_sequence_clusters_per_subject.csv', index_label=id_col)