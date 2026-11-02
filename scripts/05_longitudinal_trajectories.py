"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience,
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

### LONGITUDINAL TRAJECTORIES
# Example of NeuReval application to cluster subjects by the shape of their trajectories over time, e.g.
# number of infections in each age window (one cluster per subject for the whole follow-up; to cluster
# the time points into states, see 04_longitudinal_states.py):
#   0. Import data
#   1. Trajectories: one row per subject, decomposed into level and shape
#   2. Clustering of the shapes with NeuReval (one row per subject: no groups)
#   3. Description of the clusters, and check that they do not reflect only the level
#   4. Validation in new subjects (hold-out set, optional)
# To choose the clustering and classifier parameters, run the grid search of 01_grid_search.py on the
# same features (section 1 below). The data have one row per subject: no groups.

# Make the required imports
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from neureval.longitudinal import to_wide, level_shape_features
from neureval.metrics import cv_agreement_metrics, agreement_metrics
from neureval.visualization import plot_metrics
from neureval.utils import load_table


### 0. IMPORT DATA

# Define working directories and specify the folder 'models'
main_path = 'path/to/working/directory'
data_path = os.path.join(main_path, 'models')

# Define output folder called 'results', with a subfolder for this analysis
out_dir = os.path.join(data_path, 'results', 'longitudinal_trajectories')
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

# Counts file in long format, as for script 04: one row per subject and time window (rows of windows
# without follow-up removed, not set to zero). Example:  Subject_ID | Window | Respiratory | Ear | Gastro
counts_file = 'counts.csv'
# Subject-level variables: one row per subject, e.g. Subject_ID | Sex | PRS | SES
subject_file = 'subject_data.csv'
counts_data = load_table(os.path.join(data_path, counts_file))
subject_data = load_table(os.path.join(data_path, subject_file)).set_index('Subject_ID')

id_col = 'Subject_ID'
time_col = 'Window'                               # time window: use numbers (1, 2, 3, ...)
count_cols = ['Respiratory', 'Ear', 'Gastro']     # one trajectory for each type (or a single column, e.g. total)
time_order = None                                 # e.g. ['6-18m', '18m-3y', '3-6y', '6-10y', '10-13y'] for labels
# Time of each window (e.g. mid-point of each age window in years), if the windows are not equally spaced.
# With None the windows are treated as equally spaced.
time_points = None
# time_points = [1.0, 2.25, 4.5, 8.0, 11.5]

# Hold-out set (optional, section 4): file with the IDs of the training subjects (column id_col).
# Leave None to use all subjects in sections 1-3.
training_ids_file = None
# training_ids_file = 'training_ids.csv'


### 1. TRAJECTORIES: LEVEL AND SHAPE

# For each type: one row per subject and one column per window (NaN = missing window), log(1 + count)
# to reduce the skewness of the counts, and decomposition of each trajectory into
# - level: mean over the windows (how many infections overall)
# - linear: increase (> 0) or decrease (< 0) over time
# - quadratic: peak (< 0) or dip (> 0) in the middle windows
# degree: highest degree of the shape. With 5 windows, 2 (linear and quadratic) is usually enough; use 1 if
# only increases and decreases are expected (a quadratic term without real signal adds noise).
# min_observed: subjects with fewer observed windows are excluded (default: degree + 2).
degree = 2
levels, shapes = {}, {}
for col in count_cols:
    wide = np.log1p(to_wide(counts_data[id_col], counts_data[time_col], counts_data[col], time_order=time_order))
    feat = level_shape_features(wide, degree=degree, time_points=time_points)
    levels[col] = feat['level']
    shapes[col] = feat.drop(columns='level').add_prefix(f'{col}_')
level = pd.DataFrame(levels)
shape = pd.concat(shapes, axis=1)
shape.columns = shape.columns.droplevel(0)

# Subjects with enough observed windows for every type
complete = shape.notna().all(axis=1) & level.notna().all(axis=1)
subjects = shape.index[complete]
print(f"Subjects: {len(shape)}; with enough observed windows: {len(subjects)}")

# Training and hold-out subjects
if training_ids_file is not None:
    training_ids = load_table(os.path.join(data_path, training_ids_file))[id_col].drop_duplicates()
    training = np.isin(subjects, training_ids)
else:
    training = np.ones(len(subjects), dtype=bool)
subjects_tr = subjects[training]
print(f"Training subjects: {len(subjects_tr)}; hold-out subjects: {(~training).sum()}")

# Modalities: the shape of each type (the level is not used for the clustering, so that the clusters
# describe the timing of the infections rather than their overall number; to study the level too, add
# 'level': level.loc[subjects_tr] as a separate modality).
# Covariates for the confound correction: subject-level variables (e.g. sex).
X_tr = shape.loc[subjects_tr].reset_index(drop=True)
modalities = {col: X_tr.filter(like=f'{col}_') for col in count_cols}
cov_tr = subject_data.loc[subjects_tr, ['Sex']].astype(float).reset_index(drop=True)
covariates = {col: cov_tr for col in count_cols}


### 2. CLUSTERING OF THE SHAPES WITH NEUREVAL

c = GaussianMixture(covariance_type='full', n_init=10, random_state=42)
s = LogisticRegression(max_iter=1000)
findbestclust = FindBestClustCVConfounds(c, s, preprocessing=None, nfold=2, nrand=10, n_jobs=1,
                                         nclust_range=list(range(2, 7)))
metrics, bestncl, tr_lab = findbestclust.best_nclust_confounds(X_tr, modalities, covariates, iter_cv=10)
print(f"Best number of trajectory clusters: {bestncl}")
stab_mean, (_, stab_ci) = metrics['val'][bestncl]
print(f"Validation set normalized stability: {stab_mean:.3f} +/- {stab_ci:.3f} (mean +/- half-width of the 95% CI)")
cv_per_fold, cv_summary = cv_agreement_metrics(findbestclust.cv_results_)
cv_per_fold.to_csv('CV_metrics_per_fold_trajectories.csv', index=False)
cv_summary.to_csv('CV_metrics_summary_trajectories.csv', index=False)
plot_metrics(metrics, color=('black', 'black'), save_fig='plot_trajectories.png')

labels = pd.Series(tr_lab, index=subjects_tr, name='Trajectory_cluster')
labels.to_csv('Trajectory_clusters_per_subject.csv', index_label=id_col)


### 3. DESCRIPTION OF THE CLUSTERS

# Mean number of infections of each type in each window, for each cluster (the typical trajectory)
long_tr = counts_data[counts_data[id_col].isin(subjects_tr)].copy()
long_tr['cluster'] = long_tr[id_col].map(labels)
mean_traj = long_tr.groupby(['cluster', time_col])[count_cols].mean()
if time_order is not None:
    mean_traj = mean_traj.reindex(pd.MultiIndex.from_product([sorted(labels.unique()), time_order]))
print("Mean number of infections of each type, by cluster and window:")
print(mean_traj.round(2))
mean_traj.to_csv('Trajectory_clusters_mean_counts.csv')

# Level of each cluster (mean of log(1 + count) over the windows) and number of subjects
level_tr = level.loc[subjects_tr]
summary = level_tr.groupby(labels).mean().add_prefix('level_')
summary['n_subjects'] = labels.value_counts().sort_index()
print(summary.round(3))
summary.to_csv('Trajectory_clusters_level.csv')

# Check: do the clusters only reflect the level? Agreement with groups of the total level of the same size
# (close to 0: the clusters are not explained by the level)
total_level = level_tr.mean(axis=1)
ari_level = adjusted_rand_score(pd.qcut(total_level, bestncl, labels=False), tr_lab)
print(f"Agreement (ARI) between the clusters and groups of overall level: {ari_level:.3f}")


### 4. VALIDATION IN NEW SUBJECTS (HOLD-OUT SET, OPTIONAL)

if training_ids_file is None:
    raise Exception('Section 4 requires a hold-out set: set training_ids_file in section 0 and run all sections again')

subjects_ho = subjects[~training]
X_all = shape.loc[list(subjects_tr) + list(subjects_ho)].reset_index(drop=True)
mod_all = {col: X_all.filter(like=f'{col}_') for col in count_cols}
cov_all = subject_data.loc[list(subjects_tr) + list(subjects_ho), ['Sex']].astype(float).reset_index(drop=True)
tr_idx = np.r_[np.ones(len(subjects_tr), dtype=bool), np.zeros(len(subjects_ho), dtype=bool)]
# tr_lab: the classifier learns the clusters of section 2, so that the hold-out subjects get the same numbers
out = findbestclust.evaluate_confounds(X_all, mod_all, {col: cov_all for col in count_cols}, tr_idx, ~tr_idx,
                                       nclust=bestncl, tr_lab=tr_lab)
print(f"Hold-out normalized stability: {out.test_stability:.3f} "
      f"(0 = perfect replication, close to 1 = no better than chance)")
val_metrics, val_cm = agreement_metrics(out.test_cllab, out.test_classlab)
for name, value in val_metrics.items():
    print(f"{name:>24} = {value:.3f}" if isinstance(value, float) else f"{name:>24} = {value}")
pd.DataFrame.from_dict({'normalized_stability': out.test_stability, **val_metrics}, orient='index',
                       columns=['value']).to_csv('Val_metrics_trajectories.csv')
val_cm.to_csv('Val_confusion_matrix_trajectories.csv')
pd.Series(out.test_classlab, index=subjects_ho, name='Assigned_trajectory_cluster') \
    .to_csv('Val_trajectory_clusters_per_subject.csv', index_label=id_col)
