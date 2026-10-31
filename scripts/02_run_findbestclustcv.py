"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, 
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

### 1. RUN FindBestClustCVConfounds WITH OPTIMIZED CLUSTERING/CLASSIFIER/PREPROCESSING PARAMETERS

# Make the required imports
import pandas as pd
import numpy as np
from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from neureval.param_selection_confounds import ParamSelectionConfounds
from neureval.visualization import plot_metrics
from sklearn.metrics import zero_one_loss, adjusted_mutual_info_score
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import os
import pickle as pkl
import logging
from neureval.utils import load_table
from neureval.metrics import cv_agreement_metrics

# Define working directories and specify the folder 'models'
main_path = 'path/to/working/directory'
data_path = os.path.join(main_path,'models')

# Define output folder called 'results'. 
# Within 'results, define the subfolder corresponding to the input features (e.g., 'GM + FA').
# Within this subfolder, specify the folder corresponding to the set of covariates used (e.g., 'age_sex_TIV')
out_dir = os.path.join(data_path,'results', 'feature_name', 'covariates')
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

# Import data and covariates files
data_file = 'database.xlsx'
cov_file = 'covariates.xlsx'
# For Excel files with multiple sheets, add sheet_name='sheet_name' (default: first sheet)
data = load_table(os.path.join(data_path,data_file))
cov = load_table(os.path.join(data_path, cov_file))

# Check sample sizes of data and covariates files
if len(data) != len(cov):
    raise Exception('Sample sizes of data and covariates files are different. Please check them')

# Define two dictionaries for modalities and covariates variables.
# For each kind of modality, specify the index of the columuns corresponding to each set of input modality (e.g., DTI, GM, fMRI)
# You can also specify different sets of covariates for each modality
modalities = {'feature_01': data.iloc[:, 2:124],
              'feature_02': data.iloc[:,124:272],
              'feature_03': data.iloc[:,272:]}

covariates = {'feature_01': cov.iloc[:,2:],
              'feature_02': cov.iloc[:,2:-2],
              'feature_03': cov.iloc[:,2:-1]}

# Site harmonization with ComBat (optional). If the data come from more than one site or scanner,
# specify the site of each subject and the covariates whose effects ComBat must preserve (e.g. age, sex).
# ComBat is estimated on the training set of each fold and applied to its validation set, before the
# correction for covariates; the folds are stratified by site. Leave site = None for single-site data.
# If the site is a column of the covariates file, exclude it from the covariates dictionary above,
# which is used for the confound correction (e.g. cov[['Age', 'Sex', 'TIV']] instead of cov.iloc[:,2:]).
site = None
# site = cov['Site'].to_numpy()
harmonize_covariates = None
# harmonize_covariates = cov[['Age', 'Sex']]   # or a dictionary with one table per modality

# Define clusteirng and classifier algortihms with optimized parameters
# n_init: number of initializations of the Gaussian mixture (more stable results than a single one)
s = SVC(C=1, kernel='rbf', random_state=42)
c = GaussianMixture(covariance_type='full', n_init=10, random_state=42)
# Uncomment the following line if you also optimized preprocessing steps
# preproc = UMAP(n_components=2, n_neighbors=30, min_dist=0.0, random_state=42)
preproc = None

# Initialize FindBestClustCVConfounds class. It performs (repeated) k-folds cross-validation to select the best number of clusters.
# Parameters to be specified:
# preprocessing: if not None, specify the preprocessing algorithm used for dimensionality reduction
# nfold: cross-validation folds
# nrand: number of random labelling iterations, default 10
# n_jobs: number of jobs to run in parallel, default (number of cpus - 1)
# clust_range: list with number of clusters, default None
findbestclust = FindBestClustCVConfounds(c,s, preprocessing=preproc, nfold=2, nrand=10,  n_jobs=1, nclust_range=list(range(2,4)))

# Run FindBestClustCVConfounds. It returns normlaized stability (metrics), best number of clusters (bestncl), and clusters' labels (tr_lab).
# Parameters to be specified:
# iter_cv: number of repeated cross-validation, default 1
# strat_vect: stratification vector for cross-validation splits, default None (stratified by site if site is given)
# site, harmonize_covariates: site harmonization with ComBat (see above), default None
metrics, bestncl, tr_lab = findbestclust.best_nclust_confounds(data, modalities, covariates, iter_cv=10, strat_vect=None,
                                                               site=site, harmonize_covariates=harmonize_covariates)
val_results = list(metrics['val'].values())
val_results = np.array(val_results, dtype=object)

print(f"Best number of clusters: {bestncl}")
stab_mean, (_, stab_ci) = metrics['val'][bestncl]
print(f"Validation set normalized stability: {stab_mean:.3f} +/- {stab_ci:.3f} (mean +/- half-width of the 95% CI)")
 
# Agreement metrics in each cross-validation fold, between the clustering labels of the
# validation fold and the predictions of the classifier trained on the training fold:
# accuracy, balanced accuracy, Cohen's kappa, ARI, NMI, AMI and majority class baseline.
# cv_per_fold: one row per fold and number of clusters
# cv_summary: mean and standard deviation across folds, for each number of clusters
cv_per_fold, cv_summary = cv_agreement_metrics(findbestclust.cv_results_)
print(f"Agreement metrics across folds for {bestncl} clusters (mean and SD):")
print(cv_summary.loc[cv_summary.ncl == bestncl].T.to_string(header=False))
# Change the .csv file names according to your model
cv_per_fold.to_csv('CV_metrics_per_fold_model_name.csv', index=False)
cv_summary.to_csv('CV_metrics_summary_model_name.csv', index=False)
 
# Normalized stability plot. For each number of clusters, normalized stabilities are represented for both training (dashed line) and validation sets (continuous line).
# Colors for training and validation sets can be changed (default: ('black', 'black')).
# To save the plot, specify the file name for saving figure in png format.
plot = plot_metrics(metrics, color=('black', 'black'), save_fig='plot_model_name.png')
 
# Save dataset with clustering labels for post-hoc analyses
labels_tr = pd.DataFrame(tr_lab, columns=['Labels'])
data_all_tr = pd.concat([data, labels_tr], axis=1)
# Change the .csv file name according to your model
data_all_tr.to_csv('Labels_model_name_training.csv', index=False)




### 2. COMPUTE INTERNAL MEASURES FOR COMPARISON

# In case you want to also compute internal measures for comparison, make the following imports
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
from neureval.internal_baselines_confounds import select_best, select_best_bic_aic
from neureval.utils import kuhn_munkres_algorithm


# The code is organized in different sections for each kind of measures (i.e., silhouette score, Davies-Boudin score, AIC and BIC).
# If you want to calculate only some scores, comment the ones you don't need

# SILHOUETTE SCORE
# Use select_best to calcluate silhouette score. Specify silhouette_score as int_measure and 'max' as select parameter.
# It returns silhouette score (sil_score), number of clusters selected (sil_best), and silhouette labels (sil_labels)
logging.info("Silhouette score based selection")
sil_score, sil_best, sil_label = select_best(data, modalities, covariates, c, silhouette_score, preprocessing=preproc, select='max', nclust_range=list(range(2,5)))

logging.info(f"Best number of clusters (and scores): "
             f"{{{sil_best}({sil_score})}}")
logging.info(f'AMI (true labels vs clustering labels) training = '
             f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(sil_label)))}')
logging.info('\n\n')

# Save results obtained with silhouette score.
# Silhouette score, number of clusters, and clusters' labels are organized as a dictionary and saved as a pickle object
sil_results = {'sil_score': sil_score,
               'sil_best': sil_best,
               'sil_label': sil_label}
pkl.dump(sil_results, open('./sil_results_model_name.pkl', 'wb'))
# To open the results, uncomment the following line
# pd.read_pickle('./sil_results_model_name.pkl')


# DAVIES-BOULDIN SCORE
# Use select_best to calcluate Davies-Bouldin score. Specify davies_bouldin_score as int_measure and 'min' as select parameter.
# It returns davies-bouldin score (db_score), number of clusters selected (db_best), and davies-bouldin labels (db_labels)
logging.info("Davies-Bouldin score based selection")
db_score, db_best, db_label = select_best(data,modalities, covariates, c, davies_bouldin_score,preprocessing=preproc,
                                                   select='min', nclust_range=list(range(2,5)))

logging.info(f"Best number of clusters (and scores): "
             f"{{{db_best}({db_score})}}")
logging.info(f'AMI (true labels vs clustering labels) training = '
             f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(db_label)))}')
logging.info('\n\n')

# Save results obtained with Davies-Bouldin score.
# Davies-Bouldin score, number of clusters, and clusters' labels are organized as a dictionary and saved as a pickle object
db_results = {'db_score': db_score,
               'db_best': db_best,
               'db_label': db_label}
pkl.dump(db_results, open('./db_results_model_name.pkl', 'wb'))
# To open the results, uncomment the following line
# pd.read_pickle('./db_results_model_name.pkl')


# AIC AND BIC (for Gaussian Mixture Models)
# Calculate AIC score. Specify 'aic' as score parameter.
# It returns AIC score (aic_score), number of clusters selected (aic_best), and AIC labels (aic_labels)
logging.info("AIC score based selection")
aic_score, aic_best, aic_label = select_best_bic_aic(data, modalities, covariates, c, score='aic', preprocessing=preproc, nclust_range=list(range(2,5)))

logging.info(f"Best number of clusters (and scores): "
              f"{{{aic_best}({aic_score})}}")
logging.info(f'AMI (true labels vs clustering labels) training = '
              f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(aic_label)))}')
logging.info('\n\n')

# Save results obtained with AIC.
# AIC score, number of clusters, and clusters' labels are organized as a dictionary and saved as a pickle object
aic_results = {'aic_score': aic_score,
               'aic_best': aic_best,
               'aic_label': aic_label}
pkl.dump(aic_results, open('./aic_results_model_name.pkl', 'wb'))
# To open the results, uncomment the following line
# pd.read_pickle('./aic_results_model_name.pkl')

# Calculate BIC score. Specify 'bic' as score parameter.
# It returns BIC score (bic_score), number of clusters selected (bic_best), and BIC labels (bic_labels)
logging.info("BIC score based selection")
bic_score, bic_best, bic_label = select_best_bic_aic(data, modalities, covariates, c, score='bic', preprocessing=preproc, nclust_range=list(range(2,6)))

logging.info(f"Best number of clusters (and scores): "
              f"{{{bic_best}({bic_score})}}")
logging.info(f'AMI (true labels vs clustering labels) training = '
              f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(bic_label)))}')
logging.info('\n\n')

# Save results obtained with BIC.
# BIC score, number of clusters, and clusters' labels are organized as a dictionary and saved as a pickle object
bic_results = {'bic_score': bic_score,
               'bic_best': bic_best,
               'bic_label': bic_label}
pkl.dump(bic_results, open('./bic_results_model_name.pkl', 'wb'))
# To open the results, uncomment the following line
# pd.read_pickle('./bic_results_model_name.pkl')


### 3. EXTERNAL VALIDATION
# This section can be run on its own, without the previous ones.

# Make the required imports
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from umap import UMAP
from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from neureval.utils import load_table
from neureval.metrics import agreement_metrics

# Define working directories and specify the folder 'models'
main_path = 'path/to/working/directory'
data_path = os.path.join(main_path, 'models')

# Define output folder called 'results'.
# Within 'results', define the subfolder corresponding to the input features (e.g., 'GM + FA').
# Within this subfolder, specify the folder corresponding to the set of covariates used (e.g., 'age_sex_TIV')
out_dir = os.path.join(data_path, 'results', 'feature_name', 'covariates')
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

# Name of the column with the subjects' IDs (the same in all files, e.g. 'Subject_ID' or 'IDNUM')
id_col = 'Subject_ID'

# Import the complete data and covariates files (training + hold-out subjects, in any order).
# The covariates file must contain the same subjects, in the same order as the data file.
data_all_file = 'database_all_name.xlsx'
cov_all_file = 'covariates_all_name.xlsx'
# Supported formats: .csv, .xlsx, .xls. For Excel files with multiple sheets, add sheet_name='sheet_name' (default: first sheet)
data_all = load_table(os.path.join(data_path, data_all_file))
cov_all = load_table(os.path.join(data_path, cov_all_file))

if not data_all[id_col].equals(cov_all[id_col]):
    raise Exception('Data and covariates files must contain the same subjects in the same order. Please check them')

# Import the list of the IDs of the training subjects: a .csv or Excel file with a column named as id_col.
# For example, the file with the training labels saved in section 1 ('Labels_model_name_training.csv') can be used.
training_ids_file = 'training_ids.csv'
training_ids = load_table(os.path.join(data_path, training_ids_file))[id_col]

missing = training_ids[~training_ids.isin(data_all[id_col])]
if len(missing) > 0:
    raise Exception(f'{len(missing)} training subjects are missing from the complete file: {missing.tolist()[:10]}')
if training_ids.duplicated().any():
    raise Exception('The list of training subjects contains duplicated IDs. Please check it')

# Training and hold-out sets as boolean masks (True = subject belongs to the set)
tr_idx = data_all[id_col].isin(training_ids).to_numpy()
val_idx = ~tr_idx

# Site harmonization with ComBat (optional), estimated on the training subjects only.
# - Hold-out validation: the hold-out subjects come from the same sites as the training subjects.
#   Specify site_col (and harmonize_cols): the harmonization estimated in training is applied to them.
# - External validation in a new site (not in the training set): also specify a calibration sample
#   of the new site (e.g. healthy controls, comparable to the training population). It is used only
#   to estimate the site effects of the new site and is excluded from the evaluation.
site_col = None               # e.g. 'Site': column of the covariates file with the site of each subject
harmonize_cols = None         # e.g. ['Age', 'Sex']: covariates whose effects ComBat must preserve
calibration_ids_file = None   # e.g. 'calibration_ids.csv': file with the IDs (column id_col) of the calibration subjects
# If site_col is a column of the covariates file, exclude it from covariates_val below, which is used for
# the confound correction (e.g. cov_all[['Age', 'Sex', 'TIV']] instead of cov_all.iloc[:, 2:]).

site = cov_all[site_col].to_numpy() if site_col is not None else None
harmonize_covariates = cov_all[harmonize_cols] if harmonize_cols is not None else None
calibration_idx = None
if calibration_ids_file is not None:
    calibration_ids = load_table(os.path.join(data_path, calibration_ids_file))[id_col]
    calibration_idx = data_all[id_col].isin(calibration_ids).to_numpy()
    if (calibration_idx & tr_idx).any():
        raise Exception('Some calibration subjects are also training subjects. Please check them')
    val_idx = val_idx & ~calibration_idx     # calibration subjects are not evaluated
    print(f"Calibration sample size: {calibration_idx.sum()}")

# Check the sample size of training and hold-out sets
print(f"Training set sample size: {tr_idx.sum()}")
print(f"Hold-out set sample size: {val_idx.sum()}")

# Define two dictionaries for modalities and covariates variables, as for the training set.
# For each kind of modality, specify the index of the columns corresponding to each set of input modality (e.g., DTI, GM, fMRI)
# You can also specify different sets of covariates for each modality
modalities_val = {'feature_01': data_all.iloc[:, 2:124],
                  'feature_02': data_all.iloc[:, 124:272],
                  'feature_03': data_all.iloc[:, 272:]}

covariates_val = {'feature_01': cov_all.iloc[:, 2:],
                  'feature_02': cov_all.iloc[:, 2:-2],
                  'feature_03': cov_all.iloc[:, 2:-1]}

# Define classifier, clustering and (if used) preprocessing algorithms with the optimized parameters,
# the same used in section 1
s = SVC(C=1, kernel='rbf', random_state=42)
c = GaussianMixture(covariance_type='full', n_init=10, random_state=42)
# If you also optimized preprocessing steps, uncomment the following line and set preprocessing=preproc below
# preproc = UMAP(n_components=2, n_neighbors=30, min_dist=0.0, random_state=42)
findbestclust = FindBestClustCVConfounds(c, s, preprocessing=None, nrand=10)

# Best number of clusters identified in the training set (bestncl printed in section 1)
bestncl = 3

# Validate the identified clusters
# Parameters to be specified:
# data_all: complete dataset (training + hold-out sets)
# modalities_val, covariates_val: modalities and covariates of the complete dataset
# tr_idx, val_idx: training and hold-out subjects (boolean masks or integer row positions)
# nclust: best number of clusters identified in the training dataset (i.e., bestncl)
# tr_lab: clustering labels for the training set. If not None, the clustering algorithm is not performed and the classifier is fitted. Default None
# site, harmonize_covariates, calibration_idx: site harmonization with ComBat (see above), default None
out = findbestclust.evaluate_confounds(data_all, modalities_val, covariates_val, tr_idx, val_idx,
                                       nclust=bestncl, tr_lab=None, site=site,
                                       harmonize_covariates=harmonize_covariates,
                                       calibration_idx=calibration_idx)
if out is None:
    raise Exception('No clusters were identified in the training or hold-out set')

print(f"Training ACC: {out.train_acc:.3f}")
print(f"Hold-out ACC (raw, chance level depends on the number of clusters): {out.test_acc:.3f}")
print(f"Hold-out normalized stability: {out.test_stability:.3f} "
      f"(0 = perfect replication, close to 1 = no better than chance)")

# Agreement metrics between the clustering labels of the hold-out set (reference) and the
# predictions of the classifier trained on the training set: accuracy, balanced accuracy,
# Cohen's kappa, ARI, NMI, AMI, majority class baseline and cluster sizes, plus the confusion matrix
val_metrics, val_cm = agreement_metrics(out.test_cllab, out.test_classlab)
val_metrics = {'normalized_stability': out.test_stability, **val_metrics}
for name, value in val_metrics.items():
    print(f"{name:>24} = {value:.3f}" if isinstance(value, float) else f"{name:>24} = {value}")
print("Confusion matrix (rows = hold-out clustering, columns = classifier predictions):")
print(val_cm)

# Change the .csv file names according to your model
pd.DataFrame.from_dict(val_metrics, orient='index', columns=['value']).to_csv('Val_metrics_summary_model_name.csv')
val_cm.to_csv('Val_confusion_matrix_model_name.csv')

# Save the labels of the hold-out subjects:
# - Holdout_clustering_labels: independent clustering of the hold-out set, matched to the training clusters
# - Assigned_training_cluster: training cluster assigned to each hold-out subject by the classifier
labels_val = pd.DataFrame({id_col: data_all[id_col][val_idx].to_numpy(),
                           'Holdout_clustering_labels': out.test_cllab,
                           'Assigned_training_cluster': out.test_classlab})
# Change the .csv file name according to your model
labels_val.to_csv('Labels_val_model_name.csv', index=False)