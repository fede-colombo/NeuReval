# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 11:35:25 2023

@author: fcolo
"""

### RUN FindBestClustCVConfounds WITH OPTIMIZED CLUSTERING/CLASSIFIER/PREPROCESSING PARAMETERS

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
database = 'database.xlsx'
covariates_file = 'covariates.xlsx'
# If there are multiple sheets, specify the name of the current sheet 
data = pd.read_excel(os.path.join(data_path,database), sheet_name='sheet_name')
covariates = pd.read_excel(os.path.join(data_path, covariates_file), sheet_name='sheet_name')

# Define clusteirng and classifier algortihms with optimized parameters
s = SVC(C=1, kernel='rbf', random_state=42)
c = GaussianMixture(covariance_type='full', random_state=42)
# Uncomment the following line if you also optimized preprocessing steps
# best_preproc = UMAP(n_components=4, n_neighbors=30, min_dist=0.0, random_state=42)

# Initialize FindBestClustCVConfounds class. It performs (repeated) k-folds cross-validation to select the best number of clusters.
# Parameters to be specified:
# preprocessing: if not None, specify the preprocessing algorithm used for dimensionality reduction
# nfold: cross-validation folds
# nrand: number of random labelling iterations, default 10
# n_jobs: number of jobs to run in parallel, default (number of cpus - 1)
# clust_range: list with number of clusters, default None
findbestclust = FindBestClustCVConfounds(s,c, preprocessing=None, nfold=2, nrand=10,  n_jobs=-1, nclust_range=None)

# Run FindBestClustCVConfounds. It returns normlaized stability (metrics), best number of clusters (bestncl), and clusters' labels (tr_lab).
# Parameters to be specified:
# iter_cv: number of repeated cross-validation, default 1
# strat: stratification vector for cross-validation splits, default None
# combined_data: define whether multimodal data are used as input features. 
#               If True, different sets of covariates will be applied for each modality
#               e.g. correction for TIV only for grey matter features. Default False 
metrics, bestncl, tr_lab = findbestclust.best_nclust_confounds(data, covariates, iter_cv=10, strat_vect=None, combined_data=True)
val_results = list(metrics['val'].values())
val_results = np.array(val_results, dtype=object)

print(f"Best number of clusters: {bestncl}")
print(f"Validation set normalized stability (misclassification): {metrics['val'][bestncl]}")
print(f"Result accuracy (on test set): "
      f"{1-val_results[0,0]}")

# Normalized stability plot. For each number of clusters, normalized stabilities are represented for both training (dashed line) and validation sets (continuous line).
# Colors for training and validation sets can be changed (default: ('black', 'black')).
# To save the plot, specify the file name for saving figure in png format.
plot = plot_metrics(metrics, color=('black', 'black'), save_fig='plot_model_name.png')

# Save database with cluster labels for post-hoc analyses
labels = pd.DataFrame(tr_lab, columns=['Clustering labels'])
data_all = pd.concat([data, labels], axis=1)
data_all.to_csv('Labels_model_name.csv', index=True)



### COMPUTE INTERNAL MEASURES FOR COMPARISON

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
sil_score, sil_best, sil_label = select_best(data, covariates, c, silhouette_score, preprocessing=None,
                                                      select='max', nclust_range=list(range(2,6)), combined_data=True)

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
db_score, db_best, db_label = select_best(data, covariates, c, davies_bouldin_score,preprocessing=None,
                                                   select='min', nclust_range=list(range(2,6)), combined_data=True)

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
aic_score, aic_best, aic_label = select_best_bic_aic(data, covariates, c, score='aic', preprocessing=None, nclust_range=list(range(2,6)), combined_data=True)

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
bic_score, bic_best, bic_label = select_best_bic_aic(data, covariates, c, score='bic', preprocessing=None, nclust_range=list(range(2,6)), combined_data=True)

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

