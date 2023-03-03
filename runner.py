#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:03:07 2022

@author: psicobiologia
"""
### Example of NeurReval application with GaussianMixture as clustering algorithm, SVC as classifier, and UMAP for dimensionality reduction ###

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

## STEP 1: GRID-SEARCH CROSS-VALIDATION CLUSTERING/CLASSIFIER/PREPROCESSING PARAMETERS

# Define working directories and create a new folder called 'models'
main_path = 'path/to/working/directory'
data_path = os.path.join(main_path,'models')

# Define output folder called 'results'. 
# Within 'results, create a subfolder corresponding to the input features (e.g., 'GM + FA').
# Within this subfolder, create another folder corresponding to the set of covariates used (e.g., 'age_sex_TIV')
out_dir = os.path.join(data_path,'results', 'feature_name', 'covariates')
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

# Import data and covariates files
database = 'database_name.xlsx'
covariates_file = 'covariates_name.xlsx'
# If there are multiple sheets, specify the name of the current sheet 
data = pd.read_excel(os.path.join(data_path,database), sheet_name='sheet_name')
covariates = pd.read_excel(os.path.join(data_path, covariates_file), sheet_name='sheet_name')

# Define clustering, classifier, and preprocessing parameters to be optimized.
# params should be a dictionary of the form {‘s’: {classifier parameter grid}, ‘c’: {clustering parameter grid}} 
# including the lists of classifiers and clustering methods to fit to the data.
# In case you want to optimize also preprocessing parameters (e.g., PCA or UMAP components), 
# specify {'preprocessing':{preprocessing parameter grid}} within the dictionary.
params = {'s': {'C': [0.01, 0.1, 1, 10, 100, 1000],
                'kernel':['linear', 'rbf']},
          'c': {'covariance_type':['full']},
          'preprocessing': {'n_components': list(range(2,6))}}

# Specify clustering (c), classifier (s), and preprocessing (preproc) algorithms
c = GaussianMixture(random_state=42)
s = SVC(random_state=42)
preproc = UMAP(random_state=42)
 
# Run ParamSelectionConfounds that implements grid search cross-validation to select the best combinations of parameters for fixed classifier/clustering/preprocessing algorithms.
# Parameters to be specified:
# cv: cross-validation folds
# nrand: number of random labelling iterations, default 10
# n_jobs: number of jobs to run in parallel, default (number of cpus - 1)
# iter_cv: number of repeated cross-validation, default 1
# clust_range: list with number of clusters, default None
# strat: stratification vector for cross-validation splits, default None
# combined_data: define whether multimodal data are used as input features. 
#               If True, different sets of covariates will be applied for each modality
#               e.g. correction for TIV only for grey matter features. Default False           
best_model = ParamSelectionConfounds(params, 
                                     cv=2, 
                                     s=s, 
                                     c=c,
                                     preprocessing=preproc,
                                     nrand=10,
                                     n_jobs=-1,
                                     iter_cv=1,
                                     clust_range=None,
                                     strat=None,
                                     combined_data=False)

best_model.fit(data,covariates)

# Save model's parameters in the output directory. Change file name to match the model you performed
best_results = best_model.best_param_
pkl.dump(best_results, open('./best_results_model_name.pkl', 'wb'))

    
## STEP 2: RUN FindBestClustCVConfounds WITH OPTIMIZED CLUSTERING/CLASSIFIER/PREPROCESSING PARAMETERS

# Define clusteirng and classifier algortihms with optimized parameters
best_s = SVC(C=1, kernel='rbf', random_state=42)
best_c = GaussianMixture(covariance_type='full', random_state=42)
# Uncomment the following line if you also optimized preprocessing steps
# best_preproc = UMAP(n_components=4, n_neighbors=30, min_dist=0.0, random_state=42)

# Initialize FindBestClustCVConfounds class. It performs (repeated) k-folds cross-validation to select the best number of clusters.
# Parameters to be specified:
# preprocessing: if not None, specify the preprocessing algorithm used for dimensionality reduction
# nfold: cross-validation folds
# nrand: number of random labelling iterations, default 10
# n_jobs: number of jobs to run in parallel, default (number of cpus - 1)
# clust_range: list with number of clusters, default None
findbestclust = FindBestClustCVConfounds(s=best_s,c=best_c, preprocessing=None, nfold=2, nrand=10,  n_jobs=-1, nclust_range=None)

# Run FindBestClustCVConfounds. It returns normlaized stability (metrics), best number of clusters (bestncl), and clusters' labels (tr_lab).
# Parameters to be specified:
# iter_cv: number of repeated cross-validation, default 1
# strat: stratification vector for cross-validation splits, default None
# combined_data: define whether multimodal data are used as input features. 
#               If True, different sets of covariates will be applied for each modality
#               e.g. correction for TIV only for grey matter features. Default False 
metrics, bestncl, tr_lab = findbestclust.best_nclust_confounds(data, covariates, iter_cv=1, strat_vect=None, combined_data=False)
val_results = list(metrics['val'].values())
val_results = np.array(val_results, dtype=object)

print(f"Best number of clusters: {bestncl}")
print(f"Validation set normalized stability (misclassification): {metrics['val'][bestncl]}")
print(f"Result accuracy (on test set): "
      f"{1-val_results[0,0]}")

# Normalized stability plot. For each number of clusters, normalized stabilities are represented for both training (dashed line) and validation sets (continuous line)
plot_metrics(metrics)

# Save database with cluster labels for post-hoc analyses
labels = pd.DataFrame(tr_lab, columns=['Clustering labels'])
data_all = pd.concat([data, labels], axis=1)
data_all.to_csv('Labels_model_name.csv', index=True)


## STEP 3: VISUALIZE CLUSTERS

# Make the required imports
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# Load the file csv with cluster labels previosly created and convert from a DataFrame to a numpy array
data_plot = np.array(pd.read_csv('path/to/database/with/clusters/labels', sep=','))

# Create new hsv colormaps in range of 0.3 (green) to 0.7 (blue)
hsv_modified = cm.get_cmap('hsv', 256)
newcmp = ListedColormap(hsv_modified(np.linspace(0.1, 0.63, 256)))

# Create a scatter plot of the first two feature of the database. Points are coloured based on clusters' membership
scatter = plt.scatter(data_plot[:,3], data_plot[:,4], c=data_plot[:, -1], cmap=newcmp)
plt.legend(*scatter.legend_elements())
plt.title('Clustering labels')
plt.show()



## STEP 4: COMPUTE INTERNAL MEASURES FOR COMPARISON

# Make the required imports
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
from neureval.internal_baselines_confounds import select_best, evaluate_best
from neureval.utils import kuhn_munkres_algorithm
import logging

# SILHOUETTE SCORE
# Use select_best to calcluate silhouette score. Specify silhouette_score as int_measure and 'max' as select parameter.
# It returns silhouette score (sil_score), number of clusters selected (sil_best), and silhouette labels (sil_labels)
logging.info("Silhouette score based selection")
sil_score, sil_best, sil_label = select_best(data, covariates, best_c, int_measure=silhouette_score, preprocessing=None,
                                                      select='max', nclust_range=None, combined_data=False)

logging.info(f"Best number of clusters (and scores): "
             f"{{{sil_best}({sil_score})}}")
logging.info(f'AMI (true labels vs clustering labels) training = '
             f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(sil_label)))}')
logging.info('\n\n')

# DAVIES-BOULDIN SCORE
# Use select_best to calcluate Davies-Bouldin score. Specify davies_bouldin_score as int_measure and 'min' as select parameter.
# It returns davies-bouldin score (db_score), number of clusters selected (db_best), and davies-bouldin labels (db_labels)
logging.info("Davies-Bouldin score based selection")
db_score, db_best, db_label = select_best(data, covariates, best_c, davies_bouldin_score,preprocessing=None,
                                                   select='min', nclust_range=None, combined_data=False)

logging.info(f"Best number of clusters (and scores): "
             f"{{{db_best}({db_score})}}")
logging.info(f'AMI (true labels vs clustering labels) training = '
             f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(db_label)))}')
logging.info('\n\n')

# AIC AND BIC
# For Gaussian Mixture Models, also BIC and AIC scores can be calculated
from neureval.internal_baselines_confounds import select_best_bic_aic, evaluate_best
from neureval.utils import kuhn_munkres_algorithm
import logging

# Calculate AIC score. Specify 'aic' as score parameter.
# It returns AIC score (aic_score), number of clusters selected (aic_best), and AIC labels (aic_labels)
logging.info("AIC score based selection")
aic_score, aic_best, aic_label = select_best_bic_aic(data, covariates, c=best_c, score='aic', preprocessing=None, nclust_range=None, combined_data=False)

logging.info(f"Best number of clusters (and scores): "
             f"{{{aic_best}({aic_score})}}")
logging.info(f'AMI (true labels vs clustering labels) training = '
             f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(aic_label)))}')
logging.info('\n\n')

# Calculate BIC score. Specify 'bic' as score parameter.
# It returns BIC score (bic_score), number of clusters selected (bic_best), and BIC labels (bic_labels)
logging.info("BIC score based selection")
bic_score, bic_best, bic_label = select_best_bic_aic(data, covariates, c=best_c, score='bic', preprocessing=None, nclust_range=None, combined_data=False)

logging.info(f"Best number of clusters (and scores): "
             f"{{{bic_best}({bic_score})}}")
logging.info(f'AMI (true labels vs clustering labels) training = '
             f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(bic_label)))}')
logging.info('\n\n')


 