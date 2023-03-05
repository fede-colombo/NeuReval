# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 11:31:09 2023

@author: fcolo
"""

### GRID-SEARCH CROSS-VALIDATION CLUSTERING/CLASSIFIER/PREPROCESSING PARAMETERS
# Example of NeurReval application with GaussianMixture as clustering algorithm, SVC as classifier, and UMAP for dimensionality reduction #

# Make the required imports
import pandas as pd
import numpy as np
from neureval.param_selection_confounds import ParamSelectionConfounds
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import os
import pickle as pkl

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
# If you want to perform dimensionality reduction, uncomment the following line. Change the dimensionality reduction algorithm accordingly (e.g., UMAP, PCA)
# preproc = UMAP(random_state=42)
 
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
# To open the results, uncomment the following line
# pd.read_pickle('./best_results_model_name.pkl')
