"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, 
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

### GRID-SEARCH CROSS-VALIDATION CLUSTERING/CLASSIFIER/PREPROCESSING PARAMETERS
# Example of NeurReval application with GaussianMixture as clustering algorithm, SVC as classifier, and UMAP for dimensionality reduction #

# Make the required imports
import pandas as pd
import numpy as np
from neureval.param_selection_confounds import ParamSelectionConfounds
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
import umap
import os
import pickle as pkl
import sys
from neureval.utils import kuhn_munkres_algorithm
import logging


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
data_file = 'database_name.xlsx'
cov_file = 'covariates_name.xlsx'
# If there are multiple sheets, specify the name of the current sheet 
data = pd.read_excel(os.path.join(data_path,data_file), sheet_name='sheet_name')
cov = pd.read_excel(os.path.join(data_path, cov_file), sheet_name='sheet_name')

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

# Define clustering, classifier, and preprocessing parameters to be optimized.
# params should be a dictionary of the form {‘s’: {classifier parameter grid}, ‘c’: {clustering parameter grid}} 
# including the lists of classifiers and clustering methods to fit to the data.
# In case you want to optimize also preprocessing parameters (e.g., PCA or UMAP components), 
# specify {'preprocessing':{preprocessing parameter grid}} within the dictionary.
params = {'s': {'C': [0.01, 0.1, 1, 10, 100, 1000],
                'kernel':['linear', 'rbf']},
          'c': {'covariance_type':['full']},
          'preprocessing': {'n_components': list(range(2,6))}}

# Define classifier and clustering methods
c = GaussianMixture(random_state=42)
s = SVC(random_state=42)

# If you want to perform dimensionality reduction, uncomment the following line. Change the dimensionality reduction algorithm accordingly (e.g., UMAP, PCA)
# preproc = umap.UMAP(random_state=42)

# Parameters selection
best_model = ParamSelectionConfounds(params, 
                                      cv=2, 
                                      s=s, 
                                      c=c,
                                      preprocessing=preproc,
                                      nrand=10,
                                      n_jobs=1,
                                      iter_cv=10,
                                      clust_range=list(range(2,6)),
                                      strat=None)

best_model.fit(data,modalities,covariates)

# Save model's parameters in the output directory. Change file name to match the model you performed
best_results = best_model.best_param_
pkl.dump(best_results, open('./best_results_prova.pkl', 'wb'))
# To open the results, uncomment the following line
# pd.read_pickle('./best_results_model_name.pkl')
