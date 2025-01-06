
"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, 
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import math

def select_best(data, modalities, covariates, c, int_measure, preprocessing=None, select='max', nclust_range=None):
    """
    Select the best number of clusters that minimizes/maximizes
    the internal measure selected.

    :param data: dataset.
    :type data: array-like
    :param modalities: dictionary specifying the datasets for each type of input features
    :type modalities: dictionary
    :param covariates: dictionary specifying the covariates for each type of input features
    :type covariates: dictionary
    :param c: clustering algorithm class.
    :type c: obj
    :param int_measure: internal measure function.
    :type int_measure: obj
    :param preprocessing: data reduction algorithm class, default None.
    :type preprocessing: obj
    :param select: it can be 'min', if the internal measure is to be minimized
        or 'max' if the internal measure should be maximized.
    :type select: str
    :param nclust_range: Range of clusters to consider, default None.
    :type nclust_range: list
    :return: internal score and best number of clusters.
    :rtype: float, int
    """
    X_dic = {mod:None for mod in modalities}
    cov_dic = {mod:None for mod in modalities}
    
    for mod in modalities:
       X_dic[mod] = np.array(modalities[mod])
       cov_dic[mod] = np.array(covariates[mod])
       
    # normalize the covariate z-scoring
    X_scaled_dic = {mod:None for mod in modalities}
    cov_scaled_dic = {mod:None for mod in modalities}
    scaler = StandardScaler()
   
    for mod in modalities:
       X_scaled_dic[mod] = scaler.fit_transform(X_dic[mod])
       cov_scaled_dic[mod]= scaler.fit_transform(cov_dic[mod])
   
   
    # Adjust data for confounds of covariate
    X_cor_dic = {mod:None for mod in modalities}
    
    for mod in modalities:
       beta = np.linalg.lstsq(cov_scaled_dic[mod], X_scaled_dic[mod], rcond=None)
       X_cor_dic[mod] = (X_scaled_dic[mod].T - beta[0].T @ cov_scaled_dic[mod].T).T
    
    corr_data = np.concatenate([np.array(X_cor_dic[mod]) for mod in X_cor_dic], axis=1)
    
    if preprocessing is not None:
        corr_data = preprocessing.fit_transform(corr_data)
    else:
        corr_data = corr_data
        
    if nclust_range is not None:
        scores = []
        label_vect = []
        for ncl in nclust_range:
            if 'n_clusters' in c.get_params().keys():
                c.n_clusters = ncl
            else:
                c.n_components = ncl
            label = c.fit_predict(corr_data)
            scores.append(int_measure(corr_data, label))
            label_vect.append(label)
    else:
        label = c.fit_predict(corr_data)
        best = int_measure(corr_data, label)
        return best, len([lab for lab in np.unique(label) if lab >= 0]), label
        
    if select == 'max':
        best = np.where(np.array(scores) == max(scores))[0]
    elif select == 'min':
        best = np.where(np.array(scores) == min(scores))[0]
    if len(set(label_vect[int(max(best))])) == nclust_range[int(max(best))]:
        return scores[int(max(best))], nclust_range[int(max(best))], label_vect[int(max(best))]
    else:
        return scores[int(max(best))], len(set(label_vect[int(max(best))])), label_vect[int(max(best))]


def select_best_bic_aic(data, modalities, covariates, c, preprocessing=None, score='bic', nclust_range=None):
    """
    Function that selects the best number of clusters that minimizes BIC and AIC 
    in Gaussian Mixture Models.

    :param data: dataset.
    :type data: array-like
    :param modalities: dictionary specifying the datasets for each type of input features
    :type modalities: dictionary
    :param covariates: dictionary specifying the covariates for each type of input features
    :type covariates: dictionary
    :param c: clustering algorithm class.
    :type c: obj
    :param int_measure: internal measure function.
    :type int_measure: obj
    :param preprocessing: data reduction algorithm class, default None.
    :type preprocessing: obj
    :param score: type of score to compute. It could be 'bic' for BIC score, 'aic' for AIC score
    :type score: str
    :param nclust_range: Range of clusters to consider, default None.
    :type nclust_range: list
    :return: BIC or AIC score and best number of clusters.
    :rtype: float, int
    """
    
    X_dic = {mod:None for mod in modalities}
    cov_dic = {mod:None for mod in modalities}
    
    for mod in modalities:
       X_dic[mod] = np.array(modalities[mod])
       cov_dic[mod] = np.array(covariates[mod])
       
    # normalize the covariate z-scoring
    X_scaled_dic = {mod:None for mod in modalities}
    cov_scaled_dic = {mod:None for mod in modalities}
    scaler = StandardScaler()
   
    for mod in modalities:
       X_scaled_dic[mod] = scaler.fit_transform(X_dic[mod])
       cov_scaled_dic[mod]= scaler.fit_transform(cov_dic[mod])
   
   
    # Adjust data for confounds of covariate
    X_cor_dic = {mod:None for mod in modalities}
    
    for mod in modalities:
       beta = np.linalg.lstsq(cov_scaled_dic[mod], X_scaled_dic[mod], rcond=None)
       X_cor_dic[mod] = (X_scaled_dic[mod].T - beta[0].T @ cov_scaled_dic[mod].T).T
    
    corr_data = np.concatenate([np.array(X_cor_dic[mod]) for mod in X_cor_dic], axis=1)
    
    if preprocessing is not None:
        corr_data = preprocessing.fit_transform(corr_data)
    else:
        corr_data = corr_data
        
    if nclust_range is not None:
        scores=[]
        label_vect=[]
        for components in nclust_range:
            c.n_components = components
            label = c.fit_predict(corr_data)
            if score=='bic':
                bic_scores = c.bic(corr_data)
                scores.append(bic_scores)
            elif score=='aic':
                aic_scores = c.aic(corr_data)
                scores.append(aic_scores)
            label_vect.append(label)
    
    best = np.where(np.array(scores) == min(scores))[0]
    return scores[int(max(best))], len(set(label_vect[int(max(best))])), label_vect[int(max(best))]


def evaluate_best(data, modalities, covariates, c, int_measure, ncl=None):
    """
    Function that, given a number of clusters, returns the corresponding internal measure
    for a dataset.

    :param data: dataset.
    :type data: array-like
    :param modalities: dictionary specifying the datasets for each type of input features
    :type modalities: dictionary
    :param covariates: dictionary specifying the covariates for each type of input features
    :type covariates: dictionary
    :param c: clustering algorithm class.
    :type c: obj
    :param int_measure: internal measure function.
    :type int_measure: obj
    :param preprocessing:  dimensionality reduction algorithm class, default None.
    :type preprocessing: obj
    :param ncl: number of clusters.
    :type ncl: int
    :param combined_data: define whether multimodal data are used as input features. 
        If True, different sets of covariates will be applied for each modality
        e.g. correction for TIV only for grey matter features. Default False
    :type combined_data: boolean value
    :return: internal score.
    :rtype: float
    """
    X_dic = {mod:None for mod in modalities}
    cov_dic = {mod:None for mod in modalities}
    
    for mod in modalities:
       X_dic[mod] = np.array(modalities[mod])
       cov_dic[mod] = np.array(covariates[mod])
       
    # normalize the covariate z-scoring
    X_scaled_dic = {mod:None for mod in modalities}
    cov_scaled_dic = {mod:None for mod in modalities}
    scaler = StandardScaler()
   
    for mod in modalities:
       X_scaled_dic[mod] = scaler.fit_transform(X_dic[mod])
       cov_scaled_dic[mod]= scaler.fit_transform(cov_dic[mod])
   
   
    # Adjust data for confounds of covariate
    X_cor_dic = {mod:None for mod in modalities}
    
    for mod in modalities:
       beta = np.linalg.lstsq(cov_scaled_dic[mod], X_scaled_dic[mod], rcond=None)
       X_cor_dic[mod] = (X_scaled_dic[mod].T - beta[0].T @ cov_scaled_dic[mod].T).T

    corr_data = [np.array(X_cor_dic[mod]) for mod in X_cor_dic]
    
    if 'n_clusters' in c.get_params().keys():
        c.n_clusters = ncl
    else:
        c.n_components = ncl
        label = c.fit_predict(corr_data)
    
    return int_measure(corr_data, label)
