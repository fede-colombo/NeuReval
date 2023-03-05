import numpy as np
from sklearn.preprocessing import StandardScaler


def select_best(data, covariates, c, int_measure, preprocessing=None, select='max', nclust_range=None, combined_data=False):
    """
    Select the best number of clusters that minimizes/maximizes
    the internal measure selected.

    :param data: dataset.
    :type data: array-like
    :param c: clustering algorithm class.
    :type c: obj
    :param int_measure: internal measure function.
    :type int_measure: obj
    :param preprocessing: data reduction algorithm class, default None.
    :type preprocessing: obj
    :param select: it can be 'min', if the internal measure is to be minimized
        or 'max' if the internal measure should be macimized.
    :type select: str
    :param nclust_range: Range of clusters to consider, default None.
    :type nclust_range: list
    :param combined_data: define whether multimodal data are used as input features. 
        If True, different sets of covariates will be applied for each modality
        e.g. correction for TIV only for grey matter features. Default False
    :type combined_data: boolean value
    :return: internal score and best number of clusters.
    :rtype: float, int
    """
    data_array = np.array(data.iloc[:,2:])
    Y_array = np.array(data.iloc[:,1])
    cov_array = np.array(covariates.iloc[:,2:])
    
    if combined_data is not False:    
        num_col_DTI = len(data.loc[:,:"ACR"].T)
        
        data_DTI = data_array[:,num_col_DTI-3:]
        num_col_GM = len(data_array.T) - len(data_DTI.T)
        data_GM = data_array[:,:num_col_GM]
        cov_GM = cov_array
        cov_DTI = cov_array[:,:len(cov_array.T)-1]
    
        ### normalize the data and covariates
        scaler = StandardScaler()
        scaled_data = []
        scaled_covar = []
        for data, cov in [
                (data_GM, cov_GM),
                (data_DTI, cov_DTI)]:
            data = scaler.fit_transform(data)
            cov = scaler.fit_transform(cov)
            scaled_data.append(data)
            scaled_covar.append(cov)
        
        ### Adjust data for confounds of covariates
        cor_data = []
        for X, Y in [
            (scaled_covar[0], scaled_data[0]), 
            (scaled_covar[1], scaled_data[1])
        ]:
            beta = np.linalg.lstsq(X, Y, rcond=None)
            Xc = (Y.T - beta[0].T @ X.T).T
            cor_data.append(Xc)
        
        X_cor = np.concatenate((cor_data[0], cor_data[1]), axis=1)
    
    else:
        ### normalize the data and covariates
        scaler = StandardScaler()
        data_array = scaler.fit_transform(data_array)
        cov_array = scaler.fit_transform(cov_array)
         
        ### Adjust data for confounds of covariates
        Y = data_array
        X = cov_array
        beta = np.linalg.lstsq(X, Y, rcond=None)
        X_cor = (Y.T - beta[0].T @ X.T).T
    
    if preprocessing is not None:
        X_cor = preprocessing.fit_transform(X_cor)
    else:
        X_cor = X_cor
        
    if nclust_range is not None:
        scores = []
        label_vect = []
        for ncl in nclust_range:
            if 'n_clusters' in c.get_params().keys():
                c.n_clusters = ncl
            else:
                c.n_components = ncl
            label = c.fit_predict(X_cor)
            c.n_clusters = ncl
            label = c.fit_predict(X_cor)
            scores.append(int_measure(X_cor, label))
            label_vect.append(label)
    else:
        label = c.fit_predict(X_cor)
        best = int_measure(X_cor, label)
        return best, len([lab for lab in np.unique(label) if lab >= 0]), label
        
    if select == 'max':
        best = np.where(np.array(scores) == max(scores))[0]
    elif select == 'min':
        best = np.where(np.array(scores) == min(scores))[0]
    if len(set(label_vect[int(max(best))])) == nclust_range[int(max(best))]:
        return scores[int(max(best))], nclust_range[int(max(best))], label_vect[int(max(best))]
    else:
        return scores[int(max(best))], len(set(label_vect[int(max(best))])), label_vect[int(max(best))]

def select_best_bic_aic(data, covariates, c, preprocessing=None, score='bic', nclust_range=None, combined_data=False):
    """
    Function that selects the best number of clusters that minimizes BIC and AIC 
    in Gaussian Mixture Models.

    :param data: dataset.
    :type data: array-like
    :param c: clustering algorithm class.
    :type c: obj
    :param preprocessing: dimensionality reduction algorithm class, default None.
    :type preprocessing: obj
    :param score: type of score to compute. It could be 'bic' for BIC score, 'aic' for AIC score
    :type score: str
    :param nclust_range: Range of clusters to consider, default None.
    :type nclust_range: list
    :param combined_data: define whether multimodal data are used as input features. 
        If True, different sets of covariates will be applied for each modality
        e.g. correction for TIV only for grey matter features. Default False
    :type combined_data: boolean value
    :return: BIC or AIC score and best number of clusters.
    :rtype: float, int
    """
    
    data_array = np.array(data.iloc[:,2:])
    Y_array = np.array(data.iloc[:,1])
    cov_array = np.array(covariates.iloc[:,2:])
    
    if combined_data is not False:    
        num_col_DTI = len(data.loc[:,:"ACR"].T)
        
        data_DTI = data_array[:,num_col_DTI-3:]
        num_col_GM = len(data_array.T) - len(data_DTI.T)
        data_GM = data_array[:,:num_col_GM]
        cov_GM = cov_array
        cov_DTI = cov_array[:,:len(cov_array.T)-1]
    
        ### normalize the data and covariates
        scaler = StandardScaler()
        scaled_data = []
        scaled_covar = []
        for data, cov in [
                (data_GM, cov_GM),
                (data_DTI, cov_DTI)]:
            data = scaler.fit_transform(data)
            cov = scaler.fit_transform(cov)
            scaled_data.append(data)
            scaled_covar.append(cov)
        
        ### Adjust data for confounds of covariates
        cor_data = []
        for X, Y in [
            (scaled_covar[0], scaled_data[0]), 
            (scaled_covar[1], scaled_data[1])
        ]:
            beta = np.linalg.lstsq(X, Y, rcond=None)
            Xc = (Y.T - beta[0].T @ X.T).T
            cor_data.append(Xc)
        
        X_cor = np.concatenate((cor_data[0], cor_data[1]), axis=1)
    
    else:
        ### normalize the data and covariates
        scaler = StandardScaler()
        data_array = scaler.fit_transform(data_array)
        cov_array = scaler.fit_transform(cov_array)
         
        ### Adjust data for confounds of covariates
        Y = data_array
        X = cov_array
        beta = np.linalg.lstsq(X, Y, rcond=None)
        X_cor = (Y.T - beta[0].T @ X.T).T
    
    if preprocessing is not None:
        X_cor = preprocessing.fit_transform(X_cor)
    else:
        X_cor = X_cor
        
    if nclust_range is not None:
        scores=[]
        label_vect=[]
        for components in nclust_range:
            c.n_components = components
            label = c.fit_predict(X_cor)
            if score=='bic':
                bic_scores = c.bic(X_cor)
                scores.append(bic_scores)
            elif score=='aic':
                aic_scores = c.aic(X_cor)
                scores.append(aic_scores)
            label_vect.append(label)
    
    best = np.where(np.array(scores) == min(scores))[0]
    return scores[int(max(best))], len(set(label_vect[int(max(best))])), label_vect[int(max(best))]


def evaluate_best(data, covariates, c, int_measure, preprocessing=None, ncl=None, combined_data=False):
    """
    Function that, given a number of clusters, returns the corresponding internal measure
    for a dataset.

    :param data: dataset.
    :type data: array-like
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
    data_array = np.array(data.iloc[:,2:])
    Y_array = np.array(data.iloc[:,1])
    cov_array = np.array(covariates.iloc[:,2:])
    
    if combined_data is not False:    
        num_col_DTI = len(data.loc[:,:"ACR"].T)
        
        data_DTI = data_array[:,num_col_DTI-3:]
        num_col_GM = len(data_array.T) - len(data_DTI.T)
        data_GM = data_array[:,:num_col_GM]
        cov_GM = cov_array
        cov_DTI = cov_array[:,:len(cov_array.T)-1]
    
        ### normalize the data and covariates
        scaler = StandardScaler()
        scaled_data = []
        scaled_covar = []
        for data, cov in [
                (data_GM, cov_GM),
                (data_DTI, cov_DTI)]:
            data = scaler.fit_transform(data)
            cov = scaler.fit_transform(cov)
            scaled_data.append(data)
            scaled_covar.append(cov)
        
        ### Adjust data for confounds of covariates
        cor_data = []
        for X, Y in [
            (scaled_covar[0], scaled_data[0]), 
            (scaled_covar[1], scaled_data[1])
        ]:
            beta = np.linalg.lstsq(X, Y, rcond=None)
            Xc = (Y.T - beta[0].T @ X.T).T
            cor_data.append(Xc)
        
        X_cor = np.concatenate((cor_data[0], cor_data[1]), axis=1)
    
    else:
        ### normalize the data and covariates
        scaler = StandardScaler()
        data_array = scaler.fit_transform(data_array)
        cov_array = scaler.fit_transform(cov_array)
         
        ### Adjust data for confounds of covariates
        Y = data_array
        X = cov_array
        beta = np.linalg.lstsq(X, Y, rcond=None)
        X_cor = (Y.T - beta[0].T @ X.T).T
    
    if preprocessing is not None:
        X_cor = preprocessing.fit_transform(X_cor)
    else:
        X_cor = X_cor
        
    if 'n_clusters' in c.get_params().keys():
        c.n_clusters = ncl
    else:
        c.n_components = ncl
        label = c.fit_predict(X_cor)
    
    return int_measure(X_cor, label)
