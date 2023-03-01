import numpy as np
from sklearn.metrics import zero_one_loss
from neuroreval.utils import kuhn_munkres_algorithm
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(format='%(asctime)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


class RelativeValidationConfounds:
    """
    This class allows to perform the relative clustering validation procedure.
    A supervised algorithm is required to test cluster stability.
    Labels output from a clustering algorithm are used as true labels.

    :param s: initialized class for the supervised method.
    :type s: class
    :param c: initialized class for clustering algorithm.
    :type c: class
    :param preprocessing: initialized class for preprocessing method.
    :type preprocessing: class
    :param nrand: number of iterations to normalize cluster stability.
    :type nrand: int
    """

    def __init__(self, s, c, preprocessing=None, nrand=10):
        """
        Construct method.
        """
        self.class_method = s
        self.clust_method = c
        self.preproc_method = preprocessing
        self.nrand = nrand
        
    def GLMcorrection(self, X_train, Y_train, covar_train, X_test, covar_test, num_col_DTI=None):
        """
        Eliminate the confound of covariate, such as age and sex, from the input features.
        :param X_train: array, training features
        :param Y_train: array, training labels
        :param covar_train: array, training covariate data
        :param X_test: array, test labels
        :param covar_test: array, test covariate data
        :param num_col_DTI: index of dataset column corresponding to white matter features (i.e., "ACR" column)
        :return: corrected training & test feature data
        
        """
        if num_col_DTI is not None:
            X_train_DTI = X_train[:,num_col_DTI-3:]
            X_test_DTI = X_test[:,num_col_DTI-3:]
            num_col_GM = len(X_train.T) - len(X_train_DTI.T)
            X_train_GM = X_train[:,:num_col_GM]
            X_test_GM = X_test[:,:num_col_GM]
            Y_array = X_train[:,1]
            cov_train_GM = covar_train
            cov_test_GM = covar_test
            cov_train_DTI = covar_train[:,:len(covar_train.T)-1]
            cov_test_DTI = covar_test[:,:len(covar_test.T)-1]
            
            # normalize the covariate z-scoring
            train_data = []
            test_data = []
            scaler = StandardScaler()
            for train, test in [
                (X_train_GM, X_test_GM),
                (X_train_DTI, X_test_DTI),
                (cov_train_GM, cov_test_GM),
                (cov_train_DTI, cov_test_DTI)
                
            ]:
                train = scaler.fit_transform(train)
                test = scaler.transform(test)
                train_data.append(train)
                test_data.append(test)
            
            
            # Adjust data for confounds of covariate
            cor_train_data = []
            cor_test_data = []
            for X_tr, Y_tr, X_ts, Y_ts in [
                (train_data[2], train_data[0], test_data[2], test_data[0]),
                (train_data[3], train_data[1], test_data[3], test_data[1])
            ]:
                beta = np.linalg.lstsq(X_tr, Y_tr, rcond=None)
                X_cor_train = (Y_tr.T - beta[0].T @ X_tr.T).T
                X_cor_test =(Y_ts.T - beta[0].T @ X_ts.T).T
                cor_train_data.append(X_cor_train)
                cor_test_data.append(X_cor_test)
            
            X_train_cor = np.concatenate((cor_train_data[0], cor_train_data[1]), axis=1)
            X_test_cor = np.concatenate((cor_test_data[0], cor_test_data[1]), axis=1)
        
        else:
            # normalize the covariate z-scoring
            scaler = StandardScaler()
            train_data = []
            test_data = []
            for train, test in [
                    (X_train, X_test),
                    (covar_train, covar_test)]:
                train = scaler.fit_transform(train)
                test = scaler.transform(test)
                train_data.append(train)
                test_data.append(test)

            # Adjust data for confounds of covariate
            for X_train, Y_train, X_test, Y_test in [
                (train_data[1], train_data[0], test_data[1], test_data[0])
            ]:
                beta = np.linalg.lstsq(X_train, Y_train, rcond=None)
                X_train_cor = (Y_train.T - beta[0].T @ X_train.T).T
                X_test_cor = (Y_test.T - beta[0].T @ X_test.T).T
    
        return X_train_cor, X_test_cor

    def train(self, X_train_cor, tr_lab=None):
        """
        Method that performs training. It compares the clustering labels on training set
        (i.e., A(X) computed by :class:`reval.relative_validation.RelativeValidationConfounds.clust_method`) against
        the labels obtained from the classification algorithm
        (i.e., f(X), computed by :class:`reval.relative_validation.RelativeValidationConfounds.class_method`).
        It returns the misclassification error, the supervised model fitted to the data,
        and both clustering and classification labels.

        :param X_train_cor: counfound corrected training dataset.
        :type X_train_cor: ndarray, (n_samples, n_features)
        :param tr_lab: cluster labels found during CV for clustering methods with no `n_clusters` parameter.
            If not None the clustering method is not performed on the whole test set. Default None.
        :type tr_lab: list
        :return: misclassification error, fitted supervised model object, clustering and classification labels.
        :rtype: float, object, ndarray (n_samples,)
        """
        if self.preproc_method is not None:
            fitpreproc_tr = self.preproc_method.fit(X_train_cor)
            X_train_cor = fitpreproc_tr.transform(X_train_cor)
        else:
            X_train_cor = X_train_cor
            
        if tr_lab is None:
            clustlab_tr = self.clust_method.fit_predict(X_train_cor)  # A_k(X)
        else:
            clustlab_tr = tr_lab
        if len([cl for cl in clustlab_tr if cl >= 0]) == 0:
            logging.info(f"No clusters found during training with {self.clust_method}.")
            return None
        
        fitclass_tr = self.class_method.fit(X_train_cor, clustlab_tr)
        classlab_tr = fitclass_tr.predict(X_train_cor)
        misclass = zero_one_loss(clustlab_tr, classlab_tr)
        
        if self.preproc_method is not None:
            return misclass, fitclass_tr, clustlab_tr, fitpreproc_tr
        else:
            return misclass, fitclass_tr, clustlab_tr

    def test(self, X_test_cor, fit_model, fit_preproc=None):
        """
        Method that compares test set clustering labels (i.e., A(X'), computed by
        :class:`reval.relative_validation.RelativeValidationConfounds.clust_method`) against
        the (permuted) labels obtained through the classification algorithm fitted to the training set
        (i.e., f(X'), computed by
        :class:`reval.relative_validation.RelativeValidationConfounds.class_method`).
        It returns the misclassification error, together with
        both clustering and classification labels.

        :param X_test_cor: confound corrected test dataset.
        :type X_test_cor: ndarray, (n_samples, n_features)
        :param fit_model: fitted supervised model.
        :type fit_model: class
        :param fit_preproc: fitted preprocessing method.
        :type fit_preproc: class
        :return: misclassification error, clustering and classification labels.
        :rtype: float, dictionary of ndarrays (n_samples,)
        """
        if fit_preproc is not None:
            X_test_cor = fit_preproc.transform(X_test_cor)
        else:
            X_test_cor = X_test_cor
        
        clustlab_ts = self.clust_method.fit_predict(X_test_cor)  # A_k(X')
        if len([cl for cl in clustlab_ts if cl >= 0]) == 0:
            logging.info(f"No clusters found during testing with {self.clust_method}")
            return None
        classlab_ts = fit_model.predict(X_test_cor)
        bestperm = kuhn_munkres_algorithm(np.int32(classlab_ts), np.int32(clustlab_ts))  # array of integers
        misclass = zero_one_loss(classlab_ts, bestperm)
        
        return misclass, bestperm

    def rndlabels_traineval(self, X_train_cor, X_test_cor, train_labels, test_labels):
        """
        Method that performs random labeling on the training set
        (N times according to
        :class:`reval.relative_validation.RelativeValidationConfounds.nrand` instance attribute) and evaluates
        the fitted models on test set.

        :param X_train_cor: confound corrected training dataset.
        :type X_train_cor: ndarray, (n_samples, n_features)
        :param X_test_cor: confound corrected test dataset.
        :type X_test_cor: ndarray, (n_samples, n_features)
        :param train_labels: training set clustering labels.
        :type train_labels: ndarray, (n_samples,)
        :param test_labels: test set clustering labels.
        :type test_labels: ndarray, (n_samples,)
        :return: averaged misclassification error on the test set.
        :rtype: float
        """
        np.random.seed(0)
        shuf_tr = [np.random.permutation(train_labels)
                   for _ in range(self.nrand)]
        misclass_ts = list(map(lambda x: self._rescale_score_(X_train_cor, X_test_cor, x, test_labels), shuf_tr))
        return np.mean(misclass_ts)

    def _rescale_score_(self, xtr, xts, randlabtr, labts):
        """
        Private method that computes the misclassification error when predicting test labels
        with classification model fitted on training set with random labels.

        :param xtr: training dataset.
        :type xtr: ndarray, (n_samples, n_features)
        :param xts: test dataset.
        :type xts: ndarray, (n_samples, n_features)
        :param randlabtr: random labels.
        :type randlabtr: ndarray, (n_samples,)
        :param labts: test set labels.
        :type labts: ndarray, (n_samples,)
        :return: misclassification error.
        :rtype: float
        """
        self.class_method.fit(xtr, randlabtr)
        pred_lab = self.class_method.predict(xts)
        me_ts = zero_one_loss(pred_lab, kuhn_munkres_algorithm(np.int32(pred_lab), np.int32(labts)))
        return me_ts
