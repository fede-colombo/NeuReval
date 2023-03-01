# NeuReval
A stability-based relative clustering validation method to determine the best number of clusters based on neuroimaging data. 

## Table of contents
1. [Project Overview](#Project_Overview)
2. [Installation and Requirements](#Installation)
3. [How to use NeuReval](#Use)
5. [Example](#Example)
6. [Notes](#Notes)
7. [References](#References)

## 1. Project overview <a name="Project_Overview"></a>
NeuReval implements a stability-based relative clustering approach within a cross-validation framework to identify the clustering solution that best replicates on unseen data. Compared to commonly used internal measures that rely on the inherent characteristics of the data, this approach has the advantage to identify clusters that are robust and reproducible in other samples of the same population. NeuReval is based on reval Python package (https://github.com/IIT-LAND/reval_clustering) and extends its application to neuroimaging data. For more details about the theoretical backgroud of reval, please see Landi et al. (2021).

This package allows to:
1. Select any classification algorithm from sklearn library;
2. Select a clustering algorithm with n_clusters parameter (i.e., KMeans, AgglomerativeClustering, and SpectralClustering), Gaussian Mixture Models with n_components parameter, and HDBSCAN density-based algorithm;
3. Perform (repeated) k-fold cross-validation to determine the best number of clusters;
4. Test the final model on an held-out dataset.

The following changes were made to reval to be performed on neuroimaging data:
1. Standardization and covariates adjustement within cross-validation
2. Combine different kind of neuroimaging data and apply different set of covariates to each neuroimaging modality
3. Implementation of data reduction techniques (e.g., PCA, UMAP) and optimization of their parameters within cross-validation
