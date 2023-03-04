# NeuReval
A stability-based relative clustering validation method to determine the best number of clusters based on neuroimaging data. 

## Table of contents
1. [Project Overview](#Project_Overview)
2. [Installation and Requirements](#Installation)
3. [How to use NeuReval](#Use)
    1. [Input structure] (#Input)
    2. [Grid-search cross-validation for parameters' tuning] (#Grid-search)
    3. [Run NeuReval with opitmized clustering/classifier/preprocessing algorithms] (#NeuReval)
    4. [Visualize results] (#Visualization)
    5. [Compute internal measures] (#Internal measures)
5. [Example](#Example)
6. [Notes](#Notes)
7. [References](#References)

## 1. Project overview <a name="Project_Overview"></a>
NeuReval implements a stability-based relative clustering approach within a cross-validation framework to identify the clustering solution that best replicates on unseen data. Compared to commonly used internal measures that rely on the inherent characteristics of the data, this approach has the advantage to identify clusters that are robust and reproducible in other samples of the same population. NeuReval is based on *reval* Python package (https://github.com/IIT-LAND/reval_clustering) and extends its application to neuroimaging data. For more details about the theoretical background of *reval*, please see Landi et al. (2021).

This package allows to:
1. Select any classification algorithm from *sklearn* library;
2. Select a clustering algorithm with *n_clusters* parameter (i.e., KMeans, AgglomerativeClustering, and SpectralClustering), Gaussian Mixture Models with *n_components* parameter, and HDBSCAN density-based algorithm;
3. Perform (repeated) k-fold cross-validation to determine the best number of clusters;
4. Test the final model on an held-out dataset.

The following changes were made to reval to be performed on neuroimaging data:
1. Standardization and covariates adjustement within cross-validation;
2. Combine different kind of neuroimaging data and apply different set of covariates to each neuroimaging modality;
3. Implementation of data reduction techniques (e.g., PCA, UMAP) and optimization of their parameters within cross-validation.

## 2. Installation and Requirements <a name="Installation"></a>
Work in progress

## 3. How to use NeuReval <a name="Use"></a>
### i. Input structure <a name="Input"></a>
First, NeuReval requires that input features and covariates are organized as file excel in the following way:

for database with **input features (database.xlsx)**:
- First column: subject ID
- Second column: diagnosis (e.g., patients=1, healthy controls=0)
- From the third column: features

**Notes**: in case you want to combine with difffusion tensor imaging (DTI) extracted tract-based features, please add them after all the other neuroimaging features. The first DTI feature should be "ACR".

Example of database structure for input features:

| Subject_ID  | Diagnosis | Feature_01 | Feature_02 |
| ------------| ----------| -----------| -----------|
| sub_0001    | 0         | 0.26649221 | 2.13888054 |
| sub_0002    | 1         | 0.32667590 | 0.67116539 |
| sub_0003    | 0         | 0.35406757 | 2.35572978 |

for database with **covariates (covariates.xlsx)**:
- First column: subject ID
- Second column: diagnosis (e.g., patients=1, healhty controls=0)
- From the third column: covariates

**Notes**: if you want to correct neuroimaging features also for total intracranial volume (TIV), please add it as the last column of the database.

Example of database structure for covariates:

| Subject_ID  | Diagnosis | Age | Sex | TIV     |
| ------------| ----------| ----|-----| --------|
| sub_0001    | 0         | 54  | 0   | 1213.76 |
| sub_0002    | 1         | 37  | 1   | 1372.93 |
| sub_0003    | 0         | 43  | 0   | 1285.88 |

Examples of fake datasets are provided in the folder **NeuReval/data**


