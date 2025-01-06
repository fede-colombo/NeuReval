# NeuReval
A stability-based relative clustering validation method to determine the best number of clusters based on neuroimaging data. 

## Table of contents
1. [Project Overview](#Project_Overview)
2. [Installation and Requirements](#Installation)
3. [How to use NeuReval](#Use)
    1. [Input structure](#Input)
    2. [Grid-search cross-validation for parameters' tuning](#Grid-search)
    3. [Run NeuReval with opitmized clustering/classifier/preprocessing algorithms](#NeuReval)
    4. [Compute internal measures](#Internal_measures)
    5. [External validation](#Validation)
4. [Example](#Example)
5. [Notes](#Notes)
6. [References](#References)

## 1. Project overview <a name="Project_Overview"></a>
*NeuReval* implements a stability-based relative clustering approach within a cross-validation framework to identify the clustering solution that best replicates on unseen data. Compared to commonly used internal measures that rely on the inherent characteristics of the data, this approach has the advantage to identify clusters that are robust and reproducible in other samples of the same population. *NeuReval* is based on *reval* Python package (https://github.com/IIT-LAND/reval_clustering) and extends its application to neuroimaging data. For more details about the theoretical background of *reval*, please see Landi et al. (2021).

This package allows to:
1. Select any classification algorithm from *sklearn* library;
2. Select a clustering algorithm with *n_clusters* parameter (i.e., KMeans, AgglomerativeClustering, and SpectralClustering), Gaussian Mixture Models with *n_components* parameter, and HDBSCAN density-based algorithm;
3. Perform (repeated) k-fold cross-validation to determine the best number of clusters;
4. Test the final model on an hold-out dataset.

The following changes were made to *reval* to be performed on neuroimaging data:
1. Standardization and covariates adjustement within cross-validation;
2. Combine different kind of neuroimaging data and apply different set of covariates to each neuroimaging modality;
3. Implementation of data reduction techniques (e.g., PCA, UMAP) and optimization of their parameters within cross-validation.

** UPDATES NEUREVAL 1.1.0 **
While the original version of *NeuReval* was limited to perform different confounds removal only for 2 different types of modalities (e.g., remove the effect of TIV only for grey matter features and not for the DTI ones), the updated version offers more flexibility in the definition and number of modalities and covariates.

## 2. Installation and Requirements <a name="Installation"></a>
The most recent version of *NeuReval* can be installed with pip using the following command:

```python
pip install neureval==1.1.0
```
Dependencies useful to install *NeuReval* can be found in requirements.txt

## 3. How to use NeuReval <a name="Use"></a>
### i. Input structure <a name="Input"></a>
*NeuReval* requires that input features and covariates are organized as file excel in the following way:

for database with **input features (database.xlsx)**:
- First column: subject ID
- Second column: diagnosis (e.g., patients=1, healthy controls=0). In case *NeuReval* is run on a single diagnostic group, provide a costant value for all subjects.
- From the third column: features

Example of database structure for input features:

| Subject_ID  | Diagnosis | Feature_01 | Feature_02 |
| ------------| ----------| -----------| -----------|
| sub_0001    | 0         | 0.26649221 | 2.13888054 |
| sub_0002    | 1         | 0.32667590 | 0.67116539 |
| sub_0003    | 0         | 0.35406757 | 2.35572978 |

for database with **covariates (covariates.xlsx)**:
- First column: subject ID
- Second column: diagnosis (e.g., patients=1, healhty controls=0). In case *NeuReval* is run on a single diagnostic group, provide a costant value for all subjects.
- From the third column: covariates

Example of database structure for covariates:

| Subject_ID  | Diagnosis | Age | Sex | TIV     |
| ------------| ----------| ----|-----| --------|
| sub_0001    | 0         | 54  | 0   | 1213.76 |
| sub_0002    | 1         | 37  | 1   | 1372.93 |
| sub_0003    | 0         | 43  | 0   | 1285.88 |


**Notes**: for external validation, a single data and corresponding covariates file including both training and validation sets should be created, with observations belonging to the validation set placed after those belonging to the training set

Example of data structure with validation set **(database_all.xlsx)**:

| Subject_ID  | Diagnosis | Feature_01 | Feature_02 |
| ------------| ----------| -----------| -----------|
| sub001_tr   | 0         | 0.26649221 | 2.13888054 |
| sub002_tr   | 1         | 0.32667590 | 0.67116539 |
| sub001_ts   | 0         | 0.28786749 | 2.16871673 |
| sub002_ts   | 0         | 0.26789256 | 2.13857829 |

Example of covariates structure with validation set **(covariates_all.xlsx)**:

| Subject_ID  | Diagnosis | Age | Sex | TIV     |
| ------------| ----------| ----|-----| --------|
| sub001_tr   | 0         | 54  | 0   | 1213.76 |
| sub002_tr   | 1         | 37  | 1   | 1372.93 |
| sub001_ts   | 0         | 32  | 0   | 1224.77 |
| sub002_ts   | 0         | 23  | 1   | 1224.91 |

Templates for both datasets are provided in the folder **NeuReval/example_data**.

### ii. Grid-search cross-validation for parameters' tuning <a name="Grid-search"></a>
First, parameters for fixed classifier/clustering/preprocessing algorithms can be optimized through a grid-search cross-validation. This can be done with the ```ParamSelectionConfounds``` class:
```python
ParamSelectionConfounds(params, cv, s, c, preprocessing, nrand=10, n_jobs=-1, iter_cv=1, strat=None, clust_range=None)
```
Parameters to be specified:
- **params**: dictionary of dictionaries of the form {‘s’: {classifier parameter grid}, ‘c’: {clustering parameter grid}} including the lists of classifiers and clustering methods to fit to the data. In case you want to optimize also preprocessing parameters (e.g., PCA or UMAP components), specify {'preprocessing':{preprocessing parameter grid}} within the dictionary.
- **cv**: cross-validation folds
- **s**: classifier object
- **c**: clustering object
- **preprocessing**: data reduction algorithm object
- **nrand**: number of random labelling iterations, default 10
- **n_jobs**: number of jobs to run in parallel, default (number of cpus - 1)
- **iter_cv**: number of repeated cross-validation, default 1
- **clust_range**: list with number of clusters, default None
- **strat**: stratification vector for cross-validation splits, default ```None```

Once the ```ParamSelectionConfounds``` class is initialized, the ```fit(data_tr, mod_tr, cov_tr, nclass=None)``` class method can be used to run grid-search cross-validation.
It returns the optimal number of clusters (i.e., minimum normalized stability), the corresponding normalized stability, and the selected classifier/clustering/preprocessing parameters.

### iii. Run NeuReval with opitmized clustering/classifier/preprocessing algorithms <a name="NeuReval"></a>
After the selection of the best clustering/classifier/preprocessing parameters through grid-search cross-vallidation, we can initalize the ```FindBestClustCVConfounds``` class to assess the normalized stability associated to the best clustering solution and the corresponding clusters' labels

```python
FindBestClustCVConfounds(s, c, preprocessing=None, nrand=10, nfold=2, n_jobs=-1, nclust_range=None)
```
Parameters to be specified:
- **s**: classifier object (with opitmized parameters)
- **c**: clustering object (with optimized parameters)
- **preprocessing**: data reduction algorithm object (with optimized parameters), default None
- **nrand**: number of random labelling iterations, default 10
- **nfold**: number of cross-validation folds, default 2
- **n_jobs**: number of jobs to run in parallel, default (number of cpus - 1)
- **clust_range**: list with number of clusters, default None

Once the class has been initialized, the ```best_nclust_confounds(data, modalities, covariates, iter_cv=1, strat_vect=None)``` method can be used to obtain the normalized stability, the number of clusters associated to the optimal clustering solution, and clusters' labels. It returns:
- **metrics**: normalized stability
- **bestncl**: best number of clusters
- **tr_lab**: clusters' labels

### iv. Compute internal measures <a name="Internal_measures"></a>
Together with normalized stability, *NeuReval* also allows to compute internal measures for comparisons between the stability-based relative validation and internal validation approaches. This can be done with the ```neureval.internal_baselines_confounds``` method and the function ```select_best``` to select the best number of clusters that maximize/minimize the selected internal measure:

```python
neureval.internal_baselines_confounds.select_best(data, modalities, covariates, c, int_measure, preprocessing=None, select='max', nclust_range=None)
 ```
 Parameters to be specified:
 - **data**: features dataset
 - **covariates**: covariates dataset
 - **c**: clustering algorithm class (with optimized parameters)
 - **int_measure**: internal measure function (e.g., silhouette score, Davies-Bouldin score)
 - **preprocessing**:  data reduction algorithm object (with optimized parameters), default ```None```
 - **select**: it can be ‘min’, if the internal measure is to be minimized or ‘max’ if the internal measure should be maximized
 - **nclust_range**: range of clusters to consider, default ```None```

**Notes**: in case Gaussian Mixture Model was implemented as clustering algorithm, the ```select_best_bic_aic``` function can be used to compute Akaike and Bayesian Information Criterion (AIC, BIC) and used them for model's selection.

### v. External validation <a name="Validation"></a>
To test the replicability of clustering solutions in an external dataset, the ```evaluate_confounds(data, modalities, covariates, tr_idx, val_idx, nclust=None, tr_lab=None)``` function within the ```neureval.best_nclust_cv_confounds``` method can be used.

Parameters to be specified:
- **data** : complete dataset with both training and hold-out sets
- **modalities** : dictionary specifying the datasets for each type of input feature
- **covariates**: dictionary specifying the covariates for each type of input feature
- **tr_idx**: index specifying the training dataset
- **val_idx**: index specifying the validation dataset
- **nclust**: best number of clusters, default None.
- **tr_lab**: clustering labels for the training set. If not None the clustering algorithm is not performed and the classifier is fitted.
              Available for clustering methods without `n_clusters` parameter. Default None
  
It returns accuracies in both training and validation sets and cluster labels in for the validation set

## 4. Example <a name="Example"></a>
An example of how to perform *NeuReval* can be found in the folder **NeuReval/scripts**. These codes show the application of *NeuReval* using Gaussian Mixture Model as clustering algorithm, Support Vector Machine as classifier, and UMAP as dimensionality reduction algorithm:

- **01_grid_search**: code to perform grid-search cross-validation for clustering/classifier/preprocessing parameters tuning
- **02_run_findbestclustcv**: code to perform *NeuReval* with the optimized clustering/classifier/preprocessing algorithms. This script also provides codes to compute different kind of internal measures and external validation in an hold-out dataset
- **03_visualization**: code to create a plot for clusters' representation

## 5. Notes <a name="Notes"></a>
*NeuReval* was developed in Python 3.8.10 and tested on ubuntu 20.04. In case of any issues in running *NeuReval* on other operating systems (i.e., Windows), you can send an email at the following address: fcolombo.italia@gmail.com

## 6. References <a name="References"></a>
Landi, I., Mandelli, V., & Lombardo, M. V. (2021). reval: A Python package to determine best clustering solutions with stability-based relative clustering validation. _Patterns_, 2(4), 100228.
