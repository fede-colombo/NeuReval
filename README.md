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
    6. [Agreement metrics](#Metrics)
    7. [Site harmonization with ComBat](#Harmonization)
    8. [Several rows per subject (longitudinal data)](#Groups)
    9. [Longitudinal states and transitions](#States)
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

**UPDATES NEUREVAL 1.1.0**
While the original version of *NeuReval* was limited to perform different confounds removal only for 2 different types of modalities (e.g., remove the effect of TIV only for grey matter features and not for the DTI ones), the updated version offers more flexibility in the definition and number of modalities and covariates.

## 2. Installation and Requirements <a name="Installation"></a>
The most recent version of *NeuReval* can be installed with pip using the following command:

```python
pip install neureval
```
*NeuReval* requires Python 3.10 or later. The dependencies are installed automatically.

To develop *NeuReval* or run its tests, clone the repository and install it in editable mode, together with the dependencies used by the tests (listed in requirements.txt):

```python
pip install -e .
pip install -r requirements.txt
pytest tests/
```

## 3. How to use NeuReval <a name="Use"></a>
### i. Input structure <a name="Input"></a>
*NeuReval* requires two files, one with the input features and one with the covariates, with one row per subject (or one row per subject and time point, see [Several rows per subject](#Groups) below) and the rows in the same order in both files. Files can be CSV (.csv) or Excel (.xlsx, .xls) and are read with ```neureval.utils.load_table```.

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
- From the third column: covariates. Covariates must be numerical: encode categorical variables (e.g., sex as 0/1; variables with more than two levels as separate 0/1 columns).

Example of database structure for covariates:

| Subject_ID  | Diagnosis | Age | Sex | TIV     |
| ------------| ----------| ----|-----| --------|
| sub_0001    | 0         | 54  | 0   | 1213.76 |
| sub_0002    | 1         | 37  | 1   | 1372.93 |
| sub_0003    | 0         | 43  | 0   | 1285.88 |

The features of each modality (e.g., grey matter, FA) and the covariates to regress out for each modality are then selected as dictionaries of columns (see the scripts in **NeuReval/scripts**), e.g. ```modalities = {'GM': data.iloc[:, 2:124], 'FA': data.iloc[:, 124:]}``` and ```covariates = {'GM': cov[['Age', 'Sex', 'TIV']], 'FA': cov[['Age', 'Sex']]}```.

**Multi-site data.** To harmonize data from different sites or scanners with ComBat (see [Site harmonization with ComBat](#Harmonization)), add a column with the site of each subject to the covariates file (any labels, e.g. site names). The site is used for harmonization, not for the confound correction: **exclude it from the covariates dictionary** (e.g., ```cov[['Age', 'Sex', 'TIV']]``` rather than ```cov.iloc[:, 2:]```), and pass it separately as ```site```. The covariates whose effects ComBat must preserve (e.g., age and sex) are selected in the same way (```harmonize_covariates```).

Example of covariates structure for multi-site data:

| Subject_ID  | Diagnosis | Site   | Age | Sex | TIV     |
| ------------| ----------| -------|-----|-----| --------|
| sub_0001    | 0         | Milan  | 54  | 0   | 1213.76 |
| sub_0002    | 1         | Milan  | 37  | 1   | 1372.93 |
| sub_0003    | 0         | London | 43  | 0   | 1285.88 |

**Several rows per subject** (e.g. longitudinal data in long format, one row per subject and time point; see [Several rows per subject](#Groups)). Each row of both files is a subject at one time point: the subject ID is repeated on each row of the subject, and subjects with missing time points have fewer rows (do not add empty rows). Add a column with the time point (e.g. age window) to the covariates file. The time point is not a feature: to remove the average differences between time points, add it to the covariates dictionary as dummy variables (e.g. ```pd.get_dummies(cov['Window'], prefix='Window', drop_first=True, dtype=float)```, see the scripts). Pass the subject ID column as ```groups```, so that all the rows of a subject are kept in the same cross-validation fold (```group_col``` in scripts 01 and 02).

Example of database structure for input features with several rows per subject:

| Subject_ID  | Diagnosis | Feature_01 | Feature_02 |
| ------------| ----------| -----------| -----------|
| sub_0001    | 0         | 0.26649221 | 2.13888054 |
| sub_0001    | 0         | 0.27801345 | 2.10457211 |
| sub_0001    | 0         | 0.25433902 | 2.20915630 |
| sub_0002    | 1         | 0.32667590 | 0.67116539 |
| sub_0002    | 1         | 0.33015478 | 0.70213846 |

Example of covariates structure with several rows per subject (sub_0002 has no data for window 3):

| Subject_ID  | Diagnosis | Window | Sex |
| ------------| ----------| -------|-----|
| sub_0001    | 0         | 1      | 0   |
| sub_0001    | 0         | 2      | 0   |
| sub_0001    | 0         | 3      | 0   |
| sub_0002    | 1         | 1      | 1   |
| sub_0002    | 1         | 2      | 1   |

**External validation.** Create a single features file and a single covariates file including both training and validation subjects, in any order, and a file with the IDs of the training subjects (e.g., **training_ids.csv**), in a column with the same name as the ID column. The file with the training labels saved by script 02 (section 1) can be used. All the subjects that are not in the training list are used for validation.

Example of data structure with validation set **(database_all.xlsx)**:

| Subject_ID  | Diagnosis | Feature_01 | Feature_02 |
| ------------| ----------| -----------| -----------|
| sub_0001    | 0         | 0.26649221 | 2.13888054 |
| sub_0101    | 0         | 0.28786749 | 2.16871673 |
| sub_0002    | 1         | 0.32667590 | 0.67116539 |
| sub_0102    | 0         | 0.26789256 | 2.13857829 |

Example of covariates structure with validation set **(covariates_all.xlsx)**, here with the site column for multi-site data:

| Subject_ID  | Diagnosis | Site   | Age | Sex | TIV     |
| ------------| ----------| -------|-----|-----| --------|
| sub_0001    | 0         | Milan  | 54  | 0   | 1213.76 |
| sub_0101    | 0         | Milan  | 32  | 0   | 1224.77 |
| sub_0002    | 1         | London | 37  | 1   | 1372.93 |
| sub_0102    | 0         | London | 23  | 1   | 1224.91 |

Example of list of training subjects **(training_ids.csv)**:

| Subject_ID  |
| ------------|
| sub_0001    |
| sub_0002    |

With **several rows per subject**, the list of training subjects contains the subject IDs (repeated IDs are allowed, e.g. when using the training labels file, which has one row per subject and time point): all the rows of these subjects are used for training, all the rows of the other subjects for validation (```long_format = True``` in section 3 of script 02).

For **hold-out validation with site harmonization**, the validation subjects must come from sites that are also in the training set. For **external validation in a new site**, also create a file with the IDs of the calibration subjects of the new site (e.g., **calibration_ids.csv**, same structure as training_ids.csv): they are used only to estimate the site effects of the new site and are excluded from the evaluation. They should be comparable to the training population (e.g., healthy controls in both).

Templates for features and covariates files are provided in the folder **NeuReval/example_data** (without the optional site column).

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
 - **site**, **harmonize_covariates**: site harmonization with ComBat (see [Site harmonization with ComBat](#Harmonization)), default ```None```. Use the same setting as in the *NeuReval* analysis, so that the comparison is fair. Here ComBat is estimated on all subjects, as the internal measures are computed on the whole dataset

**Notes**: in case Gaussian Mixture Model was implemented as clustering algorithm, the ```select_best_bic_aic``` function can be used to compute Akaike and Bayesian Information Criterion (AIC, BIC) and used them for model's selection. The ```evaluate_best``` function returns the internal measure for a given number of clusters (```ncl```). The clustering object passed to these functions is not modified.

### v. External validation <a name="Validation"></a>
To test the replicability of clustering solutions in an external dataset, the ```evaluate_confounds``` function within the ```neureval.best_nclust_cv_confounds``` method can be used:

```python
out = findbestclust.evaluate_confounds(data, modalities, covariates, tr_idx, val_idx, nclust=None, tr_lab=None)
 ```
Covariates are regressed out and preprocessing (if any) is fitted on the training set only, then applied to the validation set. A classifier is trained on the training clusters, the validation set is clustered independently, and the two labelings are compared.

Parameters to be specified:
- **data** : complete dataset with both training and hold-out sets
- **modalities** : dictionary specifying the datasets for each type of input feature
- **covariates**: dictionary specifying the covariates for each type of input feature
- **tr_idx**: index specifying the training dataset
- **val_idx**: index specifying the validation dataset
- **nclust**: best number of clusters, default None.
- **tr_lab**: clustering labels for the training set. If not None the clustering algorithm is not performed and the classifier is fitted.
              Available for clustering methods without `n_clusters` parameter. Default None
  
It returns (or None if no clusters are identified):

- **train_cllab, train_acc**: clustering labels and classifier accuracy in the training set
- **test_cllab**: labels of the independent clustering of the validation set, matched to the training clusters
- **test_acc**: agreement between the validation clustering and the classifier predictions (raw accuracy: its chance level depends on the number of clusters)
- **test_classlab**: training cluster assigned by the classifier to each validation subject. Use these labels to assign new subjects to the training clusters
- **test_stability**: normalized stability in the validation set, computed as in cross-validation: 0 means perfect replication, values close to 1 mean no better than chance. Use this value, rather than test_acc, to judge replicability

The script 02_run_findbestclustcv shows how to define the training and validation sets from the subjects' IDs, so that the order of the rows in the input files does not matter.


### vi. Agreement metrics <a name="Metrics"></a>

NeuReval measures how well a classifier trained on the training clusters predicts the clusters found independently in a validation set (a cross-validation fold or a hold-out set). Besides the accuracy used by NeuReval, the neureval.metrics module describes the same agreement with metrics that correct for chance and/or for clusters of different sizes:


```python
from neureval.metrics import agreement_metrics, cv_agreement_metrics

# Cross-validation: one row per fold and number of clusters, and mean/SD for each number of clusters
cv_per_fold, cv_summary = cv_agreement_metrics(findbestclust.cv_results_)

# External validation
out = findbestclust.evaluate_confounds(data_all, modalities_val, covariates_val, tr_idx, val_idx, nclust=bestncl)
val_metrics, val_confusion_matrix = agreement_metrics(out.test_cllab, out.test_classlab)
```
Metrics:
- **accuracy**: proportion of subjects with the same label (as in NeuReval)
- **balanced_accuracy**: mean recall over the clusters, not inflated by large clusters
- **cohen_kappa**: agreement corrected for chance (0 = chance, 1 = perfect)
- **ARI**: adjusted Rand index (0 = chance, 1 = perfect)
- **NMI, AMI**: normalized and adjusted mutual information
- **majority_class_baseline**: accuracy obtained by assigning every subject to the largest cluster, a reference value for the accuracy
- **cluster sizes** and **confusion matrix** (rows: clustering labels, columns: classifier predictions)

Compare the accuracy with the majority class baseline: with clusters of different sizes, an accuracy that looks high can be close to (or below) what is obtained by assigning every subject to the largest cluster.

### vii. Site harmonization with ComBat <a name="Harmonization"></a>
With data from more than one site or scanner, site effects can be stronger than the differences between subgroups: clusters may then reflect the sites rather than biology. They can even replicate well in a hold-out set from the same sites, because the site effects are there too. *NeuReval* can harmonize the data with ComBat (```neureval.harmonization.ComBat```), removing additive and multiplicative site effects from each feature while preserving the effects of specified covariates (e.g. age, sex).

ComBat is always **estimated on the training subjects only** and then applied to the validation subjects, so that no information leaks from the validation set. It is applied separately to each modality, before the correction for covariates. Results are identical to the reference implementation [neuroCombat](https://github.com/Jfortin1/neuroCombat) (parametric empirical Bayes).

**Cross-validation** (```best_nclust_confounds``` and grid search ```ParamSelectionConfounds.fit```): give the site of each subject and, optionally, the covariates to preserve. ComBat is estimated on the training set of each fold, and the folds are stratified by site so that every site is in every training fold.

```python
metrics, bestncl, tr_lab = findbestclust.best_nclust_confounds(data, modalities, covariates, iter_cv=10,
                                                               site=site, harmonize_covariates=cov[['Age', 'Sex']])
```

**Hold-out validation** (validation subjects from the same sites as the training subjects): the harmonization estimated in training is applied to the hold-out subjects.

```python
out = findbestclust.evaluate_confounds(data_all, modalities_val, covariates_val, tr_idx, val_idx, nclust=bestncl,
                                       site=site, harmonize_covariates=harmonize_covariates)
```

**External validation in new sites** (not in the training set): ComBat cannot harmonize a site it has never seen, and *NeuReval* raises an error. Give a calibration sample of the new site (```calibration_idx```): the model estimated in training is kept fixed, only the location and scale effects of the new site are estimated from the calibration subjects, which are then excluded from the evaluation. The calibration subjects should be comparable to the training population (e.g. healthy controls in both): otherwise real differences between the populations are removed as if they were site effects.

```python
out = findbestclust.evaluate_confounds(data_all, modalities_val, covariates_val, tr_idx, val_idx, nclust=bestncl,
                                       site=site, harmonize_covariates=harmonize_covariates,
                                       calibration_idx=calibration_idx)
```

Parameters:
- **site**: site of each subject (any labels)
- **harmonize_covariates**: covariates whose effects ComBat must preserve, numerical (encode categorical variables, e.g. sex as 0/1). A single table for all modalities, or a dictionary with one table per modality (same keys as ```modalities```). For non-linear effects (e.g. of age), add the corresponding terms (e.g. age squared)
- **calibration_idx**: calibration subjects of new sites, for external validation only
- **harmonization** (class parameter): ComBat options, e.g. ```FindBestClustCVConfounds(..., harmonization=ComBat(eb=True, mean_only=False))```. Default: ComBat with empirical Bayes when a site is given

Each site needs at least 2 subjects in each training set (in practice many more: site effects are estimated from these subjects).

Script **02_run_findbestclustcv** shows how to use site harmonization in cross-validation (section 1) and in hold-out or external validation (section 3).

### viii. Several rows per subject (longitudinal data) <a name="Groups"></a>
Some data have more than one row per subject, e.g. longitudinal data in **long format**, with one row per subject and time point (or age window). The files have the same structure as in [Input structure](#Input), with the subject ID repeated on each row of the subject and a column with the time point; subjects with missing time points simply have fewer rows. The clusters are then defined on the rows (e.g. states that a subject can occupy at different ages), and the sequence of labels of each subject can be analysed afterwards (e.g. transitions between states).

The rows of the same subject are similar to each other. If the cross-validation splits them between the training and the validation fold, the classifier is evaluated on subjects it has already seen and the stability is **overestimated**: clusters may appear stable even when there are none. Give the subject ID of each row with ```groups```: all the rows of a subject are kept in the same fold, so that each validation fold contains only new subjects.

```python
groups = data['Subject_ID'].to_numpy()
metrics, bestncl, tr_lab = findbestclust.best_nclust_confounds(data, modalities, covariates, iter_cv=10,
                                                               groups=groups)
best_model.fit(data, modalities, covariates, groups=groups)          # grid search
```

```groups``` can be combined with ```strat_vect``` and with site harmonization (```site```): the folds are then stratified as far as possible while keeping the subjects together. Repeated cross-validation (```iter_cv```) uses a different random split of the subjects at each repetition.

**Effect of age.** To remove the average differences between time points, add the time point to the covariates of each modality as dummy variables, e.g. ```pd.get_dummies(cov['Window'], prefix='Window', drop_first=True, dtype=float)```: the confound correction, estimated on the training set of each fold, then removes them.

**External validation**: select the training subjects by ID, so that all their rows are in the training set, and give ```groups``` to ```evaluate_confounds```: an error is raised if a subject has rows in both the training and the validation (or calibration) set.

```python
out = findbestclust.evaluate_confounds(data_all, modalities_val, covariates_val, tr_idx, val_idx, nclust=bestncl,
                                       groups=data_all['Subject_ID'].to_numpy())
```

The internal measures do not use cross-validation and need no groups: they treat each row as an observation. The same holds for ComBat, which treats the rows of a subject as independent observations; methods that model repeated measures in harmonization (e.g. longitudinal ComBat, Beer et al., 2020) are not implemented.

Scripts **01_grid_search** and **02_run_findbestclustcv** show how to use groups (```group_col``` and, in section 3 of script 02, ```long_format```).

### ix. Longitudinal states and transitions <a name="States"></a>
With longitudinal data in long format, *NeuReval* can define **states** (clusters of rows: a subject can be in different states at different time points) and the module ```neureval.longitudinal``` describes how subjects move between states. It was developed for counts of different types of events in time windows (e.g. infections of different types in age windows), but most functions work with any states.

**States or trajectories?** Longitudinal data can be clustered in two ways, which answer different questions:

| | Trajectories | States |
| - | ------------ | ------ |
| Data | one row per subject, the time points side by side (wide format) | one row per subject and time point (long format) |
| A cluster is | a typical course over time (e.g. "high in early childhood, then typical") | a profile at one time point (e.g. "mostly ear infections") |
| Labels per subject | one, for the whole follow-up | one for each time point, can change over time |
| Time | within the cluster (shape of the trajectory) | between clusters (transitions between states) |
| Question | which courses over time exist? | which profiles exist, and how do subjects move between them? |
| *NeuReval* | usual analysis, without groups, on the shape of the trajectories (```level_shape_features```) | ```groups``` and the functions below |

With trajectories, each subject belongs to one cluster ("children in the persistent-high trajectory"). With states, a subject is in a state at a given time ("children in the respiratory state at 3-6 years"): always report the time point, and report the prevalence of the states as a proportion of the rows of each time point. When clustering trajectories, the clusters may reflect only the overall level (e.g. low, medium, high at all time points), as the time points are usually positively correlated: cluster the shape of the trajectories, separated from their level with ```level_shape_features```, and check whether the cluster profiles are parallel. Sequences of states can also be clustered afterwards, to group subjects with similar courses.

| Function | What it does |
| -------- | ------------ |
| ```composition_features``` | Describes each row by the composition of its counts (which types prevail, as isometric log-ratio coordinates) and, separately, by its total, so that the states are not defined only by the total number of events. Rows without counts get missing values. Each row is transformed on its own: no leakage in cross-validation |
| ```select_rows``` | Selects the same rows from data, covariates, groups, etc. (e.g. the rows with counts, to be clustered) |
| ```add_reference_state``` | Combines the states found by *NeuReval* with a state defined a priori for the other rows: 0 = reference (e.g. no counts), 1..k = states found by *NeuReval* |
| ```state_sequences``` | Sequence of states of each subject: one row per subject, one column per time window |
| ```transition_matrices``` | Transitions between consecutive time windows (number of subjects or probabilities; the diagonal is the persistence) |
| ```transition_table``` | One row per transition, with subject-level variables (e.g. polygenic scores), for regression models of the transitions |
| ```sequence_features``` | Features describing the sequence of states of each subject (one-hot states by time window, or time spent in each state), to cluster subjects with similar sequences with *NeuReval* (one row per subject, no groups) |
| ```sequence_profiles``` | Typical course of each cluster of sequences: proportion of its subjects in each state in each time window |
| ```to_wide``` | One row per subject and one column per time window, from data in long format |
| ```level_shape_features``` | Decomposes the trajectory of each subject into its level (mean over time) and its shape (linear: increase or decrease; quadratic: peak or dip; ...) with orthonormal polynomial contrasts, to cluster trajectories by their shape. Missing time points are allowed; each subject is transformed on its own: no leakage |

```python
from neureval.longitudinal import (composition_features, select_rows, add_reference_state,
                                   transition_matrices, transition_table)

features = composition_features(counts[['Respiratory', 'Ear', 'Gastro']])
informative = features.notna().all(axis=1).to_numpy()                # rows with at least one count
feat, cov_inf, groups_inf = select_rows(informative, features, covariates, groups)
metrics, k, labels = findbestclust.best_nclust_confounds(feat, {'counts': feat}, {'counts': cov_inf},
                                                         iter_cv=10, groups=groups_inf)
states = add_reference_state(labels, informative)                   # 0 = no counts, 1..k = states
transitions = transition_matrices(groups, window, states)
table = transition_table(groups, window, states, subject_data=prs.set_index('Subject_ID'))
```

Notes:
- **Rows without follow-up** in a time window must be removed, not set to zero: zero means "followed up, no events". Transitions are counted only between consecutive windows in which the subject has a row.
- **Rare types** of events (few counts) can form a state of their own, because a single event changes the composition of a row a lot: merge them (e.g. into "Other") or check that the states are similar without them.
- **Effect of age**: with the time window as dummy covariates, the states describe each row relative to the subjects of the same age; without them, some states may occur only at some ages. Report the choice.
- The **stability** is computed on the rows that are clustered; the reference state is defined by a rule.
- **Regression of the transitions** (e.g. multinomial logistic regression with statsmodels, not installed with *NeuReval*): each subject contributes several transitions, so use standard errors clustered by subject.
- **Hold-out validation**: pass the training labels (```tr_lab```) to ```evaluate_confounds```, so that the hold-out subjects get the same state numbers as the training subjects.
- **Trajectories**: transform counts first (e.g. ```np.log1p```). Use ```degree=1``` if only increases and decreases are expected: a quadratic term without real signal adds noise. Give ```time_points``` if the time windows are not equally spaced. Check that the clusters are not explained by the level (e.g. agreement with groups of overall level).
- **Clusters of sequences**: with ```encoding='onehot'``` subjects are similar if they are in the same states at the same times; with ```encoding='duration'``` only the time spent in each state matters. Subjects with more missing windows than ```max_missing``` are excluded; the missing windows of the others are filled from the nearest observed windows of the same subject. The clusters may reflect mainly the time spent in the most frequent state (e.g. no events): check the profiles.

Script **04_longitudinal_states** shows the whole analysis, from the counts file to the regression of the transitions, the validation in new subjects and the clusters of sequences.


## 4. Example <a name="Example"></a>
An example of how to perform *NeuReval* can be found in the folder **NeuReval/scripts**. These codes show the application of *NeuReval* using Gaussian Mixture Model as clustering algorithm, Support Vector Machine as classifier, and UMAP as dimensionality reduction algorithm:

- **01_grid_search**: code to perform grid-search cross-validation for clustering/classifier/preprocessing parameters tuning
- **02_run_findbestclustcv**: code to perform *NeuReval* with the optimized clustering/classifier/preprocessing algorithms. This script also provides codes to compute different kind of internal measures and external validation in an hold-out dataset
- **03_visualization**: code to create a plot for clusters' representation
- **04_longitudinal_states**: states and transitions in longitudinal count data in long format (see [Longitudinal states and transitions](#States))
- **05_longitudinal_trajectories**: clusters of the shape of longitudinal trajectories (see [Longitudinal states and transitions](#States))

## 5. Notes <a name="Notes"></a>
*NeuReval* was originally developed in Python 3.8.10 on Ubuntu 20.04; the current version requires Python 3.10 or later. In case of any issues in running *NeuReval* on other operating systems (i.e., Windows), you can send an email at the following address: fcolombo.italia@gmail.com

## 6. References <a name="References"></a>
Aitchison, J. (1986). _The Statistical Analysis of Compositional Data_. Chapman and Hall.

Beer, J. C., Tustison, N. J., Cook, P. A., Davatzikos, C., Sheline, Y. I., Shinohara, R. T., & Linn, K. A. (2020). Longitudinal ComBat: A method for harmonizing longitudinal multi-scanner imaging data. _NeuroImage_, 220, 117129.

Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G., & Barceló-Vidal, C. (2003). Isometric logratio transformations for compositional data analysis. _Mathematical Geology_, 35(3), 279-300.

Fortin, J. P., Cullen, N., Sheline, Y. I., Taylor, W. D., Aselcioglu, I., Cook, P. A., ... & Shinohara, R. T. (2018). Harmonization of cortical thickness measurements across scanners and sites. _NeuroImage_, 167, 104-120.

Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. _Biostatistics_, 8(1), 118-127.

Landi, I., Mandelli, V., & Lombardo, M. V. (2021). reval: A Python package to determine best clustering solutions with stability-based relative clustering validation. _Patterns_, 2(4), 100228.