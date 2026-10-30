"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience,
         IRCCS San Raffaele Scientific Institute, Milan, Italy

Agreement metrics between two labelings of the same subjects.

In NeuReval the two labelings are, for each validation set (a cross-validation
fold or an external/hold-out set):

- the clustering labels: the validation set is clustered independently and its
  labels are matched to the training clusters (reference labels);
- the classifier labels: the training cluster predicted for each validation subject
  by the classifier trained on the training clusters.

The accuracy between the two is the quantity used by NeuReval (1 - misclassification
error). The other metrics describe the same agreement from different points of view
and, unlike the accuracy, correct for chance and/or for clusters of different sizes.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, adjusted_rand_score,
                             normalized_mutual_info_score, adjusted_mutual_info_score)


def agreement_metrics(ref_labels, pred_labels):
    """
    Agreement between reference (clustering) labels and predicted (classifier) labels.

    :param ref_labels: reference labels, e.g. the clustering labels of the validation set
        matched to the training clusters (`test_cllab` of `evaluate_confounds`).
    :type ref_labels: array-like, (n_samples,)
    :param pred_labels: predicted labels, e.g. the classifier predictions on the validation
        set (`test_classlab` of `evaluate_confounds`).
    :type pred_labels: array-like, (n_samples,)
    :return: dictionary of metrics, and confusion matrix as a DataFrame (rows: reference
        labels, columns: predicted labels). Metrics:

        - accuracy: proportion of subjects with the same label (as in NeuReval);
        - balanced_accuracy: mean recall over the reference clusters, not inflated by large clusters;
        - cohen_kappa: agreement corrected for chance (0 = chance, 1 = perfect);
        - ARI: adjusted Rand index (0 = chance, 1 = perfect), does not need matched labels;
        - NMI: normalized mutual information (0 = independent, 1 = perfect);
        - AMI: adjusted mutual information, NMI corrected for chance;
        - majority_class_baseline: accuracy obtained by assigning every subject to the
          largest reference cluster, a reference value for the accuracy;
        - n_ref_cluster_<k>, n_pred_cluster_<k>: number of subjects in each cluster.
    :rtype: dict, pandas DataFrame
    """
    ref = np.asarray(ref_labels)
    pred = np.asarray(pred_labels)
    if ref.shape != pred.shape or ref.ndim != 1:
        raise ValueError(f"The two labelings must be 1-D and of the same length, "
                         f"got {ref.shape} and {pred.shape}.")
    if len(ref) == 0:
        raise ValueError("The labelings are empty.")

    labels = np.unique(np.concatenate([ref, pred]))
    _, ref_counts = np.unique(ref, return_counts=True)
    res = {'accuracy': float(accuracy_score(ref, pred)),
           'balanced_accuracy': _balanced_accuracy(ref, pred),
           'cohen_kappa': float(cohen_kappa_score(ref, pred)),
           'ARI': float(adjusted_rand_score(ref, pred)),
           'NMI': float(normalized_mutual_info_score(ref, pred)),
           'AMI': float(adjusted_mutual_info_score(ref, pred)),
           'majority_class_baseline': float(ref_counts.max() / len(ref))}
    for lab in labels:
        res[f'n_ref_cluster_{lab}'] = int(np.sum(ref == lab))
    for lab in labels:
        res[f'n_pred_cluster_{lab}'] = int(np.sum(pred == lab))

    cm = pd.DataFrame(confusion_matrix(ref, pred, labels=labels),
                      index=[f'clustering_{lab}' for lab in labels],
                      columns=[f'classifier_{lab}' for lab in labels])
    return res, cm


def _balanced_accuracy(ref, pred):
    """
    Balanced accuracy, without the warning issued when the classifier predicts a cluster
    that is absent from the reference labels (common in small validation folds, and
    visible anyway in the cluster sizes and in the confusion matrix).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
        return float(balanced_accuracy_score(ref, pred))


SUMMARY_METRICS = ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'ARI', 'NMI', 'AMI',
                   'majority_class_baseline']


def cv_agreement_metrics(cv_results):
    """
    Agreement metrics for each cross-validation fold and number of clusters, and their
    summary (mean and standard deviation across folds) for each number of clusters.

    :param cv_results: cross-validation results of
        :meth:`neureval.best_nclust_cv_confounds.FindBestClustCVConfounds.best_nclust_confounds`
        (the `cv_results_` attribute), which includes the clustering labels (`val_labels`)
        and the classifier predictions (`val_classlab`) of each validation fold.
    :type cv_results: pandas DataFrame
    :return: one row per fold and number of clusters (with the normalized stability `ms_val`
        and the metrics of :func:`agreement_metrics`, except the cluster sizes), and one row
        per number of clusters with mean and standard deviation of each metric.
    :rtype: pandas DataFrame, pandas DataFrame
    """
    if 'val_classlab' not in cv_results.columns:
        raise ValueError("cv_results has no 'val_classlab' column: it was produced by a NeuReval "
                         "version older than 1.2.0. Run best_nclust_confounds again.")
    rows = []
    for row_id, row in cv_results.iterrows():
        res, _ = agreement_metrics(row['val_labels'], row['val_classlab'])
        rows.append({'row': row_id, 'ncl': row['ncl'], 'ms_val': row['ms_val'],
                     **{m: res[m] for m in SUMMARY_METRICS}})
    per_fold = pd.DataFrame(rows)

    cols = ['ms_val'] + SUMMARY_METRICS
    summary = per_fold.groupby('ncl')[cols].agg(['mean', 'std'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary.insert(0, 'n_folds', per_fold.groupby('ncl').size())
    return per_fold, summary.reset_index()
