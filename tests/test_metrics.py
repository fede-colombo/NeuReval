"""
Tests for the agreement metrics (neureval.metrics).

Run from the repository root:  pytest tests/test_metrics.py -v
"""
import logging
import os
import sys
import warnings

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, zero_one_loss
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from neureval.metrics import agreement_metrics, cv_agreement_metrics, SUMMARY_METRICS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_cases import make_synthetic_data  # noqa: E402

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# agreement_metrics
# ---------------------------------------------------------------------------
def test_perfect_agreement():
    labels = [0, 0, 1, 1, 2, 2, 2]
    res, cm = agreement_metrics(labels, labels)
    for m in ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'ARI', 'NMI', 'AMI']:
        assert res[m] == pytest.approx(1.0)
    assert np.array_equal(cm.to_numpy(), np.diag([2, 2, 3]))


def test_known_example():
    ref = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 0])
    res, cm = agreement_metrics(ref, pred)
    assert res['accuracy'] == pytest.approx(4 / 6)
    assert res['balanced_accuracy'] == pytest.approx((3 / 4 + 1 / 2) / 2)
    assert res['majority_class_baseline'] == pytest.approx(4 / 6)
    assert res['cohen_kappa'] == pytest.approx(cohen_kappa_score(ref, pred))
    assert (res['n_ref_cluster_0'], res['n_ref_cluster_1']) == (4, 2)
    assert (res['n_pred_cluster_0'], res['n_pred_cluster_1']) == (4, 2)
    # rows = reference (clustering), columns = predicted (classifier)
    assert cm.loc['clustering_0', 'classifier_1'] == 1
    assert cm.loc['clustering_1', 'classifier_0'] == 1


@pytest.mark.parametrize('seed', range(5))
def test_consistency_random_labels(seed):
    """Checks of the original validation-metrics script, on random labelings."""
    rng = np.random.default_rng(seed)
    ref, pred = rng.integers(0, 3, 50), rng.integers(0, 3, 50)
    res, cm = agreement_metrics(ref, pred)
    assert res['accuracy'] == pytest.approx(1 - zero_one_loss(ref, pred))
    assert res['cohen_kappa'] <= res['accuracy'] + 1e-9
    assert cm.to_numpy().sum() == 50
    assert np.trace(cm.to_numpy()) / 50 == pytest.approx(res['accuracy'])


def test_label_only_in_one_labeling():
    """A cluster present only in the predictions still gets a row and a column."""
    res, cm = agreement_metrics([0, 0, 1, 1], [0, 2, 1, 1])
    assert cm.shape == (3, 3)
    assert res['n_ref_cluster_2'] == 0 and res['n_pred_cluster_2'] == 1


def test_invalid_input_raises():
    with pytest.raises(ValueError, match='same length'):
        agreement_metrics([0, 1, 1], [0, 1])
    with pytest.raises(ValueError, match='empty'):
        agreement_metrics([], [])


# ---------------------------------------------------------------------------
# With NeuReval
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def fitted():
    data, mods, covs = make_synthetic_data(cluster_std=10.0)
    fb = FindBestClustCVConfounds(GaussianMixture(random_state=42), SVC(random_state=42),
                                  preprocessing=None, nrand=5, nclust_range=[2, 3, 4])
    fb.best_nclust_confounds(data, mods, covs, iter_cv=2)
    return fb, data, mods, covs


def test_external_validation_accuracy_matches_neureval(fitted):
    fb, data, mods, covs = fitted
    tr = np.zeros(len(data), dtype=bool)
    tr[:80] = True
    out = fb.evaluate_confounds(data, mods, covs, tr, ~tr, nclust=3)
    res, _ = agreement_metrics(out.test_cllab, out.test_classlab)
    assert res['accuracy'] == pytest.approx(out.test_acc)


def test_cv_agreement_metrics(fitted):
    fb = fitted[0]
    per_fold, summary = cv_agreement_metrics(fb.cv_results_)
    # 2 folds x 2 repetitions for each of the 3 numbers of clusters
    assert len(per_fold) == 12
    assert list(summary['ncl']) == [2, 3, 4]
    assert list(summary['n_folds']) == [4, 4, 4]
    for m in SUMMARY_METRICS:
        assert summary[f'{m}_mean'].between(-1, 1).all()
    # summary = mean of the per-fold values
    acc_k3 = per_fold.loc[per_fold.ncl == 3, 'accuracy'].mean()
    assert summary.loc[summary.ncl == 3, 'accuracy_mean'].item() == pytest.approx(acc_k3)
    # normalized stability is the one computed by NeuReval
    assert summary.loc[summary.ncl == 3, 'ms_val_mean'].item() == pytest.approx(
        fb.cv_results_.loc[fb.cv_results_.ncl == 3, 'ms_val'].mean())


def test_cv_results_without_classifier_labels_raise(fitted):
    old_style = fitted[0].cv_results_.drop(columns='val_classlab')
    with pytest.raises(ValueError, match='older than 1.2.0'):
        cv_agreement_metrics(old_style)
