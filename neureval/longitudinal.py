"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience,
         IRCCS San Raffaele Scientific Institute, Milan, Italy

Tools for longitudinal data in long format (one row per subject and time window), to
define states with NeuReval and study how subjects move between states over time:

1. composition_features: features describing the composition of counts (e.g. types of
   infection) in each row, separately from their total.
2. select_rows and add_reference_state: cluster only the informative rows (e.g. rows with
   at least one count) and give the other rows a state defined a priori (e.g. "no counts").
3. state_sequences and transition_matrices: sequence of states of each subject and
   transitions between consecutive time windows.
4. transition_table: one row per transition, with subject-level variables, for regression
   models of the transitions (e.g. multinomial logistic regression).
"""

import numpy as np
import pandas as pd
from scipy.linalg import helmert


def composition_features(counts, pseudocount=0.5, min_total=1, include_total=True):
    """
    Describe each row by the composition of its counts (which types are more or less
    frequent) and, separately, by its total, so that the clustering is not dominated by
    the total (rows with many counts of every type).

    The composition is computed with the isometric log-ratio transform (ILR): proportions
    of each type (after adding `pseudocount` to every count, so that zero counts are
    allowed), log-ratios centred on their mean (CLR), and projection onto an orthonormal
    basis. The ILR coordinates are not collinear (the CLR values of a row always sum to
    zero), so they can be used with any clustering algorithm, including Gaussian mixtures.
    To describe the clusters, use the proportions of each type, not the ILR coordinates.
    Types with very few counts can form a cluster of their own (a single count changes the
    composition of a row a lot): merge rare types or check the results without them.

    Each row is transformed on its own: no information is shared between rows, so the
    features can be computed before the cross-validation without leakage.

    :param counts: counts of each type (columns) in each row, non-negative.
    :type counts: DataFrame, (n_rows, n_types)
    :param pseudocount: value added to every count before computing the proportions.
    :type pseudocount: float
    :param min_total: rows whose total is smaller than this value have no composition
        (all features are NaN): they must be handled separately (e.g. as a "no counts"
        state) and excluded from the clustering. Default 1.
    :type min_total: float
    :param include_total: if True, add a column 'log_total' = log(1 + total).
    :type include_total: bool
    :return: ILR coordinates ('ilr_1' ... 'ilr_{n_types - 1}') and, if requested,
        'log_total', with the same index as `counts`.
    :rtype: DataFrame
    """
    counts = pd.DataFrame(counts)
    if counts.shape[1] < 2:
        raise ValueError("counts needs at least 2 types (columns) to compute a composition.")
    values = counts.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("counts contains missing values.")
    if (values < 0).any():
        raise ValueError("counts must be non-negative.")
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive, so that zero counts are allowed.")

    total = values.sum(axis=1)
    log_p = np.log(values + pseudocount)
    clr = log_p - log_p.mean(axis=1, keepdims=True)
    ilr = clr @ helmert(values.shape[1]).T
    ilr[total < min_total] = np.nan

    features = pd.DataFrame(ilr, index=counts.index,
                            columns=[f'ilr_{j}' for j in range(1, values.shape[1])])
    if include_total:
        features['log_total'] = np.log1p(total)
        features.loc[total < min_total, 'log_total'] = np.nan
    return features


def select_rows(mask, *objects):
    """
    Select the same rows from several objects (e.g. data, modalities, covariates, groups),
    for example the rows to cluster. The row order is kept and the index is reset, as
    required by NeuReval.

    :param mask: rows to select, one boolean value per row.
    :type mask: array-like of bool, (n_rows,)
    :param objects: DataFrame, Series, array or dictionary of these (e.g. modalities),
        each with one row per row of `mask`. None is returned as None.
    :return: the selected rows of each object, in the same order as `objects`.
    :rtype: tuple
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 1:
        raise ValueError("mask must be a one-dimensional boolean array (True = row selected).")

    def _select(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: _select(v) for k, v in obj.items()}
        if len(obj) != len(mask):
            raise ValueError(f"An object has {len(obj)} rows but mask has {len(mask)} values.")
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.loc[mask].reset_index(drop=True)
        return np.asarray(obj)[mask]

    selected = tuple(_select(obj) for obj in objects)
    return selected[0] if len(selected) == 1 else selected


def add_reference_state(labels, mask):
    """
    Combine the states found by the clustering on a subset of rows (e.g. rows with at least
    one count) with a reference state defined a priori for the other rows (e.g. "no counts").

    The reference state is labelled 0 and the clusters 1, 2, ..., k (clustering label + 1).
    Rows labelled -1 by the clustering (noise, e.g. with HDBSCAN) keep the label -1.

    :param labels: clustering labels of the rows where `mask` is True, in the same order
        (e.g. tr_lab returned by best_nclust_confounds, or test_classlab returned by
        evaluate_confounds).
    :type labels: array-like, (mask.sum(),)
    :param mask: rows that were clustered, one boolean value per row of the whole dataset.
    :type mask: array-like of bool, (n_rows,)
    :return: state of every row: 0 = reference state, 1..k = clusters, -1 = noise.
    :rtype: ndarray of int, (n_rows,)
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 1:
        raise ValueError("mask must be a one-dimensional boolean array (True = row clustered).")
    labels = np.asarray(labels)
    if len(labels) != mask.sum():
        raise ValueError(f"labels has {len(labels)} values but {mask.sum()} rows were clustered "
                         f"(True in mask).")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.mod(labels, 1) == 0):
            raise ValueError("labels must be integer clustering labels.")
        labels = labels.astype(int)
    states = np.zeros(len(mask), dtype=int)
    states[mask] = np.where(labels >= 0, labels + 1, -1)
    return states


def _check_long(groups, time, states, time_order=None):
    """
    Check the subject, time window and state of each row, and return them as a DataFrame
    with the time windows as an ordered categorical variable.
    """
    groups, time, states = np.asarray(groups), np.asarray(time), np.asarray(states)
    if not (groups.ndim == time.ndim == states.ndim == 1) or not (len(groups) == len(time) == len(states)):
        raise ValueError("groups, time and states must be one-dimensional, with one value per row.")
    for name, values in (('groups', groups), ('time', time), ('states', states)):
        if pd.isna(values).any():
            raise ValueError(f"{name} contains missing values.")
    if time_order is None:
        time_order = sorted(pd.unique(time))
    else:
        time_order = list(time_order)
        unknown = set(pd.unique(time)) - set(time_order)
        if unknown:
            raise ValueError(f"time contains values not in time_order: {sorted(unknown, key=str)}.")
        if len(set(time_order)) != len(time_order):
            raise ValueError("time_order contains repeated values.")
    df = pd.DataFrame({'subject': groups,
                       'time': pd.Categorical(time, categories=time_order, ordered=True),
                       'state': states})
    dup = df.duplicated(['subject', 'time'])
    if dup.any():
        first = df.loc[dup].iloc[0]
        raise ValueError(f"{dup.sum()} row(s) repeat a subject and time window already present "
                         f"(e.g. subject {first['subject']!r}, time {first['time']!r}): each subject "
                         f"must have at most one row per time window.")
    return df


def state_sequences(groups, time, states, time_order=None):
    """
    Sequence of states of each subject: one row per subject, one column per time window.

    :param groups: subject of each row (e.g. subject ID).
    :type groups: array-like, (n_rows,)
    :param time: time window of each row. Windows are ordered by sorting their values
        (use numbers, e.g. 1, 2, 3, or give `time_order`: labels such as '6-18m', '18m-3y'
        are not sorted correctly).
    :type time: array-like, (n_rows,)
    :param states: state of each row (e.g. returned by :func:`add_reference_state`).
    :type states: array-like, (n_rows,)
    :param time_order: time windows from the first to the last, default None (sorted values).
    :type time_order: list
    :return: states, with subjects as index and time windows as columns (in time order);
        <NA> where a subject has no row for a time window.
    :rtype: DataFrame
    """
    df = _check_long(groups, time, states, time_order)
    seq = df.pivot(index='subject', columns='time', values='state')
    seq = seq.reindex(columns=df['time'].cat.categories)
    if pd.api.types.is_integer_dtype(df['state']):
        seq = seq.astype('Int64')
    seq.columns = list(seq.columns)
    seq.index.name = None
    return seq


def transition_matrices(groups, time, states, time_order=None, normalize=True):
    """
    Transitions between states in consecutive time windows.

    For each pair of consecutive time windows (t, t+1), counts the subjects in each state at
    t (rows) that are in each state at t+1 (columns). Only subjects with a row in both windows
    are counted: a subject with a missing window contributes no transition into or out of it
    (e.g. windows 1 and 3 without 2 give no transition). The matrix 'all' sums the transitions
    of all pairs of consecutive windows.

    :param groups: subject of each row (e.g. subject ID).
    :type groups: array-like, (n_rows,)
    :param time: time window of each row (see :func:`state_sequences` for the order).
    :type time: array-like, (n_rows,)
    :param states: state of each row.
    :type states: array-like, (n_rows,)
    :param time_order: time windows from the first to the last, default None (sorted values).
    :type time_order: list
    :param normalize: if True, each row is divided by its total: probability of each state at
        t+1 given the state at t (NaN if no subject is in that state at t). If False, number
        of subjects.
    :type normalize: bool
    :return: one matrix for each pair of consecutive windows, with keys (t, t+1), and the
        matrix 'all'. Rows: state at t ('from'); columns: state at t+1 ('to'). All matrices
        have every state, in the same order.
    :rtype: dict of DataFrame
    """
    df = _check_long(groups, time, states, time_order)
    order = list(df['time'].cat.categories)
    all_states = sorted(pd.unique(df['state']), key=lambda v: (str(type(v)), v))
    seq = df.pivot(index='subject', columns='time', values='state').reindex(columns=order)

    def _matrix(frm, to):
        table = pd.crosstab(pd.Categorical(frm, categories=all_states),
                            pd.Categorical(to, categories=all_states), dropna=False)
        table.index.name, table.columns.name = 'from', 'to'
        return table

    counts = {}
    for t0, t1 in zip(order[:-1], order[1:]):
        both = seq[[t0, t1]].dropna()
        counts[(t0, t1)] = _matrix(both[t0].to_numpy(), both[t1].to_numpy())
    total = sum(counts.values()) if counts else _matrix([], [])
    counts['all'] = total
    if not normalize:
        return counts
    return {key: table.div(table.sum(axis=1).replace(0, np.nan), axis=0) for key, table in counts.items()}


def transition_table(groups, time, states, time_order=None, subject_data=None):
    """
    One row per transition between consecutive time windows of a subject, for regression
    models of the transitions, e.g. multinomial logistic regression of the state at t+1 on
    the state at t and on subject-level variables (state_to ~ state_from + PRS + SES).

    Only transitions between consecutive windows in which the subject has a row are included,
    as in :func:`transition_matrices` (a subject with windows 1 and 3 without 2 has none).
    Each subject usually contributes several transitions: use standard errors that account
    for this (e.g. clustered by subject).

    :param groups: subject of each row (e.g. subject ID).
    :type groups: array-like, (n_rows,)
    :param time: time window of each row (see :func:`state_sequences` for the order).
    :type time: array-like, (n_rows,)
    :param states: state of each row.
    :type states: array-like, (n_rows,)
    :param time_order: time windows from the first to the last, default None (sorted values).
    :type time_order: list
    :param subject_data: variables of each subject (e.g. PRS, SES, sex), with the subject ID
        as index (e.g. ``table.set_index('Subject_ID')``), one row per subject. Added to each
        transition of the subject. Default None.
    :type subject_data: DataFrame
    :return: columns 'subject', 'time_from', 'time_to', 'state_from', 'state_to' and the
        columns of `subject_data`; one row per transition, ordered by subject and time.
    :rtype: DataFrame
    """
    df = _check_long(groups, time, states, time_order)
    order = list(df['time'].cat.categories)
    position = {t: i for i, t in enumerate(order)}
    # Each row as the start of a transition (state at t) and as its end (state at t+1);
    # rows are matched by subject and by the position of the window in the time order
    start = pd.DataFrame({'subject': df['subject'], 'pos': df['time'].astype(object).map(position),
                          'time_from': df['time'].astype(object), 'state_from': df['state']})
    end = pd.DataFrame({'subject': df['subject'], 'pos': df['time'].astype(object).map(position) - 1,
                        'time_to': df['time'].astype(object), 'state_to': df['state']})
    table = (start.merge(end, on=['subject', 'pos'], how='inner')
                  .sort_values(['subject', 'pos'], kind='stable')
                  [['subject', 'time_from', 'time_to', 'state_from', 'state_to']]
                  .reset_index(drop=True))

    if subject_data is not None:
        subject_data = pd.DataFrame(subject_data)
        if subject_data.index.duplicated().any():
            raise ValueError("subject_data must have one row per subject (repeated subject IDs in the index).")
        clash = set(subject_data.columns) & set(table.columns)
        if clash:
            raise ValueError(f"subject_data has columns with reserved names: {sorted(clash)}.")
        missing = set(pd.unique(table['subject'])) - set(subject_data.index)
        if missing:
            raise ValueError(f"{len(missing)} subject(s) are not in subject_data "
                             f"(e.g. {sorted(missing, key=str)[:5]}): its index must be the subject ID.")
        table = table.join(subject_data, on='subject')
    return table