"""
Tests for the tools for longitudinal data (neureval.longitudinal).

Run from the repository root:  pytest tests/test_longitudinal.py -v
"""
import numpy as np
import pandas as pd
import pytest

from sklearn.cluster import KMeans
from sklearn.svm import SVC

from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from neureval.longitudinal import (composition_features, select_rows, add_reference_state,
                                   state_sequences, transition_matrices, transition_table)


# ---------------------------------------------------------------------------
# Step 1: composition_features
# ---------------------------------------------------------------------------
def test_composition_does_not_depend_on_total():
    """Two rows with the same proportions of each type have the same composition."""
    counts = pd.DataFrame({'resp': [2, 20], 'gastro': [4, 40], 'ear': [6, 60]})
    feat = composition_features(counts, pseudocount=1e-9)
    ilr = feat.filter(like='ilr').to_numpy()
    np.testing.assert_allclose(ilr[0], ilr[1], atol=1e-6)
    assert feat['log_total'].iloc[1] > feat['log_total'].iloc[0]


def test_ilr_preserves_distances_between_compositions():
    """ILR coordinates have the same distances as the CLR values, without collinearity."""
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.poisson(3, (50, 4)) + 1, columns=list('abcd'))
    ilr = composition_features(counts, include_total=False).to_numpy()
    log_p = np.log(counts.to_numpy() + 0.5)
    clr = log_p - log_p.mean(axis=1, keepdims=True)
    d_ilr = np.linalg.norm(ilr[:, None] - ilr[None], axis=2)
    d_clr = np.linalg.norm(clr[:, None] - clr[None], axis=2)
    np.testing.assert_allclose(d_ilr, d_clr, atol=1e-10)
    assert ilr.shape[1] == 3
    assert np.linalg.matrix_rank(ilr - ilr.mean(0)) == 3        # no collinear columns


def test_rows_without_counts_have_no_composition():
    counts = pd.DataFrame({'a': [0, 1, 3], 'b': [0, 0, 2]}, index=['x', 'y', 'z'])
    feat = composition_features(counts)
    assert feat.loc['x'].isna().all()
    assert feat.loc[['y', 'z']].notna().all().all()
    feat2 = composition_features(counts, min_total=2)
    assert feat2.loc[['x', 'y']].isna().all().all() and feat2.loc['z'].notna().all()
    assert list(feat.index) == ['x', 'y', 'z']


def test_composition_input_checks():
    with pytest.raises(ValueError, match="at least 2 types"):
        composition_features(pd.DataFrame({'a': [1, 2]}))
    with pytest.raises(ValueError, match="non-negative"):
        composition_features(pd.DataFrame({'a': [1, -2], 'b': [1, 1]}))
    with pytest.raises(ValueError, match="missing values"):
        composition_features(pd.DataFrame({'a': [1, np.nan], 'b': [1, 1]}))
    with pytest.raises(ValueError, match="pseudocount"):
        composition_features(pd.DataFrame({'a': [1, 2], 'b': [1, 1]}), pseudocount=0)


# ---------------------------------------------------------------------------
# Step 2: select_rows and add_reference_state
# ---------------------------------------------------------------------------
def test_select_rows_keeps_order_and_resets_index():
    df = pd.DataFrame({'a': [10, 11, 12, 13]}, index=[5, 6, 7, 8])
    mods = {'m1': df, 'm2': df * 2}
    groups = np.array(['s1', 's1', 's2', 's2'])
    mask = np.array([True, False, True, True])
    d, m, g, none = select_rows(mask, df, mods, groups, None)
    assert d['a'].tolist() == [10, 12, 13] and d.index.tolist() == [0, 1, 2]
    assert m['m2']['a'].tolist() == [20, 24, 26]
    assert g.tolist() == ['s1', 's2', 's2'] and none is None
    assert select_rows(mask, groups).tolist() == ['s1', 's2', 's2']   # a single object
    with pytest.raises(ValueError, match="rows but mask has"):
        select_rows(mask, groups[:3])
    with pytest.raises(ValueError, match="boolean"):
        select_rows([1, 0, 1, 1], groups)


def test_add_reference_state():
    mask = np.array([True, False, True, True, False])
    states = add_reference_state([0, 2, -1], mask)
    assert states.tolist() == [1, 0, 3, -1, 0]
    assert add_reference_state(np.array([1.0, 0.0, 1.0]), mask).tolist() == [2, 0, 1, 2, 0]
    with pytest.raises(ValueError, match="3 rows were clustered"):
        add_reference_state([0, 1], mask)
    with pytest.raises(ValueError, match="integer"):
        add_reference_state([0.5, 1, 1], mask)


def test_states_with_reference_state_end_to_end():
    """
    Counts of 3 types with 2 compositional states (mostly type a / mostly type c) and rows
    without counts: the rows with counts are clustered with NeuReval (subjects kept together),
    the rows without counts get the reference state 0.
    """
    rng = np.random.default_rng(0)
    n_subj, n_win = 80, 4
    subj = np.repeat(np.arange(n_subj), n_win)
    true = rng.integers(0, 3, len(subj))              # 0 = no counts, 1 = mostly a, 2 = mostly c
    lam = np.array([[0, 0, 0], [6, 1, 0.5], [0.5, 1, 6]])[true]
    counts = pd.DataFrame(rng.poisson(lam), columns=['a', 'b', 'c'])
    true[counts.sum(axis=1).to_numpy() == 0] = 0      # rows that happen to have no counts
    feat = composition_features(counts, include_total=False)
    informative = feat.notna().all(axis=1).to_numpy()
    cov = pd.DataFrame({'window': np.tile(np.arange(n_win), n_subj).astype(float)})

    f_inf, cov_inf, g_inf = select_rows(informative, feat, cov, subj)
    fb = FindBestClustCVConfounds(KMeans(n_init=10, random_state=0), SVC(random_state=42), None, nrand=10,
                                  nclust_range=[2, 3, 4])
    _, k, lab = fb.best_nclust_confounds(f_inf, {'comp': f_inf}, {'comp': cov_inf}, iter_cv=2, groups=g_inf)
    states = add_reference_state(lab, informative)

    assert k == 2
    assert len(states) == len(counts)
    assert (states[~informative] == 0).all() and set(states[informative]) == {1, 2}
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(true, states) > 0.8


# ---------------------------------------------------------------------------
# Step 3: state_sequences and transition_matrices
# ---------------------------------------------------------------------------
# Three subjects, windows 1-3. B has no row for window 2; rows are not in time order.
LONG = pd.DataFrame({'subject': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'],
                     'window':  [1, 2, 3, 3, 1, 1, 2, 3],
                     'state':   [0, 1, 1, 2, 1, 1, 1, 0]})


def test_state_sequences():
    seq = state_sequences(LONG['subject'], LONG['window'], LONG['state'])
    assert list(seq.columns) == [1, 2, 3] and list(seq.index) == ['A', 'B', 'C']
    assert seq.loc['A'].tolist() == [0, 1, 1]
    assert seq.loc['B', 1] == 1 and pd.isna(seq.loc['B', 2]) and seq.loc['B', 3] == 2
    assert str(seq.dtypes.iloc[0]) == 'Int64'


def test_transition_counts_and_probabilities():
    counts = transition_matrices(LONG['subject'], LONG['window'], LONG['state'], normalize=False)
    assert set(counts) == {(1, 2), (2, 3), 'all'}
    # 1 -> 2: A 0->1, C 1->1 (B has no window 2)
    assert counts[(1, 2)].loc[0, 1] == 1 and counts[(1, 2)].loc[1, 1] == 1
    assert counts[(1, 2)].to_numpy().sum() == 2
    # 2 -> 3: A 1->1, C 1->0 (B has no window 2, so no transition 1 -> 3)
    assert counts[(2, 3)].loc[1, 1] == 1 and counts[(2, 3)].loc[1, 0] == 1
    assert counts[(2, 3)].to_numpy().sum() == 2
    pd.testing.assert_frame_equal(counts['all'], counts[(1, 2)] + counts[(2, 3)])
    # every matrix has all states, in the same order
    for table in counts.values():
        assert list(table.index) == [0, 1, 2] and list(table.columns) == [0, 1, 2]

    prob = transition_matrices(LONG['subject'], LONG['window'], LONG['state'])
    assert prob[(2, 3)].loc[1].tolist() == [0.5, 0.5, 0.0]
    np.testing.assert_allclose(prob['all'].loc[[0, 1]].sum(axis=1), 1.0)
    assert prob['all'].loc[2].isna().all()            # nobody in state 2 before the last window


def test_time_order_for_labels():
    """Labels such as '6-18m' are not sorted correctly: give the order explicitly."""
    labels = {1: '6-18m', 2: '18m-3y', 3: '3-6y'}
    time = LONG['window'].map(labels)
    order = ['6-18m', '18m-3y', '3-6y']
    seq = state_sequences(LONG['subject'], time, LONG['state'], time_order=order)
    assert list(seq.columns) == order and seq.loc['A'].tolist() == [0, 1, 1]
    counts = transition_matrices(LONG['subject'], time, LONG['state'], time_order=order, normalize=False)
    assert set(counts) == {('6-18m', '18m-3y'), ('18m-3y', '3-6y'), 'all'}
    with pytest.raises(ValueError, match="not in time_order"):
        state_sequences(LONG['subject'], time, LONG['state'], time_order=order[:2])


def test_sequences_input_checks():
    dup = pd.concat([LONG, LONG.iloc[[0]]])
    with pytest.raises(ValueError, match="at most one row per time window"):
        transition_matrices(dup['subject'], dup['window'], dup['state'])
    with pytest.raises(ValueError, match="one value per row"):
        state_sequences(LONG['subject'], LONG['window'][:-1], LONG['state'])
    with pytest.raises(ValueError, match="missing values"):
        state_sequences(LONG['subject'], LONG['window'], LONG['state'].replace(2, np.nan))


# ---------------------------------------------------------------------------
# Step 4: transition_table
# ---------------------------------------------------------------------------
def test_transition_table():
    prs = pd.DataFrame({'PRS': [0.5, -1.0, 2.0], 'SES': [1, 2, 3]}, index=['A', 'B', 'C'])
    table = transition_table(LONG['subject'], LONG['window'], LONG['state'], subject_data=prs)
    assert list(table.columns) == ['subject', 'time_from', 'time_to', 'state_from', 'state_to', 'PRS', 'SES']
    # A: 1->2, 2->3; B: no consecutive windows; C: 1->2, 2->3
    assert table[['subject', 'time_from', 'time_to', 'state_from', 'state_to']].values.tolist() == [
        ['A', 1, 2, 0, 1], ['A', 2, 3, 1, 1], ['C', 1, 2, 1, 1], ['C', 2, 3, 1, 0]]
    assert table.loc[table.subject == 'C', 'PRS'].tolist() == [2.0, 2.0]


def test_transition_table_matches_transition_matrices():
    rng = np.random.default_rng(1)
    subj = np.repeat(np.arange(40), 5)
    win = np.tile(np.arange(1, 6), 40)
    keep = rng.random(len(subj)) > 0.2
    st = rng.integers(0, 4, len(subj))
    subj, win, st = subj[keep], win[keep], st[keep]
    table = transition_table(subj, win, st)
    counts = transition_matrices(subj, win, st, normalize=False)
    del counts['all']
    for (t0, t1), m in counts.items():
        sub = table[(table.time_from == t0) & (table.time_to == t1)]
        assert len(sub) == m.to_numpy().sum()
        assert (pd.crosstab(sub.state_from, sub.state_to)
                .reindex(index=m.index, columns=m.columns, fill_value=0).to_numpy() == m.to_numpy()).all()


def test_transition_table_input_checks():
    with pytest.raises(ValueError, match="not in subject_data"):
        transition_table(LONG['subject'], LONG['window'], LONG['state'],
                         subject_data=pd.DataFrame({'PRS': [0.1]}, index=['A']))
    with pytest.raises(ValueError, match="one row per subject"):
        transition_table(LONG['subject'], LONG['window'], LONG['state'],
                         subject_data=pd.DataFrame({'PRS': [0.1, 0.2]}, index=['A', 'A']))
    with pytest.raises(ValueError, match="reserved names"):
        transition_table(LONG['subject'], LONG['window'], LONG['state'],
                         subject_data=pd.DataFrame({'state_to': [1, 1, 1]}, index=['A', 'B', 'C']))