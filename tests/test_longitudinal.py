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
                                   state_sequences, transition_matrices, transition_table,
                                   sequence_features, sequence_profiles, to_wide,
                                   level_shape_features)


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


# ---------------------------------------------------------------------------
# Step 7: sequence_features and sequence_profiles
# ---------------------------------------------------------------------------
SEQ = state_sequences(LONG['subject'], LONG['window'], LONG['state'])   # B misses window 2


def test_sequence_features_onehot():
    feat = sequence_features(SEQ, max_missing=1)
    assert list(feat.columns) == ['1_0', '1_1', '1_2', '2_0', '2_1', '2_2', '3_0', '3_1', '3_2']
    assert feat.loc['A'].tolist() == [1, 0, 0, 0, 1, 0, 0, 1, 0]
    # B: window 2 missing, between state 1 (window 1) and state 2 (window 3)
    assert feat.loc['B', ['2_0', '2_1', '2_2']].tolist() == [0, 0.5, 0.5]
    for w in ('1', '2', '3'):
        np.testing.assert_allclose(feat[[c for c in feat.columns if c.startswith(w + '_')]].sum(axis=1), 1)
    # subjects with too many missing windows are excluded
    assert sequence_features(SEQ).index.tolist() == ['A', 'C']


def test_sequence_features_missing_at_the_edges():
    seq = state_sequences(['A', 'A', 'B', 'B', 'B'], [2, 3, 1, 2, 3], [1, 2, 0, 0, 0])
    feat = sequence_features(seq, max_missing=1)
    assert feat.loc['A', ['1_0', '1_1', '1_2']].tolist() == [0, 1, 0]      # nearest observed window


def test_sequence_features_duration():
    feat = sequence_features(SEQ, encoding='duration', max_missing=1)
    assert list(feat.columns) == ['prop_0', 'prop_1', 'prop_2', 'changes']
    np.testing.assert_allclose(feat.loc['A'], [1 / 3, 2 / 3, 0, 0.5])
    np.testing.assert_allclose(feat.loc['B'], [0, 0.5, 0.5, 1.0])         # 1 -> 2 between observed windows
    with pytest.raises(ValueError, match="encoding"):
        sequence_features(SEQ, encoding='hamming')


def test_sequence_profiles():
    labels = pd.Series([0, 1, 0], index=['A', 'B', 'C'])
    prof = sequence_profiles(SEQ, labels)
    assert prof.loc[(0, 1)].tolist() == [0.5, 0.5, 0.0, 2]
    assert prof.loc[(0, 2), 1] == 1.0
    assert (1, 2) not in prof.index                                      # B not observed in window 2
    assert prof.loc[(1, 3), 'n_subjects'] == 1
    # labels as an array in the order of the sequences
    pd.testing.assert_frame_equal(sequence_profiles(SEQ, labels.to_numpy()), prof)
    with pytest.raises(ValueError, match="not in sequences"):
        sequence_profiles(SEQ, pd.Series([0], index=['Z']))


def test_clusters_of_sequences_end_to_end():
    """
    Three typical courses over 5 windows (each window follows the course with probability 0.75):
    the clusters of the one-hot sequences found with NeuReval follow the courses.
    """
    rng = np.random.default_rng(1)
    courses = {0: [2, 2, 0, 0, 0], 1: [1, 1, 1, 1, 1], 2: [0, 0, 0, 3, 3]}
    course = rng.integers(0, 3, 300)
    rows = [[i, w + 1, courses[course[i]][w] if rng.random() < 0.75 else rng.integers(0, 4)]
            for i in range(300) for w in range(5) if rng.random() >= 0.1]
    df = pd.DataFrame(rows, columns=['id', 'window', 'state'])
    seq = state_sequences(df['id'], df['window'], df['state'])
    feat = sequence_features(seq, max_missing=1)
    X = feat.reset_index(drop=True)
    cov = pd.DataFrame({'sex': rng.integers(0, 2, len(X)).astype(float)})
    from sklearn.linear_model import LogisticRegression
    fb = FindBestClustCVConfounds(KMeans(n_init=10, random_state=0), LogisticRegression(max_iter=1000), None,
                                  nrand=10, nclust_range=[2, 3, 4, 5])
    _, k, lab = fb.best_nclust_confounds(X, {'seq': X}, {'seq': cov}, iter_cv=2)
    from sklearn.metrics import adjusted_rand_score
    assert k == 3
    assert adjusted_rand_score(course[feat.index.to_numpy()], lab) > 0.8
    prof = sequence_profiles(seq, pd.Series(lab, index=feat.index))
    assert set(prof.index.get_level_values('cluster')) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Step 8: to_wide and level_shape_features
# ---------------------------------------------------------------------------
TRAJ = pd.DataFrame([[1, 2, 3, 4, 5],       # A: increasing
                     [11, 12, 13, 14, 15],  # B: same shape as A, higher level
                     [5, 4, 3, 2, 1],       # C: decreasing
                     [1, 3, 5, 3, 1],       # D: peak in the middle
                     [5, 3, 1, 3, 5]],      # E: dip in the middle
                    index=list('ABCDE'), dtype=float)


def test_to_wide():
    wide = to_wide(LONG['subject'], LONG['window'], LONG['state'] * 10)
    assert list(wide.columns) == [1, 2, 3] and wide.loc['A'].tolist() == [0, 10, 10]
    assert np.isnan(wide.loc['B', 2]) and wide.loc['B', 3] == 20


def test_level_and_shape_are_separated():
    feat = level_shape_features(TRAJ)
    assert list(feat.columns) == ['level', 'linear', 'quadratic']
    np.testing.assert_allclose(feat['level'], TRAJ.mean(axis=1))
    # same shape, different level
    np.testing.assert_allclose(feat.loc['A', ['linear', 'quadratic']], feat.loc['B', ['linear', 'quadratic']], atol=1e-9)
    assert feat.loc['A', 'linear'] > 0 and feat.loc['C', 'linear'] < 0
    assert abs(feat.loc['A', 'quadratic']) < 1e-9
    assert feat.loc['D', 'quadratic'] < 0 < feat.loc['E', 'quadratic']        # peak < 0, dip > 0
    assert abs(feat.loc['D', 'linear']) < 1e-9


def test_level_shape_degree_and_time_points():
    feat = level_shape_features(TRAJ, degree=4)
    assert list(feat.columns) == ['level', 'linear', 'quadratic', 'cubic', 'degree_4']
    # with all degrees the decomposition keeps all the information: same distances between subjects
    # (level rescaled from a mean to a sum of squares, as the other coefficients)
    feat_scaled = feat.copy()
    feat_scaled['level'] *= np.sqrt(TRAJ.shape[1])
    d_feat = np.linalg.norm(feat_scaled.to_numpy()[:, None] - feat_scaled.to_numpy()[None], axis=2)
    d_traj = np.linalg.norm(TRAJ.to_numpy()[:, None] - TRAJ.to_numpy()[None], axis=2)
    np.testing.assert_allclose(d_feat, d_traj, atol=1e-9)
    # unequally spaced windows: a linear trajectory in time has no quadratic component
    t = np.array([0.5, 2, 4.5, 8, 11.5])
    lin = pd.DataFrame([2 + 0.3 * t])
    assert abs(level_shape_features(lin, time_points=t).loc[0, 'quadratic']) < 1e-9
    assert abs(level_shape_features(lin).loc[0, 'quadratic']) > 0.01  # equally spaced: not linear
    with pytest.raises(ValueError, match="degree must be"):
        level_shape_features(TRAJ, degree=5)
    with pytest.raises(ValueError, match="one value per window"):
        level_shape_features(TRAJ, time_points=[1, 2, 3])


def test_level_shape_missing_windows():
    traj = TRAJ.copy()
    traj.loc['A', 2] = np.nan                        # a perfect line: same result without a window
    traj.loc['D', [1, 2, 3]] = np.nan                # 2 observed windows: too few for degree 2
    feat = level_shape_features(traj)
    np.testing.assert_allclose(feat.loc['A'], level_shape_features(TRAJ).loc['A'], atol=1e-9)
    assert feat.loc['D'].isna().all() and feat.drop('D').notna().all().all()
    assert level_shape_features(traj, degree=1, min_observed=2).loc['D'].notna().all()


def test_shape_clusters_follow_timing_not_level():
    """
    Three shapes over time (early, middle, late peak) and a large subject-specific level:
    clustering the shape finds the 3 shapes, independently of the level.
    """
    rng = np.random.default_rng(0)
    shapes = np.array([[3, 2, 1, .5, .5], [.5, 1.5, 3, 1.5, .5], [.5, .5, 1, 2, 3]])
    shape = rng.integers(0, 3, 300)
    mult = np.exp(rng.normal(0, 0.7, 300))
    traj = pd.DataFrame(np.log1p(rng.poisson(4 * shapes[shape] * mult[:, None])))
    feat = level_shape_features(traj)
    X = feat[['linear', 'quadratic']]
    cov = pd.DataFrame({'sex': rng.integers(0, 2, 300).astype(float)})
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import adjusted_rand_score
    fb = FindBestClustCVConfounds(KMeans(n_init=10, random_state=0), LogisticRegression(max_iter=1000), None,
                                  nrand=10, nclust_range=[2, 3, 4, 5])
    _, k, lab = fb.best_nclust_confounds(X, {'shape': X}, {'shape': cov}, iter_cv=2)
    assert k == 3
    assert adjusted_rand_score(shape, lab) > 0.8
    assert abs(adjusted_rand_score(pd.qcut(feat['level'], 3, labels=False), lab)) < 0.1