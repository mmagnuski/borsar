import os.path as op
import random

import pytest
import numpy as np

from scipy import sparse
from scipy.io import loadmat
from skimage.filters import gaussian

import mne
from borsar.utils import has_numba, _get_test_data_dir
from borsar.cluster.utils import create_fake_data_for_cluster_test
from borsar.cluster.label import find_clusters, _cluster_3d_numpy
from borsar.cluster.stats import (_compute_threshold_via_permutations,
                                  _find_stat_fun, _compute_threshold)
from borsar.cluster import construct_adjacency_matrix, cluster_based_regression


data_dir = _get_test_data_dir()
fwd_fname = 'DiamSar-eeg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(op.join(data_dir, fwd_fname))


def test_construct_adjacency():
    '''Test various ways in which adjacency can be constructed.'''
    T, F = True, False
    ch_names = list('ABCD')
    adj_correct = np.array([[F, T, T, F],
                            [T, F, T, F],
                            [T, T, F, T],
                            [F, F, T, F]])

    # construct the neighborhood
    dtypes = [('label', 'O'), ('neighblabel', 'O')]
    arr = np.array([(ch_names[idx], np.array(ch_names)[adj_correct[idx]])
                    for idx in range(adj_correct.shape[0])], dtype=dtypes)

    # test 1, general use case
    adj = construct_adjacency_matrix(arr)
    assert (adj_correct == adj).all()

    # test 2, selected channels
    idx = np.ix_([0, 1, 3], [0, 1, 3])
    adj = construct_adjacency_matrix(arr, ch_names=list('ABD'))
    assert (adj_correct[idx] == adj).all()

    # test 3, as_sparse
    adj = construct_adjacency_matrix(arr, ch_names=list('ABC'), as_sparse=True)
    assert (adj_correct[:3, :3] == adj.toarray()).all()

    # test 4, ch_names must be a list
    with pytest.raises(AssertionError):
        construct_adjacency_matrix(arr, ch_names='abc')

    # test 5, ch_names must contain string
    with pytest.raises(AssertionError):
        construct_adjacency_matrix(arr, ch_names=['A', 23, 'C'])

    # test 6, channel not found in neighbours
    with pytest.raises(ValueError):
        construct_adjacency_matrix(arr, ch_names=['A', 'Bi', 'C'])

    # test 7, multiple channels with the same name found in neighbours
    arr = np.array([(ch_names[idx], np.array(ch_names)[adj_correct[idx]])
                    for idx in range(adj_correct.shape[0])]
                   + [(ch_names[0], np.array(ch_names)[adj_correct[0]])],
                   dtype=dtypes)
    with pytest.raises(ValueError):
        construct_adjacency_matrix(arr, ch_names=['A', 'B', 'C'])


@pytest.mark.skipif(not has_numba(), reason="requires numba")
def test_numba_3d_clustering():
    '''Test clustering/labeling with numba.'''
    from borsar.cluster.label_numba import _cluster_3d_numba
    data = np.load(op.join(data_dir, 'test_clustering.npy'))

    # smooth each 'channel' independently
    for idx in range(data.shape[0]):
        data[idx] = gaussian(data[idx])

    mask_test = data > (data.mean() + data.std())

    # adjacency
    T, F = True, False
    adj = np.array([[F, T, T, F, F],
                    [T, F, T, F, T],
                    [T, T, F, F, F],
                    [F, F, F, F, T],
                    [F, T, F, T, F]])

    clst1 = _cluster_3d_numpy(mask_test, adj)
    clst2 = _cluster_3d_numba(mask_test, adj)

    assert (clst1 == clst2).all()


@pytest.mark.skipif(not has_numba(), reason="requires numba")
def test_2d_clustering():
    '''Test clustering/labeling in 2d with numba and various settings of
    ``min_adj_ch``.'''

    from borsar.cluster.label_numba import _cluster_2d_numba
    T, F = True, False

    data = np.array([[T, T, F, F, F, F, T, F],
                     [F, T, T, F, F, T, T, T],
                     [F, F, F, F, F, F, T, F],
                     [F, F, F, F, F, T, F, F]])

    adjacency = np.zeros((4, 4), dtype='bool')
    adjacency[0, [1, 2]] = T
    adjacency[[1, 2], 0] = T
    adjacency[1, 3] = T
    adjacency[3, 1] = T

    correct_labels = np.array(
        [[1, 1, 0, 0, 0, 0, 2, 0],
         [0, 1, 1, 0, 0, 2, 2, 2],
         [0, 0, 0, 0, 0, 0, 2, 0],
         [0, 0, 0, 0, 0, 2, 0, 0]])

    correct_labels_min_adj_1 = np.array(
        [[0, 1, 0, 0, 0, 0, 2, 0],
         [0, 1, 0, 0, 0, 2, 2, 0],
         [0, 0, 0, 0, 0, 0, 2, 0],
         [0, 0, 0, 0, 0, 2, 0, 0]])

    correct_labels_min_adj_2 = np.array(
        [[0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    correct_answers = [correct_labels, correct_labels_min_adj_1,
                       correct_labels_min_adj_2]

    # test 2d numba clustering for min_adj_ch 0, 1 and 2
    for min_adj, correct in zip([0, 1, 2], correct_answers):
        labels = _cluster_2d_numba(data.copy(), adjacency,
                                   min_adj_ch=min_adj)
        assert (labels == correct).all()


def test_find_clusters():
    threshold = 2.
    T, F = True, False
    adjacency = np.array([[F, T, F], [T, F, T], [F, T, F]])
    data = np.array(
        [
            [
                [2.1, 2., 2.3],
                [1.2, -2.1, -2.3],
                [2.5, -2.05, 1.3]
            ],
            [
                [2.5, 2.4, 2.2],
                [0.3, -2.4, 0.7],
                [2.3, -2.1, 0.7]
            ],
            [
                [2.2, 1.7, 1.4],
                [2.3, 1.4, 1.9],
                [2.1, 1., 0.5]
            ]
        ]
    )
    correct_clst = [data > threshold, data < - threshold]
    backends = ['auto', 'numpy']
    if has_numba():
        backends.append('numba')

    for backend in backends:
        clst, stat = find_clusters(data, threshold, adjacency=adjacency,
                                   backend=backend)
        assert (clst[0] == correct_clst[0]).all()
        assert (clst[1] == correct_clst[1]).all()
        assert data[correct_clst[0]].sum() == stat[0]
        assert data[correct_clst[1]].sum() == stat[1]

    # check using mne backend
    adjacency = np.array([[F, T, F], [T, F, T], [F, T, F]])
    data = np.array([[1., 1.5, 2.1, 2.3, 1.8],
                     [1., 1.4, 1.9, 2.3, 2.2],
                     [0.1, 0.8, 1.5, 1.9, 2.1]])
    correct_clst = data > threshold

    clst, stat = find_clusters(data, threshold, adjacency=adjacency,
                               backend='mne')
    assert (clst[0] == correct_clst).all()


def test_expected_find_clusters_errors():
    from borsar.cluster.label import _check_backend

    data, adj = create_fake_data_for_cluster_test(ndim=2, adjacency=True,
                                                  dim_size=[16, 150])

    # data has to match the shape of adjacency
    msg = ('First data dimension has to correspond to the passed '
           'adjacency matrix.')
    with pytest.raises(ValueError, match=msg):
        find_clusters(data.T, 0.7, adjacency=adj, backend='mne')

    # adjacency has to be 2d
    msg = 'Adjacency has to be a 2d square matrix'
    with pytest.raises(ValueError, match=msg):
        find_clusters(data, 0.7, adjacency=adj[:, 0], backend='mne')

    # mne does not support min_adj_ch
    data_3d, adj2 = create_fake_data_for_cluster_test(ndim=3, adjacency=True)
    msg = "``min_adj_ch`` is not available for the ``'mne'`` backend."
    with pytest.raises(ValueError, match=msg):
        find_clusters(data_3d, 0.7, adjacency=adj2, backend='mne',
                      min_adj_ch=1)

    # min_adj_ch > 0 is allowed only when adjacency is passed
    msg = 'requires that adjacency is not None'
    with pytest.raises(ValueError, match=msg):
        find_clusters(data, 0.7, adjacency=None, min_adj_ch=1)

    # min_adj_ch > 0 is currently available only for 3d data without numba
    if not has_numba():
        # TODO: this error message should be improved in the future,
        #       if backend='auto' it is quite surprising...
        msg = "``min_adj_ch`` is not available for the ``'mne'`` backend."
        with pytest.raises(ValueError, match=msg):
            find_clusters(data, 0.7, adjacency=adj, backend='auto',
                          min_adj_ch=1)

    # 2d with adjacency is available only for numba backend
    expected_msg = 'Currently only "numba" backend can handle '
    with pytest.raises(ValueError, match=expected_msg):
        _check_backend(data, adj, backend='numpy')

    if not has_numba():
        # if you don't have numba you get an error if you ask for numba backend
        expected_msg = 'You need numba package to use the "numba"'
        with pytest.raises(ValueError, match=expected_msg):
            _check_backend(data, adj, backend='numba')
    else:
        # make sure adjacency is present when using numba
        expected_msg = 'Numba backend requires an adjacency matrix.'
        with pytest.raises(ValueError, match=expected_msg):
            _check_backend(data, backend='numba')


def test_3d_clustering_with_min_adj_ch():
    # test data
    data = [
        [
            [0, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0]
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1]
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1]
        ]
    ]

    data = np.array(data).astype('bool')

    # first case - lattice adjacency:
    adjacency = [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0],
                 [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]]
    adjacency = np.array(adjacency).astype('bool')

    # standard clustering
    clusters = _cluster_3d_numpy(data, adjacency)
    assert ((clusters == clusters.max()) == data).all()

    # clustering with min_adj_ch=1 will give two clusters instead of one
    data_copy = data.copy()
    clusters1 = _cluster_3d_numpy(data_copy, adjacency, min_adj_ch=1)
    # we test with 3 because we include 0 (background)
    assert len(np.unique(clusters1)) == 3

    # make sure data were modified in-place
    # (this is not ideal but is ok for find_clusters which passes copies
    #  data > threshold)
    assert not (data == data_copy).all()

    # with higher min_adj_ch only two points remain - all others have < 2
    # adjacent elements in channel dimension
    clusters = _cluster_3d_numpy(data.copy(), adjacency, min_adj_ch=2)
    cluster_ids = np.unique(clusters)[1:]
    for clst_id in cluster_ids:
        assert (clusters == clst_id).sum() == 1

    # numba min_adj_ch > 0
    if has_numba():
        from borsar.cluster.label_numba import _cluster_3d_numba
        clusters1_numba = _cluster_3d_numba(data.copy(), adjacency,
                                           min_adj_ch=1)
        assert len(np.unique(clusters1)) == 3

        masks = [clusters1 == idx for idx in range(1, 3)]
        masks_numba = [clusters1_numba == idx for idx in range(1, 3)]
        assert any([(masks_numba[0] == m).all() for m in masks])
        assert any([(masks_numba[1] == m).all() for m in masks])


@pytest.mark.skipif(not has_numba(), reason="requires numba")
def test_2d_numba_clustering_with_min_adj():
    '''Test 2d numba clustering when min_adj_ch is specified.'''

    data = np.array([[1, 0, 0, 1, 1],
                     [1, 0, 0, 1, 1],
                     [1, 1, 1, 1, 1]])
    adjacency = np.array([[False, True, False],
                          [True, False, True],
                          [False, True, False]])

    # find clusters
    clusters, cluster_stats = find_clusters(
        data.copy(), threshold=0.5, adjacency=adjacency, min_adj_ch=1)

    assert len(clusters) == 2
    # first cluster - first column
    assert clusters[0][:, 0].all()
    # ... and nothing else
    assert not clusters[0][:, 1:].any()
    # second cluster - two last columns
    assert clusters[1][:, 3:].all()
    # ... and nothing else
    assert not clusters[1][:, :3].any()

    assert cluster_stats[0] == 3
    assert cluster_stats[1] == 6


def test_1d_clustering_no_adjacency():
    '''Test clustering on 1d simple data without adjacency.'''
    import skimage

    # create 1d data with gaussian blobs centered at given x positions:
    values = np.zeros(100)
    x_pos = [25, 68, 92]
    values[x_pos] = 1.
    values = skimage.filters.gaussian(values, sigma=5.5)

    # check both backends
    backends = ['numpy']
    if has_numba():
        backends.append('numba')

    for backend in backends:
        # use find clusters with appropriate threshold:
        clusters, _ = find_clusters(values, 0.03, backend=backend)
        assert len(clusters) == 3

        # make sure detected clusters are centered around correct x positions:
        for idx in range(3):
            assert np.abs(np.where(clusters[idx])[0].mean() - x_pos[idx]) < 2


def test_2d_numpy_clustering_no_adjacency():
    '''Test clustering on 2d simple data without adjacency.'''
    import skimage

    # create 2d data with gaussian blobs centered at given x, y positions:
    values = np.zeros((50, 50))
    x_pos = [10, 23, 42]
    y_pos = [27, 6, 44]
    values[x_pos, y_pos] = 1.
    values = skimage.filters.gaussian(values, sigma=5.5)

    # use find clusters with appropriate threshold:
    clusters, stats = find_clusters(values, 0.003)
    assert len(clusters) == 3

    # make sure detected clusters are centered around correct x, y positions:
    for idx in range(3):
        cluster_pos = np.array([pos.mean() for pos in np.where(clusters[idx])])
        true_pos = np.array([x_pos[idx], y_pos[idx]])
        dist = np.sqrt(((true_pos - cluster_pos) ** 2).sum())
        assert dist < 2


def test_get_cluster_fun():
    from borsar.cluster.label import _get_cluster_fun, _check_backend

    # TODO: test errors in a separate function
    # check expected errors
    # ---------------------
    data = np.random.random((4, 10)) > 0.75
    adj = np.zeros((4, 4), dtype='bool')
    adj[[0, 0, 1, 1, 2, 3], [1, 3, 0, 2, 1, 0]] = True

    # check correct outputs
    # ---------------------
    if has_numba():
        from borsar.cluster.label_numba import (_cluster_2d_numba,
                                                _cluster_3d_numba)
        backend = _check_backend(data, adj, backend='auto')
        func = _get_cluster_fun(data, adj, backend=backend)
        assert func == _cluster_2d_numba

        data = np.random.random((4, 10, 5)) > 0.75
        backend = _check_backend(data, adj, backend='auto')
        func = _get_cluster_fun(data, adj, backend=backend)
        assert func == _cluster_3d_numba

    if not has_numba():
        from borsar.cluster.label import _cluster_3d_numpy

        data = np.random.random((4, 10, 5)) > 0.75
        backend = _check_backend(data, adj, backend='auto')
        func = _get_cluster_fun(data, adj, backend=backend)
        assert func == _cluster_3d_numpy


def test_cluster_based_regression():
    np.random.seed(23)
    data_dir = _get_test_data_dir()

    # TEST 1
    # ======

    # read data and fieldtrip's stat results
    stat = loadmat(
        op.join(data_dir, 'ft_stat_test_01.mat'), squeeze_me=True)['stat']
    all_data = loadmat(
        op.join(data_dir, 'cluster_regr_test_01.mat'), squeeze_me=True)

    data = all_data['data']
    pred = all_data['pred']

    t_values, clusters, cluster_p, dist = cluster_based_regression(
        data, pred, return_distribution=True, progressbar=False)

    # cluster p values should be very similar
    # ---------------------------------------
    cluster_p_ft = np.concatenate([stat['posclusters'].item()['prob'],
                                  [stat['negclusters'].item()['prob'].item()]]
                                  ).astype('float')

    # for small p-values the differences should be smaller,
    # for large they could reach up to 0.095
    assert (np.abs(cluster_p_ft - cluster_p) < [0.01, 0.095, 0.11]).all()

    # distributions should be very similar
    # ------------------------------------
    dist_ft = {prefix: stat['{}distribution'.format(prefix)].item()
                  for prefix in ['pos', 'neg']}

    vals = np.array([5., 15, 30, 50, 100])
    max_perc_error = np.array([7, 7, 5, 5, 4.5]) / 100.

    for fun, prefix, vls in zip([np.less, np.greater],
                                ['pos', 'neg'], [vals, vals * -1]):
        ft_perc = np.array([fun(dist_ft[prefix], v).mean() for v in vls])
        borsar_perc = np.array([fun(dist[prefix], v).mean() for v in vls])
        assert (np.abs(ft_perc - borsar_perc) < max_perc_error).all()

    # masks should be the same
    # ------------------------
    pos_mat = stat['posclusterslabelmat'].item()
    neg_mat = stat['negclusterslabelmat'].item()
    assert ((pos_mat == 1) == clusters[0]).all()
    assert ((pos_mat == 2) == clusters[1]).all()
    assert ((neg_mat == 1) == clusters[2]).all()

    # t values should be almost the same
    # ----------------------------------
    np.testing.assert_allclose(stat['stat'].item(), t_values, rtol=1e-10)

    # TEST 2
    # ======
    # smoke test for running cluster_based_regression with adjacency
    data = np.random.random((15, 4, 4))
    preds = np.random.random(15)

    T, F = True, False
    adjacency = sparse.coo_matrix([[F, T, T, F], [T, F, T, F], [T, T, F, T],
                                   [F, F, T, F]])

    cluster_based_regression(data, preds, adjacency=adjacency)


def test_cluster_based_regression_3d_simulated():
    np.random.seed(23)
    T, F = True, False

    # ground truth - cluster locations
    data = np.random.normal(size=(10, 3, 5, 5))
    adjacency = np.array([[F, T, T], [T, F, T], [T, T, F]])
    pos_clst = [[0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                [1, 1, 2, 0, 0, 1, 2, 2, 0, 1, 2, 3],
                [1, 2, 2, 0, 1, 1, 2, 3, 0, 0, 3, 3]]
    neg_clst = [[0, 0, 0, 0, 2, 2, 2],
                [3, 4, 4, 4, 3, 4, 4],
                [4, 2, 3, 4, 1, 1, 2]]
    pred = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1])

    # create pos cluster
    weight = 1.9
    pos_idx = tuple([slice(None)] + pos_clst)
    weight_noise = np.random.sample((len(data), len(pos_clst[0]))) * 0.2 - 0.05
    data[pos_idx] += pred[:, np.newaxis] * (weight + weight_noise)

    # create neg cluster
    weight = -1.5
    neg_idx = tuple([slice(None)] + neg_clst)
    weight_noise = np.random.sample((len(data), len(neg_clst[0]))) * 0.3 - 0.2
    data[neg_idx] += pred[:, np.newaxis] * (weight + weight_noise)

    # prepare data and run cluster_based_regression
    reg_data = data.copy()
    stat, clst, pvals = cluster_based_regression(
        reg_data, pred, adjacency=adjacency, stat_threshold=2.,
        progressbar=False)

    # find index of positive and negative clusters
    clst_stat = np.array([stat[c].sum() for c in clst])
    pos_clst_idx = (pvals[clst_stat > 0].argmin()
                    + np.where(clst_stat > 0)[0][0])
    neg_clst_idx = (pvals[clst_stat < 0].argmin()
                    + np.where(clst_stat < 0)[0][0])

    # assert that clusters are similar to locations of original effects
    assert clst[pos_clst_idx][pos_idx[1:]].mean() > 0.75
    assert clst[pos_clst_idx][neg_idx[1:]].mean() < 0.29
    assert clst[neg_clst_idx][neg_idx[1:]].mean() > 0.75


def test_clustering_parameter_combinations():
    '''Test all possible parameter combinations for find_clusters.

    Only parameters listed in ``get_supported_find_clusters_parameters`` are
    tested.'''
    from borsar.cluster.label import get_supported_find_clusters_parameters
    supported = get_supported_find_clusters_parameters()

    for _, row in supported.iterrows():
        ndim = row['n dimensions']
        use_adj = row['channel dimension'] == 'yes'

        data, adj = create_fake_data_for_cluster_test(
            ndim=ndim, adjacency=use_adj)
        min_adj_ch = (0 if not row['min_adj_ch'] == 'yes'
                      else random.choice([1, 2, 3, 4]))

        backend = row['backend']
        should_succeed = row['supported'] == 'yes'

        if backend == 'numba' and not has_numba():
            should_succeed = False

        if should_succeed:
            find_clusters(data, threshold=0.6, adjacency=adj, backend=backend,
                          min_adj_ch=min_adj_ch)
        else:
            with pytest.raises(ValueError):
                find_clusters(data, threshold=0.6, adjacency=adj,
                              backend=backend, min_adj_ch=min_adj_ch)


def test_clustering_backend_selection():
    from borsar.cluster.label import _check_backend
    from borsar.cluster.label import get_supported_find_clusters_parameters

    has_numba_lib = has_numba()
    supported = get_supported_find_clusters_parameters()
    cols = supported.columns.to_list()
    uniq = supported.groupby(cols[:3], as_index=False).count()

    for _, row in uniq.iterrows():
        ndim = row['n dimensions']
        has_adj = row['channel dimension']
        min_adj_ch = row['min_adj_ch']

        msk = (
            (supported['n dimensions'] == ndim)
            & (supported['channel dimension'] == has_adj)
            & (supported['min_adj_ch'] == min_adj_ch)
        )
        sel = supported.loc[msk, :]

        is_supported = sel.supported == 'yes'
        n_supported = is_supported.sum()
        if n_supported == 2:
            should_select = 'numba' if has_numba_lib else 'numpy'
        elif n_supported == 1:
            should_select = sel.loc[is_supported, 'backend'].values[0]
            if should_select == 'numba' and not has_numba_lib:
                should_select = 'mne'
        elif n_supported == 0:
            should_select = 'mne'

        data, adj = create_fake_data_for_cluster_test(
            ndim=ndim, adjacency=has_adj == 'yes')
        min_adj_ch = 2 if min_adj_ch == 'yes' else 0

        try:
            selected = _check_backend(data, adjacency=adj,
                                      min_adj_ch=min_adj_ch)
            assert selected == should_select
        except ValueError:
            assert should_select == 'mne'


def test_custom_filter_function():
    from functools import partial

    # create data
    test_data = np.array(
        [[1, 0, 0, 1, 0, 0],
         [0, 1, 0, 1, 1, 0],
         [0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0],
         [1, 1, 1, 0, 1, 1]]
    )

    corrected_for_min_4_ngb = np.array(
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0],
         [0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype='bool'
    )

    corrected_for_min_3_ngb = np.array(
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0],
         [0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 1, 0],
         [0, 2, 0, 0, 0, 0],
         [0, 2, 0, 0, 0, 0]]
    )

    # custom function to remove pixels with too little neighbours
    # (including diagonal neighbours)
    def filter_fun(clusters, adjacency=None, min_adj=4):
        from scipy.signal import correlate2d

        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        n_ngb = correlate2d(clusters, kernel, mode='same')
        return clusters & (n_ngb >= min_adj)

    clusters, _ = find_clusters(
        test_data, threshold=0.5, filter_fun=filter_fun)
    assert (clusters[0] == corrected_for_min_4_ngb).all()

    this_filter = partial(filter_fun, min_adj=3)
    clusters, _ = find_clusters(
        test_data, threshold=0.5, filter_fun=this_filter)
    assert (clusters[0] == (corrected_for_min_3_ngb == 1)).all()
    assert (clusters[1] == (corrected_for_min_3_ngb == 2)).all()


def test_custom_post_filter_function():
    # create data
    test_data = np.array(
        [[1, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0],
         [1, 1, 1, 0, 1, 1]]
    )

    data_after_removing_diagonal_clusters = np.array(
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0]],
        dtype='bool'
    )

    # custom function to remove clusters on the diagonal
    # (may be useful for far off-diagonal generalization effects in
    #  time generalization decoding)
    def remove_diagonal_clusters(clusters, adjacency=None):
        diag_mask = np.zeros(clusters.shape, dtype='bool')
        np.fill_diagonal(diag_mask, True)

        cluster_ids = np.unique(clusters)
        if cluster_ids[0] == 0:
            cluster_ids = cluster_ids[1:]

        for clst_id in cluster_ids:
            this_mask = clusters == clst_id
            is_diagonal = (diag_mask & this_mask).any()
            if is_diagonal:
                clusters[this_mask] = 0

        return clusters

    clusters, _ = find_clusters(
        test_data, threshold=0.5, filter_fun_post=remove_diagonal_clusters)
    assert (clusters[0] == data_after_removing_diagonal_clusters).all()


def test_compute_threshold_via_permutations():
    """Make sure that threshold computed through permutations is correct.

    Check that the threshold computed through permutations/randomization
    on data that fulfills assumptions of analytical tests is sufficiently
    close to the analytical threshold.
    """
    n_groups = 2

    for paired in [False, True]:
        if paired:
            n_obs = [101, 101]
            data = [np.random.randn(n_obs[0])]
            data.append(data[0] + np.random.randn(n_obs[0]))
        else:
            n_obs = [102, 100]
            data = [np.random.randn(n) for n in n_obs]

        analytical_threshold = _compute_threshold(
            data=data, threshold=None, p_threshold=0.05, paired=paired,
            one_sample=False)

        stat_fun = _find_stat_fun(
            n_groups, paired=paired, tail='both')

        permutation_threshold = (
            _compute_threshold_via_permutations(
                data, paired=paired, tail='both', stat_fun=stat_fun,
                n_permutations=2_000, progress=False
            )
        )

        avg_perm = np.abs(permutation_threshold).mean()
        error = analytical_threshold - avg_perm

        print('paired:', paired)
        print('analytical_threshold:', analytical_threshold)
        print('permutation threshold:', permutation_threshold)
        print('average permutation threshold:', avg_perm)
        print('difference:', error)

        assert np.abs(error) < 0.15
