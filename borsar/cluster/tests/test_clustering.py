import os.path as op
import pytest
import numpy as np

from scipy import sparse
from scipy.io import loadmat
from skimage.filters import gaussian

import mne
from borsar.utils import has_numba, _get_test_data_dir
from borsar.cluster import (construct_adjacency_matrix, find_clusters,
                            cluster_based_regression, cluster_3d)


data_dir = _get_test_data_dir()
fwd_fname = 'DiamSar-eeg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(op.join(data_dir, fwd_fname))


def test_contstruct_adjacency():
    T, F = True, False
    ch_names = list('ABCD')
    adj_correct = np.array([[F, T, T, F],
                            [T, F, T, F],
                            [T, T, F, T],
                            [F, F, T, F]])

    # contruct neighbourhood
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


def test_numba_clustering():
    if has_numba():
        from borsar.cluster.label_numba import cluster_3d_numba
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

        clst1 = cluster_3d(mask_test, adj)
        clst2 = cluster_3d_numba(mask_test, adj)

        assert (clst1 == clst2).all()


def test_find_clusters():
    threshold = 2.
    T, F = True, False
    adjacency = np.array([[F, T, F], [T, F, T], [F, T, F]])
    data = np.array([[[2.1, 2., 2.3], [1.2, -2.1, -2.3], [2.5, -2.05, 1.3]],
                     [[2.5, 2.4, 2.2], [0.3, -2.4, 0.7], [2.3, -2.1, 0.7]],
                     [[2.2, 1.7, 1.4], [2.3, 1.4, 1.9], [2.1, 1., 0.5]]])
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
    data = np.array([[1., 1.5, 2.1, 2.3, 1.8], [1., 1.4, 1.9, 2.3, 2.2],
                     [0.1, 0.8, 1.5, 1.9, 2.1]])
    correct_clst = data.T > threshold

    with pytest.raises(ValueError, match='of the correct size'):
        clst, stat = find_clusters(data, threshold, adjacency=adjacency,
                                   backend='mne')

    clst, stat = find_clusters(data.T, threshold, adjacency=adjacency,
                               backend='mne')
    assert (clst[0] == correct_clst).all()


def test_3d_clustering_with_min_adj_ch():
    # test data
    data = [[[0, 1, 1, 0, 0],
             [1, 1, 1, 1, 0],
             [1, 1, 1, 1, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 1, 1, 0]],
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1]],
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1]]]

    data = np.array(data).astype('bool')

    # first case - lattice adjacency:
    adjacency = [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0],
                 [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]]
    adjacency = np.array(adjacency).astype('bool')

    # standard clustering
    clusters = cluster_3d(data, adjacency)
    assert ((clusters == clusters.max()) == data).all()

    # clustering with min_adj_ch=1 will give two clusters instead of one
    clusters = cluster_3d(data, adjacency, min_adj_ch=1)
    assert len(np.unique(clusters)) == 3  # 3 because we include 0 (background)

    # with higher min_adj_ch only two points remain - all others have < 2
    # adjacent elements in channel dimension
    clusters = cluster_3d(data, adjacency, min_adj_ch=2)
    cluster_ids = np.unique(clusters)[1:]
    for clst_id in cluster_ids:
        assert (clusters == clst_id).sum() == 1


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

    t_values, clusters, cluster_p, distrib = cluster_based_regression(
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
    distrib_ft = {prefix: stat['{}distribution'.format(prefix)].item()
                  for prefix in ['pos', 'neg']}

    vals = np.array([5., 15, 30, 50, 100])
    max_perc_error = np.array([7, 7, 5, 5, 4.5]) / 100.

    for fun, prefix, vls in zip([np.less, np.greater],
                                ['pos', 'neg'], [vals, vals * -1]):
        ft = np.array([fun(distrib_ft[prefix], v).mean() for v in vls])
        brsr = np.array([fun(distrib[prefix], v).mean() for v in vls])
        assert (np.abs(ft - brsr) < max_perc_error).all()

    # masks should be the same
    # ------------------------
    posmat = stat['posclusterslabelmat'].item()
    negmat = stat['negclusterslabelmat'].item()
    assert ((posmat == 1) == clusters[0]).all()
    assert ((posmat == 2) == clusters[1]).all()
    assert ((negmat == 1) == clusters[2]).all()

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

    tvals, clst, clst_p = cluster_based_regression(data, preds,
                                                   adjacency=adjacency)


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
    wght = 1.9
    pos_idx = tuple([slice(None)] + pos_clst)
    wght_noise = np.random.sample((len(data), len(pos_clst[0]))) * 0.2 - 0.05
    data[pos_idx] += pred[:, np.newaxis] * (wght + wght_noise)

    # create neg cluster
    wght = -1.5
    neg_idx = tuple([slice(None)] + neg_clst)
    wght_noise = np.random.sample((len(data), len(neg_clst[0]))) * 0.3 - 0.2
    data[neg_idx] += pred[:, np.newaxis] * (wght + wght_noise)

    # prepare data and run cluster_based_regression
    reg_data = data.copy().swapaxes(1, -1)
    stat, clst, pvals = cluster_based_regression(
        reg_data, pred, adjacency=adjacency, stat_threshold=2.,
        progressbar=False)

    # swapaxes back to orig size
    stat = stat.swapaxes(0, -1)
    clst = [c.swapaxes(0, -1) for c in clst]

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
