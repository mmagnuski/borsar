import os.path as op
import pytest
import numpy as np
from scipy import sparse
from scipy.io import loadmat

import borsar
from borsar.utils import download_test_data, _get_test_data_dir
from borsar.cluster import (construct_adjacency_matrix, Clusters, read_cluster,
                            _get_mass_range, cluster_based_regression,
                            _index_from_dim)


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
                    for idx in range(adj_correct.shape[0])] +
                   [(ch_names[0], np.array(ch_names)[adj_correct[0]])],
                   dtype=dtypes)
    with pytest.raises(ValueError):
        construct_adjacency_matrix(arr, ch_names=['A', 'B', 'C'])


def test_cluster_based_regression():
    data_dir = op.join(op.split(borsar.__file__)[0], 'data')

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
    # for large they could reach up to 0.09
    assert (np.abs(cluster_p_ft - cluster_p) < [0.01, 0.09, 0.09]).all()

    # distributions should be very similar
    # ------------------------------------
    distrib_ft = {prefix: stat['{}distribution'.format(prefix)].item()
                  for prefix in ['pos', 'neg']}

    vals = np.array([5., 15, 30, 50, 100])
    max_perc_error = np.array([7, 6, 5, 4, 3.5]) / 100.

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
    data = np.random.random((15, 4, 4))
    preds = np.random.random(15)

    T, F = True, False
    adjacency = sparse.coo_matrix([[F, T, T, F], [T, F, T, F], [T, T, F, T],
                                   [F, F, T, F]])

    tvals, clst, clst_p = cluster_based_regression(data, preds,
                                                   adjacency=adjacency)


def test_get_mass_range():
    contrib = np.array([0.15, 0.04, 0.09, 0.16, 0.21, 0.1, 0.05,
                        0.01, 0.08, 0.11])
    assert _get_mass_range(contrib, 0.1) == slice(4, 5)
    assert _get_mass_range(contrib, 0.3) == slice(3, 5)
    assert _get_mass_range(contrib, 0.37) == slice(3, 5)
    assert _get_mass_range(contrib, 0.38) == slice(3, 6)
    assert _get_mass_range(contrib, 0.48) == slice(2, 6)
    assert _get_mass_range(contrib, 0.57) == slice(2, 7)


def test_index_from_dim():
    dimnames = ['chan', 'freq', 'time']
    dimcoords = [None, np.arange(8., 12.1, step=0.5),
                 np.arange(-0.2, 0.51, step=0.1)]
    assert _index_from_dim(dimnames[1:2], dimcoords[1:2]) == (slice(None),)
    assert _index_from_dim(dimnames[1:], dimcoords[1:]) == (slice(None),) * 2
    assert (_index_from_dim(dimnames, dimcoords, freq=[10, 11.5]) ==
            (slice(None), slice(4, 8), slice(None)))
    assert (_index_from_dim(dimnames, dimcoords, freq=[9.5, 10], time=[0, 0.3])
            == (slice(None), slice(3, 5), slice(2, 6)))


def test_clusters():
    import mne
    import matplotlib.pyplot as plt

    data_dir = _get_test_data_dir()
    download_test_data()
    fname = 'DiamSar-eeg-oct-6-fwd.fif'
    clst_file = op.join(data_dir, 'alpha_range_clusters.hdf5')
    fwd = mne.read_forward_solution(op.join(data_dir, fname))

    clst = read_cluster(clst_file, src=fwd['src'], subjects_dir=data_dir)
    assert (len(clst) == len(clst.pvals) == len(clst.clusters)
            == len(clst.cluster_polarity))
    assert len(clst) == 14

    # test selection
    clst2 = clst.copy().select(p_threshold=0.2)
    assert len(clst2) == 3

    # test contribution
    clst_0_freq_contrib = clst2.get_contribution(cluster_idx=0, along='freq')
    len(clst_0_freq_contrib) == len(clst2.dimcoords[1])

    # tests for plot_contribution
    ax = clst2.plot_contribution('freq')
    assert isinstance(ax, plt.Axes)
    children = ax.get_children()
    isline = [isinstance(chld, plt.Line2D) for chld in children]
    assert sum(isline) == len(clst2)
    which_line = np.where(isline)[0]
    line_data = children[which_line[0]].get_data()[1]
    assert (line_data / line_data.sum() == clst_0_freq_contrib).all()

    # get cluster limits
    idx = clst2.get_cluster_limits(0, retain_mass=0.75)
    clst_0_freq_contrib[idx[1]].sum() > 0.75

    idx = clst2.get_index(freq=[8, 10])
    assert idx[1] == slice(2, 7)

    # test iteration
    pvls = list()
    for c in clst2:
        pvls.append(c.pvals[0])
    assert (clst2.pvals == pvls).all()

    # test selection with percentage_in and n_points_in
    clst3 = clst2.copy().select(percentage_in=0.7, freq=[7, 9])
    assert len(clst3) == 1

    clst3 = clst2.copy().select(n_points_in=2900)
    assert len(clst3) == 2

    clst3 = clst2.copy().select(n_points_in=340, freq=[10.5, 12.5])
    assert len(clst3) == 1
