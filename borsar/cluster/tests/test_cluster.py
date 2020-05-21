import os
import os.path as op
import warnings

import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.io import loadmat
from skimage.filters import gaussian
import mne

import borsar
from borsar.stats import format_pvalue
from borsar.utils import (download_test_data, _get_test_data_dir, has_numba,
                          find_index)
from borsar.cluster import (Clusters, cluster_3d, find_clusters,
                            construct_adjacency_matrix, read_cluster,
                            cluster_based_regression)
from borsar.cluster.checks import (_clusters_safety_checks, _check_description,
                                   _clusters_chan_vert_checks,
                                   _check_dimnames_kwargs)
from borsar.cluster.utils import (_check_stc, _label_from_cluster, _get_clim,
                                  _prepare_cluster_description, _handle_dims,
                                  _aggregate_cluster, _get_units,
                                  _get_dimcoords, _get_mass_range,
                                  _format_cluster_pvalues, _index_from_dim)
from borsar.cluster.viz import _label_axis

# setup
download_test_data()
data_dir = _get_test_data_dir()
fwd_fname = 'DiamSar-eeg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(op.join(data_dir, fwd_fname))


def _create_random_clusters(dims='ch_tm', n_clusters=1):
    n_channels, n_times = 15, 35

    try:
        # mne > 0.18
        mntg = mne.channels.make_standard_montage('standard_1020')
    except AttributeError:
        # mne 0.18 and below
        mntg = mne.channels.read_montage('standard_1020')

    ch_names = mntg.ch_names[slice(0, 89, 6)]
    times = np.linspace(-0.2, 0.5, num=n_times)
    sfreq = 1 / np.diff(times[:2])[0]
    try:
        info = mne.create_info(ch_names, sfreq, ch_types=['eeg'] * n_channels,
                               montage=mntg, verbose=False)
    except TypeError:
        info = mne.create_info(ch_names, sfreq, ch_types=['eeg'] * n_channels,
                               verbose=False)
        info.set_montage(mntg)

    if dims == 'ch_tm':
        dimnames = ['chan', 'time']
        dimsizes = (n_channels, n_times)
        dimcoords = [ch_names, times]
    elif dims == 'ch_fr_tm':
        n_freqs = 15
        freqs = np.arange(5, 20)
        dimnames = ['chan', 'freq', 'time']
        dimsizes = (n_channels, n_freqs, n_times)
        dimcoords = [ch_names, freqs, times]

    data = np.random.random(dimsizes)
    clusters = [np.random.random(dimsizes) >= 0.5 for ix in range(n_clusters)]
    clst = Clusters(data, clusters, [0.01], dimnames=dimnames,
                    dimcoords=dimcoords, info=info)
    return clst


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
    assert clst[neg_clst_idx][pos_idx[1:]].mean() < 0.29


def test_get_mass_range():
    contrib = np.array([0.15, 0.04, 0.09, 0.16, 0.21, 0.1, 0.05,
                        0.01, 0.08, 0.11])
    assert _get_mass_range(contrib, 0.1) == slice(4, 5)
    assert _get_mass_range(contrib, 0.3) == slice(3, 5)
    assert _get_mass_range(contrib, 0.37) == slice(3, 5)
    assert _get_mass_range(contrib, 0.38) == slice(3, 6)
    assert _get_mass_range(contrib, 0.48) == slice(2, 6)
    assert _get_mass_range(contrib, 0.57) == slice(2, 7)

    assert (_get_mass_range(contrib, 0.3, adjacent=False)
            == np.array([3, 4])).all()
    assert (_get_mass_range(contrib, 0.38, adjacent=False)
            == np.array([0, 3, 4])).all()
    assert (_get_mass_range(contrib, 0.53, adjacent=False)
            == np.array([0, 3, 4, 9])).all()

    # with break
    contrib = np.array([0.15, 0.15, 0., 0.15, 0.2, 0.1, 0.])
    slc = _get_mass_range(contrib, 0.5)
    assert slc == slice(3, 6)
    assert contrib[slc].sum() < 0.5


def test_index_from_dim():
    dimnames = ['chan', 'freq', 'time']
    dimcoords = [None, np.arange(8., 12.1, step=0.5),
                 np.arange(-0.2, 0.51, step=0.1)]
    assert _index_from_dim(dimnames[1:2], dimcoords[1:2]) == (slice(None),)
    assert _index_from_dim(dimnames[1:], dimcoords[1:]) == (slice(None),) * 2
    assert (_index_from_dim(dimnames, dimcoords, freq=(10, 11.5))
            == (slice(None), slice(4, 8), slice(None)))
    assert (_index_from_dim(dimnames, dimcoords, freq=(9.5, 10), time=(0, 0.3))
            == (slice(None), slice(3, 5), slice(2, 6)))
    print(_index_from_dim(dimnames, dimcoords, freq=[9.5, 10], time=(0, 0.3)))
    idx = _index_from_dim(dimnames, dimcoords, freq=[9.5, 10], time=(0, 0.3))
    assert (idx[0] == slice(None) and (idx[1] == [3, 4]).all()
            and idx[2] == slice(2, 6))
    with pytest.raises(TypeError):
        _index_from_dim(dimnames, dimcoords, freq=[9.5], time=(0, 0.2, 0.3))


def test_cluster_limits():
    T, F = True, False
    use_contrib = np.array([0.25, 0.1, 0.38, 0.07, 0.15, 0.05, 0.1])
    use_clusters = [np.array([T, F, T, T, T, F, T])]

    clusters = np.zeros((7, 100), dtype='bool')
    for idx, (cntrb, cl) in enumerate(zip(use_contrib, use_clusters[0])):
        if cl:
            clusters[idx, :int(cntrb * 100)] = True

    pvals = [0.02]
    stat = np.ones((7, 100))
    info = mne.create_info(list('ABCDEFG'), sfreq=250.)
    # FIX: dimcoords should not be necessary, but we get error without
    dimcoords = [list('ABCDEFG'), np.linspace(3., 25., num=100)]
    clst = Clusters(stat, [clusters], pvals, dimnames=['chan', 'freq'],
                    dimcoords=dimcoords, info=info)

    lmts = clst.get_cluster_limits(0, retain_mass=0.66, dims=['chan', 'freq'])
    assert (lmts[0] == np.array([0, 2])).all()

    lmts = clst.get_cluster_limits(0, retain_mass=0.68, dims=['chan', 'freq'])
    assert (lmts[0] == np.array([0, 2, 4])).all()

    lmts = clst.get_cluster_limits(0, retain_mass=0.83, dims=['chan', 'freq'])
    assert (lmts[0] == np.array([0, 2, 4, 6])).all()


def test_clusters():
    import mne
    import matplotlib.pyplot as plt

    # the second call should not do anything if all is downloaded
    download_test_data()

    # read source-space cluster results
    clst_file = op.join(data_dir, 'alpha_range_clusters.hdf5')
    clst = read_cluster(clst_file, src=fwd['src'], subjects_dir=data_dir)

    assert (len(clst) == len(clst.pvals) == len(clst.clusters)
            == len(clst.cluster_polarity))
    assert len(clst) == 14

    txt = repr(clst)
    correct_repr = ('<borsar.Clusters  |  14 clusters in '
                    'vertices x frequency space>')
    assert txt == correct_repr

    # selection
    # ---------

    # p value
    clst2 = clst.copy().select(p_threshold=0.2)
    assert len(clst2) == 3

    # selection with percentage_in
    clst3 = clst2.copy().select(percentage_in=0.7, freq=(7, 9))
    assert len(clst3) == 1

    # using n_points_in without dimension ...
    clst3 = clst2.copy().select(n_points_in=2900)
    assert len(clst3) == 2

    # ... works the same as using n_points
    clst3 = clst2.copy().select(n_points=2900)
    assert len(clst3) == 2

    # n_points_in with dimension range
    clst3 = clst2.copy().select(n_points_in=340, freq=(10.5, 12.5))
    assert len(clst3) == 1

    # using `selection` arg
    clst3 = clst2.copy().select(selection=[0, 1])
    assert len(clst3) == 2
    clst3 = clst2.copy().select(selection=np.array([0, 1]))
    assert len(clst3) == 2

    # selection that results in no clusters
    clst_no = clst.copy().select(p_threshold=0.05)
    assert len(clst_no) == 0

    # selection that starts with no clusters
    clst_no.select(n_points_in=10)
    assert len(clst_no) == 0

    # selection that selects all
    clst3 = clst2.copy().select(p_threshold=0.5, n_points_in=100)
    assert len(clst3) == 3

    # write - read round-trip
    # ----------------------
    fname = op.join(data_dir, 'temp_clst.hdf5')
    clst2.save(fname)
    clst_read = read_cluster(fname, src=fwd['src'], subjects_dir=data_dir)
    assert len(clst_read) == len(clst2)
    assert (clst_read.pvals == clst2.pvals).all()
    assert (clst_read.clusters == clst2.clusters).all()
    assert (clst_read.stat == clst2.stat).all()
    # delete the file
    os.remove(op.join(data_dir, 'temp_clst.hdf5'))

    with pytest.raises(TypeError):
        clst2.save(fname, description=list('abc'))

    # test contribution
    # -----------------
    clst_0_freq_contrib = clst2.get_contribution(cluster_idx=0, along='freq')
    len(clst_0_freq_contrib) == len(clst2.dimcoords[1])

    # along as int
    clst_0_freq_contrib2 = clst2.get_contribution(cluster_idx=0, along=1)
    assert (clst_0_freq_contrib == clst_0_freq_contrib2).all()

    # get_contribution when no cluster_idx is passed
    all_contrib = clst2.get_contribution(along='freq')
    assert all_contrib.shape[0] == len(clst2)
    assert all_contrib.shape[1] == clst2.stat.shape[1]

    # get_contribution with idx argument
    all_contrib = clst2.get_contribution(along='freq', idx=(None, [0, 1, 2]))
    assert all_contrib.shape[0] == len(clst2)
    assert all_contrib.shape[1] == 3

    # non string
    match = r'has to be string \(dimension name\) or int \(dimension index\)'
    with pytest.raises(TypeError, match=match):
        clst2.get_contribution(cluster_idx=0, along=all_contrib)

    # negative (could later work)
    with pytest.raises(ValueError, match='must be greater or equal to 0'):
        clst2.get_contribution(cluster_idx=0, along=-1)

    # int greater there is dimensions - 1
    with pytest.raises(ValueError, match='must be greater or equal to 0'):
        clst2.get_contribution(cluster_idx=0, along=2)

    # tests for plot_contribution
    ax = clst2.plot_contribution('freq')
    assert isinstance(ax, plt.Axes)
    children = ax.get_children()
    isline = [isinstance(chld, plt.Line2D) for chld in children]
    assert sum(isline) == len(clst2)
    which_line = np.where(isline)[0]
    line_data = children[which_line[0]].get_data()[1]
    assert (line_data / line_data.sum() == clst_0_freq_contrib).all()

    clst2.dimcoords, dcoords = None, clst2.dimcoords
    ax = clst2.plot_contribution('freq')
    xlab = ax.get_xlabel()
    assert xlab == 'frequency bins'
    clst2.dimcoords = dcoords

    match = 'Clusters has to have `dimnames` attribute'
    with pytest.raises(TypeError, match=match):
        dnames = clst2.dimnames
        clst2.dimnames = None
        ax = clst2.plot_contribution('freq')
    clst2.dimnames = dnames

    match = 'does not seem to have the dimension you requested'
    with pytest.raises(ValueError, match=match):
        clst2.plot_contribution('abc')

    with pytest.raises(ValueError, match='No clusters present'):
        clst_no.plot_contribution('freq')

    # get index and limits
    # --------------------
    idx = clst2.get_cluster_limits(0, retain_mass=0.75)
    clst_0_freq_contrib[idx[1]].sum() > 0.75

    idx = clst2.get_index(freq=(8, 10))
    assert idx[1] == slice(2, 7)

    idx = clst2.get_index(freq=[8, 10])
    assert (idx[1] == [2, 6]).all()

    idx = clst2.get_index(cluster_idx=1, freq=0.6)
    contrib = clst2.get_contribution(1, 'freq')
    assert contrib[idx[1]].sum() >= 0.6

    assert clst2.get_index() == (slice(None), slice(None))
    with pytest.raises(ValueError, match='Could not find requested dimension'):
        clst2.get_index(abc=[1, 2])
    with pytest.raises(TypeError, match='Clusters has to have dimnames'):
        dnames = clst2.dimnames
        clst2.dimnames = None
        clst2.get_index(freq=(10, 11))
    clst2.dimnames = dnames

    with pytest.raises(TypeError, match='Clusters has to have dimcoords'):
        dcoords = clst2.dimcoords
        clst2.dimcoords = None
        clst2.get_index(freq=(8.5, 10))
    clst2.dimcoords = dcoords
    match = (r'either specific points \(list or array of values\), ranges '
             r'\(tuple of two values\) or cluster extent to retain \(float\)')
    with pytest.raises(TypeError, match=match):
        clst2.get_index(freq='abc')

    # test iteration
    pvls = list()
    for c in clst2:
        pvls.append(c.pvals[0])
    assert (clst2.pvals == pvls).all()

    # plotting
    # --------
    clst2.dimnames, dnames = None, clst2.dimnames
    match = 'construct the cluster using the dimnames'
    with pytest.raises(TypeError, match=match):
        clst2.plot()
    clst2.dimnames = dnames

    with pytest.raises(TypeError, match='context'):
        _check_dimnames_kwargs(clst2, allow_lists=False, freq=[8, 9, 10])

    # _clusters_chan_vert_checks(dimnames, info, src, subject, subjects_dir)

    # clusterutils
    # ------------
    assert clst2.stc is None
    _check_stc(clst2)
    assert isinstance(clst2.stc, mne.SourceEstimate)

    # get clim
    data = np.array([2.3, 1., -1.5, -2.])
    vmin, vmax = _get_clim(data)
    assert vmin == -2.5
    assert vmax == 2.5

    data = np.array([2.2, 1., -1.5, -2.1])
    clim = _get_clim(data, pysurfer=True)
    assert clim['lims'][0] == -2.
    assert clim['lims'][-1] == 2.

    # cluster label (3d contour)
    label = _label_from_cluster(clst2, clst2.clusters[0][:, 2])
    label = _label_from_cluster(clst2, clst2.clusters[1][:, 2])
    assert isinstance(label, mne.Label)

    # color limits
    vmin, vmax = _get_clim(data, vmin=-2.)
    assert vmax == 2.
    vmin, vmax = _get_clim(data, vmax=3.5)
    assert vmin == -3.5

    # _prepare_cluster_description
    clst_idx = 1
    idx = clst2.get_index(freq=(8, 10))
    got_desc = _prepare_cluster_description(clst, clst_idx, idx)
    pval_desc = format_pvalue(clst.pvals[clst_idx])
    correct_desc = '8.0 - 10.0 Hz\n{}'.format(pval_desc)
    assert got_desc == correct_desc

    # _aggregate_cluster - 2d
    mask, stat, idx = _aggregate_cluster(clst2, 0, mask_proportion=0.5,
                                         retain_mass=0.65)
    correct_idx = clst2.get_index(cluster_idx=0, retain_mass=0.65,
                                  ignore_dims='vert')
    assert (idx == correct_idx)
    assert (stat == clst2.stat[idx].mean(axis=-1)).all()
    assert (mask == (clst2.clusters[0][idx].mean(axis=-1) >= 0.5)).all()

    with pytest.raises(ValueError, match='dimensions must be fully specified'):
        mask, stat, idx = _aggregate_cluster(clst2, [0, 1])

    # aggregate two clusters in 2d
    mask, stat, idx = _aggregate_cluster(clst2, [0, 1], freq=(8, 10))
    correct_idx = clst2.get_index(cluster_idx=0, freq=(8, 10),
                                  ignore_dims='vert')
    correct_mask = (clst2.clusters[[0, 1]][(slice(None),) + idx].mean(
                    axis=-1) >= 0.5)
    assert (idx == correct_idx)
    assert (stat == clst2.stat[idx].mean(axis=-1)).all()
    assert (mask == correct_mask).all()

    # _aggregate_cluster - 1d
    slice_idx = 2
    sel_clusters = [c[:, slice_idx] for c in clst2.clusters[:2]]
    clst_1d = Clusters(clst2.stat[:, slice_idx], sel_clusters, clst2.pvals[:2],
                       dimnames=[clst2.dimnames[0]], dimcoords=[None],
                       src=clst2.src, subject=clst2.subject,
                       subjects_dir=clst2.subjects_dir)
    mask, stat, idx = _aggregate_cluster(clst_1d, 0, mask_proportion=0.5,
                                         retain_mass=0.65)
    assert (mask == clst_1d.clusters[0]).all()
    assert (stat == clst_1d.stat).all()
    assert idx == (slice(None),)

    # additional checks on 1d cluster
    assert clst_1d.stc is None
    _check_stc(clst_1d)
    assert isinstance(clst_1d.stc, mne.SourceEstimate)

    desc = _prepare_cluster_description(clst_1d, 0, idx)
    assert desc == '{}'.format(format_pvalue(clst_1d.pvals[0]))

    # _get_units
    assert _get_units('time', fullname=True) == 'seconds'

    # _format_cluster_pvalues
    clst_1d.pvals = np.array([0.4, 0.2111, 0.11])
    pvals_txt = _format_cluster_pvalues(clst_1d, np.array([0, 1, 2]))
    assert pvals_txt == 'p = 0.4, 0.211, 0.11'

    pvals_txt = _format_cluster_pvalues(clst_1d, 1)
    assert pvals_txt == 'p = 0.211'

    clst_1d.pvals = None
    pvals_txt = _format_cluster_pvalues(clst_1d, 0)
    assert pvals_txt == 'p = NA'

    # create empty clusters
    clst_empty = Clusters(
        clst2.stat[slice_idx], None, None, dimcoords=[clst2.dimcoords[1]],
        dimnames=[clst2.dimnames[1]])
    clst_empty = Clusters(
        clst2.stat[slice_idx], np.zeros(0, dtype='bool'), np.zeros(0),
        dimcoords=[clst2.dimcoords[1]], dimnames=[clst2.dimnames[1]])
    clst_empty = Clusters(
        clst2.stat[slice_idx], list(), np.zeros(0),
        dimcoords=[clst2.dimcoords[1]], dimnames=[clst2.dimnames[1]])
    assert len(clst_empty) == 0


def test_clusters_safety_checks():

    # _clusters_safety_checks
    # -----------------------

    # clusters have to be of the same shape
    tmp = list()
    clusters = [np.zeros((2, 2)), np.zeros((2, 3))]
    with pytest.raises(ValueError, match='have to be of the same shape.'):
        _clusters_safety_checks(clusters, tmp, tmp, tmp, tmp, tmp)

    # clusters have to be boolean arrays
    clusters[1] = clusters[1][:, :2]
    with pytest.raises(TypeError, match='have to be boolean arrays.'):
        _clusters_safety_checks(clusters, tmp, tmp, tmp, tmp, tmp)

    # stat has to be a numpy array
    clusters = [np.zeros((2, 2), dtype='bool') for _ in range(2)]
    with pytest.raises(TypeError, match='must be a numpy array.'):
        _clusters_safety_checks(clusters, tmp, 'abc', tmp, tmp, tmp)

    # stat has to be of the same shape as each cluster
    stat = np.zeros((2, 3))
    with pytest.raises(ValueError, match='same shape as stat.'):
        _clusters_safety_checks(clusters, tmp, stat, tmp, tmp, tmp)

    # clusters have to be a list of arrays or one array
    with pytest.raises(TypeError, match='list of arrays or one array'):
        _clusters_safety_checks('abc', tmp, stat, tmp, tmp, tmp)

    # pvals have to be a list of floats or numpy array
    stat = np.zeros((2, 2))
    with pytest.raises(TypeError, match='list of floats or numpy array'):
        _clusters_safety_checks(clusters, 'abc', stat, tmp, tmp, tmp)

    # dimnames have to be a list
    with pytest.raises(TypeError, match='`dimnames` must be a list'):
        _clusters_safety_checks(clusters, [0.1, 0.2], stat, 'abc', tmp, tmp)

    # each dimnames element has to be a string
    match_str = "are not strings, for example: <class 'int'>"
    with pytest.raises(TypeError, match=match_str):
        _clusters_safety_checks(clusters, [0.1, 0.2], stat, [1], tmp, tmp)

    # there has to be the same number of dimnames as there are stat dimensions
    with pytest.raises(ValueError, match='Length of `dimnames` must be'):
        _clusters_safety_checks(clusters, [0.1, 0.2], stat, ['a', 'b', 'c'],
                                tmp, tmp)

    # spatial dimension has to be the first one
    with pytest.raises(ValueError, match='must be the first dimension'):
        _clusters_safety_checks(clusters, [0.1, 0.2], stat, ['freq', 'chan'],
                                tmp, tmp)

    # dimcoords has to be a list
    with pytest.raises(TypeError, match='`dimcoords` must be a list of'):
        _clusters_safety_checks(clusters, [0.1, 0.2], stat, ['chan', 'freq'],
                                'abc', tmp)

    # each dimcoords element has to match the leangth of corresponding stat dim
    match = 'The length of each dimension coordinate'
    with pytest.raises(ValueError, match=match):
        _clusters_safety_checks(clusters, [0.1, 0.2], stat, ['chan', 'freq'],
                                [np.arange(2), np.arange(3)], tmp)

    # number of dimcoords has to match the number of dimensions
    stat = np.random.rand(5, 10)
    clusters = [np.random.rand(5, 10) > 0.8 for _ in range(3)]
    info = mne.create_info(list('abcde'), 250.)

    with pytest.raises(ValueError, match='Length of `dimcoords` must be'):
        Clusters(stat, clusters, [0.1, 0.1, 0.15], dimnames=['chan', 'time'],
                 dimcoords=[None], info=info)

    # _check_description
    with pytest.raises(TypeError, match='has to be either a string or a dict'):
        _check_description(['abc'])

    # _clusters_chan_vert_checks
    # --------------------------

    # sensor space (chan dimname) require passing an mne.Info
    with pytest.raises(TypeError, match='must pass an `mne.Info`'):
        _clusters_chan_vert_checks(['chan', 'freq'], None, None, None, None,
                                   None)

    # source space (vert dimname) requires mne.SourceSpaces
    with pytest.raises(TypeError, match='must pass an `mne.SourceSpaces`'):
        _clusters_chan_vert_checks(['vert', 'freq'], None, None, None, None,
                                   None)

    # source space also requires subject ...
    with pytest.raises(TypeError, match='must pass a subject string'):
        _clusters_chan_vert_checks(['vert', 'freq'], None, None, fwd['src'],
                                   None, None)

    # ... and subjects_dir
    subjects_dir = mne.utils.get_subjects_dir()
    if subjects_dir is None:
        with pytest.raises(TypeError, match='must pass a `subjects_dir`'):
            _clusters_chan_vert_checks(['vert', 'freq'], None, None,
                                       fwd['src'], 'fsaverage', None)

    # if vertices are used - they can't exceeds source space size
    n_vert_lh = len(fwd['src'][0]['vertno'])
    n_vert = n_vert_lh + len(fwd['src'][1]['vertno'])
    vertices = np.arange(n_vert + 1)
    with pytest.raises(ValueError, match='vertex indices exceed'):
        _clusters_chan_vert_checks(['vert', 'freq'], [vertices, None],
                                   None, fwd['src'], 'fsaverage', data_dir)

    # if correct vertices are passed they are turned to a dictionary
    vertices = np.array([2, 5, n_vert_lh + 2, n_vert_lh + 5])
    dimcoords = _clusters_chan_vert_checks(['vert', 'freq'], [vertices, None],
                                           None, fwd['src'], 'fsaverage',
                                           data_dir)
    assert (dimcoords[0]['lh'] == np.array([2, 5])).all()
    assert (dimcoords[0]['rh'] == np.array([2, 5])).all()

    # if dict vertices are passed they are left untouched (if they are OK)
    vert_idx = np.array([2, 5])
    vertices = dict(lh=vert_idx, rh=vert_idx)
    dimcoords = _clusters_chan_vert_checks(['vert', 'freq'], [vertices, None],
                                           None, fwd['src'], 'fsaverage',
                                           data_dir)
    assert (dimcoords[0]['lh'] == vert_idx).all()
    assert (dimcoords[0]['rh'] == vert_idx).all()

    # if dict vertices exceed the space, and error is raised
    vert_idx = np.array([2, 5])
    vertices = dict(lh=vert_idx, rh=vert_idx + n_vert_lh)
    with pytest.raises(ValueError, match='vertex indices exceed'):
        dimcoords = _clusters_chan_vert_checks(
            ['vert', 'freq'], [vertices, None], None, fwd['src'], 'fsaverage',
            data_dir)


def test_cluster_pvals_and_polarity_sorting():
    pvals = np.array([0.5, 0.1, 0.32, 0.002, 0.73])
    stat = np.array([-1, -1, 1, -1, 1])
    correct_sorting = np.argsort(pvals)

    clusters_tmp = np.zeros(len(pvals), dtype='bool')
    clusters = list()
    for idx in range(len(pvals)):
        this_cluster = clusters_tmp.copy()
        this_cluster[idx] = True
        clusters.append(this_cluster)

    dimnames = ['freq']
    dimcoords = [np.arange(3, 8)]

    clst_nosrt = Clusters(stat, clusters, pvals, dimnames=dimnames,
                          dimcoords=dimcoords, sort_pvals=False)
    clst_srt = Clusters(stat, clusters, pvals, dimnames=dimnames,
                        dimcoords=dimcoords, sort_pvals=True)

    # make sure polarities are correct:
    correct_polarity = {1: 'pos', -1: 'neg'}
    assert all(correct_polarity[val] == pol
               for val, pol in zip(stat, clst_nosrt.cluster_polarity))

    # make sure pvals are correctly sorted
    assert (clst_srt.pvals == pvals[correct_sorting]).all()

    # make sure polarities are correctly sorted:
    assert (np.array(clst_srt.cluster_polarity)
            == np.array(clst_nosrt.cluster_polarity)[correct_sorting]).all()

    # make sure clusters are correctly sorted
    assert (correct_sorting == [np.where(c)[0][0] for c in clst_srt.clusters]
            ).all()


def test_chan_freq_clusters():
    from mne import create_info
    from mne.externals import h5io
    import matplotlib.pyplot as plt

    fname = op.join(data_dir, 'chan_alpha_range.hdf5')
    data_dict = h5io.read_hdf5(fname)

    try:
        info = create_info(data_dict['dimcoords'][0], sfreq=250.,
                           ch_types='eeg', montage='easycap-M1')
    except TypeError:
        info = create_info(data_dict['dimcoords'][0], sfreq=250.,
                           ch_types='eeg')
        mntg = mne.channels.make_standard_montage('easycap-M1')
        info.set_montage(mntg)

    clst = Clusters(
        data_dict['stat'], data_dict['clusters'], data_dict['pvals'],
        dimnames=data_dict['dimnames'], dimcoords=data_dict['dimcoords'],
        info=info, description=data_dict['description'])

    topo = clst.plot(cluster_idx=1, freq=(8, 8.5))
    plt.close(topo.fig)
    clst.clusters = None
    topo = clst.plot(freq=(10, 11.5))
    plt.close(topo.fig)
    topo = clst.plot(freq=(10, 11.5), contours=4)
    plt.close(topo.fig)

    # multi axes:
    topo = clst.plot(cluster_idx=1, freq=[8, 10])
    assert len(topo.axes) == 2
    plt.close(topo.fig)

    marker_kwargs = dict(marker='+')
    topo = clst.plot(cluster_idx=1, freq=[8, 10], mark_kwargs=marker_kwargs)
    plt.close(topo.fig)


def test_cluster_ignore_dims():
    '''Test 1. ignoring cluster dimensions in aggregation and 2. dimension
    selection in plotting.'''

    # Evoked-like tests
    # -----------------
    clst = _create_random_clusters()
    data = clst.stat.copy()

    # test that _handle_dims works well
    assert _handle_dims(clst, 'chan') == [0]
    assert _handle_dims(clst, 'time') == [1]
    assert (_handle_dims(clst, ['chan', 'time']) == np.array([0, 1])).all()

    # _handle_dims can only take None as the second arg if the first dim
    # in the CLusters is spatial
    clst2 = clst.copy()
    clst2.dimnames = clst2.dimnames.copy()
    clst2.dimnames[0] = 'cheese'
    with pytest.raises(ValueError, match="Can't infer the dimensions to plot"):
        _handle_dims(clst2, None)

    # check aggregation with ignored dims passed
    clst_mask, clst_stat, _ = _aggregate_cluster(clst, [None])
    assert clst_mask is None
    assert (clst_stat == data.mean(axis=-1)).all()

    clst_mask, clst_stat, _ = _aggregate_cluster(clst, [None], ignore_dims=[])
    assert clst_mask is None
    assert (clst_stat == data.mean(axis=(0, 1))).all()

    clst_mask, clst_stat, _ = _aggregate_cluster(
        clst, [None], ignore_dims=['time'])
    assert clst_mask is None
    assert (clst_stat == data.mean(axis=0)).all()

    mark_kwargs = {'markerfacecolor': 'seagreen',
                   'markeredgecolor': 'seagreen'}
    topo = clst.plot(cluster_idx=0, time=[0.05, 0.15, 0.25, 0.35],
                     outlines='skirt', extrapolate='head',
                     mark_kwargs=mark_kwargs)

    # * topo should be Topographies
    assert isinstance(topo, borsar.viz.Topo)
    assert len(topo) == 4

    # * topo channels should have the picked color
    assert topo.marks[1][0].get_markerfacecolor() == 'seagreen'
    assert topo.marks[1][0].get_markeredgecolor() == 'seagreen'

    # * check mask reduction - whether channel marks are where they should be
    times = clst.dimcoords[1]
    t2 = find_index(times, 0.15)
    mask = clst.clusters[0][:, t2]
    mark_pos = np.stack([x.data for x in topo.marks[1][0].get_data()], axis=1)
    assert (topo.chan_pos[mask, :] == mark_pos).all()

    # check whether topography is correct:
    assert (topo.values[:, 1] == clst.stat[:, t2]).all()

    # time-frequency test
    # -------------------
    clst = _create_random_clusters(dims='ch_fr_tm')
    data = clst.stat.copy()
    clusters = clst.clusters.copy()

    clst_mask, clst_stat, _ = _aggregate_cluster(clst, [None])
    assert clst_mask is None
    assert (clst_stat == data.mean(axis=(1, 2))).all()

    clst_mask, clst_stat1, _ = _aggregate_cluster(
        clst, [None], ignore_dims=['freq', 'time'])
    assert clst_mask is None
    assert (clst_stat1 == data.mean(axis=0)).all()

    clst_mask, clst_stat, _ = _aggregate_cluster(
        clst, [None], ignore_dims=['freq'])
    assert clst_mask is None
    assert (clst_stat == data.mean(axis=(0, 2))).all()

    # make sure two lists or two arrays work
    clst_mask, clst_stat, _ = _aggregate_cluster(
        clst, 0, ignore_dims=['chan'], freq=[8, 9, 10, 11],
        time=[0.15, 0.17, 0.19])

    freq_idx = np.array([3, 4, 5, 6])[np.newaxis, :, np.newaxis]
    time_idx = np.array([17, 18, 19])[np.newaxis, np.newaxis, :]
    assert (clst_mask == clusters[0][:, freq_idx, time_idx]).all()
    assert (clst_stat == data[:, freq_idx, time_idx]).all()

    # test heatmap
    # ------------
    clst.clusters[0] = np.zeros(clst.stat.shape, dtype='bool')
    clst.clusters[0, 4:7, 8:13, 20:26] = True

    # (A) 65% channels reduced automatically
    axs = clst.plot(cluster_idx=0, dims=['freq', 'time'])

    # make sure image data is correct
    data = np.array(axs[0].images[0].get_array())
    assert (clst.stat[5:7].mean(axis=0) == data).all()

    # (B) we request all channels reduced
    axs = clst.plot(cluster_idx=0, dims=['freq', 'time'], chan=1.)

    # make sure image data is correct
    data = np.array(axs[0].images[0].get_array())
    assert (clst.stat[4:7].mean(axis=0) == data).all()

    # (C) make sure axes are annotated correctly
    xlab = axs[0].get_xlabel()
    ylab = axs[0].get_ylabel()
    assert ylab == 'Frequency (Hz)'
    assert xlab == 'Time (s)'

    # (D) test that axes are inverted when dimension order is
    axs = clst.plot(cluster_idx=0, dims=['time', 'freq'])
    data = np.array(axs[0].images[0].get_array())
    assert (clst.stat[5:7].mean(axis=0).T == data).all()
    assert axs[0].get_xlabel() == 'Frequency (Hz)'
    assert axs[0].get_ylabel() == 'Time (s)'

    # test empty clusters -> no outlines
    clst2 = clst.copy()
    clst2.clusters = clst2.clusters.copy()
    clst2.clusters[0] = False
    axs = clst.plot(cluster_idx=0, dims=['time', 'freq'])
    # FIXME - check that there is no outlines now / no cluster mask

    # _get_dimcoords
    coords = _get_dimcoords(clst, 1)
    assert (coords == clst.dimcoords[1]).all()

    coords = _get_dimcoords(clst, 2)
    assert (coords == clst.dimcoords[2]).all()

    n_channels = clst.stat.shape[0]
    coords = _get_dimcoords(clst, 0)
    assert (coords == np.arange(n_channels)).all()

    idx = slice(2, 5)
    coords = _get_dimcoords(clst, 0, idx=idx)
    assert (coords == np.arange(n_channels)[idx]).all()

    # _label_axis (aspects that are not tested earlier)
    fig, ax = plt.subplots()
    _label_axis(ax, clst, 0, 'x')
    assert ax.get_xlabel() == 'Channels'

    _label_axis(ax, clst, 0, 'y')
    assert ax.get_ylabel() == 'Channels'

    # no more than 2 dims:
    with pytest.raises(ValueError, match='more than two'):
        clst.plot(cluster_idx=0, dims=['chan', 'freq', 'time'])

    # line plot
    # ---------
    ax = clst.plot(dims='time')
    chld = ax.get_children()
    rectangles = [ch for ch in chld
                  if isinstance(ch, mpl.patches.Rectangle)]
    assert len(rectangles) > 1
    assert np.any([isinstance(ch, mpl.lines.Line2D) for ch in chld])


def test_multi_cluster_plots():
    clst = _create_random_clusters(dims='ch_fr_tm', n_clusters=3)
    clst.plot(cluster_idx=[0, 1], dims=['chan', 'time'], freq=(8, 11))
    clst.plot(cluster_idx=[0, 1], dims=['chan', 'time'], freq=(8, 11),
              cluster_colors=['red', 'seagreen'])
    clst.plot(cluster_idx=[0, 1], time=(0.15, 0.25), freq=[8, 9, 10])
    clst.plot(cluster_idx=[0, 1], time=(0.15, 0.25), freq=[8, 9, 10],
              cluster_colors=['red', 'seagreen'])

    # one cluster in list:
    topo = clst.plot(cluster_idx=[0])
    assert isinstance(topo, borsar.viz.Topo)
    assert len(topo) == 1

    topo2 = clst.plot(cluster_idx=0)
    assert (topo.img.get_array() == topo2.img.get_array()).all()

    plt.close('all')


def test_clst_with_arrays():
    '''Test that using array indexing in .plot() .get_index() and
    .find_range() works as expected.'''
    clst = _create_random_clusters(dims='ch_fr_tm', n_clusters=3)

    # see if plots work without errors:
    clst.plot(0, freq=np.array([6, 7, 8]))
    clst.plot(np.array([0, 1]), time=(0.1, 0.2), freq=np.array([10, 11]))
    clst.plot(np.array([0, 1]), time=[0.1, 0.12, 0.15], freq=np.array([10]))
    clst.plot(np.array([0, 1]), time=[0.1, 0.12, 0.15], freq=np.array([10]),
              cluster_colors=['red', 'green'])

    plt.close('all')


@pytest.mark.skip(reason="mayavi kills CI tests")
def test_mayavi_viz():
    # mayavi import adapted from mne:
    with warnings.catch_warnings(record=True):  # traits
        from mayavi import mlab
    mlab.options.backend = 'test'

    clst2 = read_cluster(op.join(data_dir, 'temp_clst.hdf5'), src=fwd['src'],
                         subjects_dir=data_dir)
    os.remove(op.join(data_dir, 'temp_clst.hdf5'))

    # mayavi plotting
    # ---------------
    # only smoke tests currently
    brain = clst2.plot(0, freq=[8, 9], set_light=False)
    fig = brain._figures[0][0]
    mlab.close(fig)

    brain = clst2.plot(1, freq=0.7, set_light=False)
    fig = brain._figures[0][0]
    mlab.close(fig)
