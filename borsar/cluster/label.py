import numpy as np
from scipy import sparse

try:
    import mne
    from packaging.version import Version
    MNE_1_8 = Version(mne.__version__) >= Version('1.8.0')
except ImportError:
    MNE_1_8 = False

from ..utils import has_numba


# TODO:
# - [ ] compare speed against mne clustering
def _cluster_3d_numpy(data, adjacency, min_adj_ch=0):
    '''
    Cluster three-dimensional data given adjacency matrix.

    WARNING, ``min_adj_ch > 0`` can modify data in-place! Pass a copy of the
    data (``data.copy()``) if you don't want that to happen.
    This in-place modification does not happen in ``find_clusters`` function
    which passes ``data > threshold`` to lower-level functions like this one
    (so it creates new data array each time that is used only in lower-level
    clustering function).

    Parameters
    ----------
    data : numpy array
        Boolean array of shape ``(n_channels, dim2, dim3)`` where ``dim2`` and
        ``dim3`` are arbitrary dimensions.
    adjacency : numpy array
        Two dimensional boolean matrix with information about channel/vertex
        adjacency. If ``adjacency[i, j]`` is ``True`` that means channel ``i``
        and ``j`` are adjacent.
    min_adj_ch : int
        Number of minimum adjacent True channels/vertices for given point to be
        included in a cluster.

    Returns
    -------
    clusters : array of int
        3d integer matrix with cluster labels.
    '''
    # data has to be bool
    assert data.dtype == bool

    if min_adj_ch > 0:
        data = _cross_channel_adjacency_3d(
            data, adjacency, min_adj_ch=min_adj_ch)

    clusters = _per_channel_adjacency_3d(data)
    clusters = _cross_channel_adjacency_3d(clusters, adjacency)

    return clusters


def _per_channel_adjacency_3d(data):
    '''Identify clusters within channels.'''
    from skimage.measure import label

    # label each channel separately
    clusters = np.zeros(data.shape, dtype='int')
    max_cluster_id = 0
    n_chan = data.shape[0]
    for ch in range(n_chan):
        clusters[ch, :, :] = label(
            data[ch, :, :], connectivity=1, background=False)

        # relabel so that layers do not have same cluster ids
        num_clusters = clusters[ch, :, :].max()
        if ch > 0:
            clusters[ch, clusters[ch, :] > 0] += max_cluster_id
        max_cluster_id += num_clusters

    return clusters


def _cross_channel_adjacency_3d(clusters, adjacency, min_adj_ch=0):
    '''Connect clusters identified within channels.'''
    n_chan = clusters.shape[0]
    # unrolled views into clusters for ease of channel comparison:
    unrolled = [clusters[ch, :].ravel() for ch in range(n_chan)]

    if min_adj_ch > 0:
        adj_ch = np.zeros((len(unrolled), len(unrolled[0])), dtype='int')

    # check channel neighbours and merge clusters across channels
    for ch1_idx in range(n_chan - 1):  # last chan will be already checked
        ch1_val = unrolled[ch1_idx]
        ch1_where_idx = np.where(ch1_val)[0]
        if len(ch1_where_idx) == 0:
            continue  # no clusters, no fun...

        # get unchecked neighbours
        neighbours = np.where(adjacency[ch1_idx + 1:, ch1_idx])[0]
        if len(neighbours) > 0:
            neighbours += ch1_idx + 1

            for ngb_idx in neighbours:
                ch2_val = unrolled[ngb_idx]
                for ind in ch1_where_idx:
                    if ch2_val[ind]:
                        # count number of adjacent channels for each point
                        if min_adj_ch > 0:
                            adj_ch[ch1_idx, ind] += 1
                            adj_ch[ngb_idx, ind] += 1

                        # relabel clusters if adjacent and not the same id
                        elif not (ch1_val[ind] == ch2_val[ind]):
                            c1, c2 = np.sort([ch1_val[ind], ch2_val[ind]])
                            clusters[clusters == c2] = c1

    # filter by min_adj_ch
    if min_adj_ch > 0:
        adj_ch = adj_ch.reshape(clusters.shape)
        mask = adj_ch < min_adj_ch
        clusters[mask] = 0

    return clusters


# TODO: ideally, these checks should be done in _get_cluster_fun
def _cluster_1d_or_2d_no_adj(data, adjacency=None, min_adj_ch=0):
    from skimage.measure import label
    assert adjacency is None
    assert min_adj_ch == 0
    return label(data, connectivity=1, background=False)


def _check_backend(data, adjacency=None, backend='auto', min_adj_ch=0,
                   filter_fun=None, filter_fun_post=None):
    '''Select adequate backend if 'auto', else check the selected backend.
    '''
    n_dims = data.ndim
    has_numba_lib = False
    has_adjacency = adjacency is not None

    # if backend is auto or numba - check if numba is available
    if backend in ['numba', 'auto']:
        has_numba_lib = has_numba()
    if backend == 'numba' and not has_numba_lib:
        raise ValueError('You need numba package to use the "numba" '
                         'backend.')

    # min_adj_ch requires adjacency
    if not has_adjacency and min_adj_ch > 0:
        raise ValueError('min_adj_ch > 0 requires that adjacency is not None')

    # select backend for 'auto':
    # --------------------------
    # numba, else numpy else mne
    if backend == 'auto':
        is_2d_adj = has_adjacency and n_dims in [2, 3]
        is_3d_adj = has_adjacency and n_dims == 3
        is_2d_no_adj = n_dims == 2 and not has_adjacency
        is_1d_no_adj = n_dims == 1 and not has_adjacency
        # numba works for 2d, 3d with adjacency
        if has_numba_lib and (is_3d_adj or is_2d_adj or is_1d_no_adj):
            backend = 'numba'
        elif is_3d_adj or is_2d_no_adj or is_1d_no_adj:
            backend = 'numpy'
        else:
            backend = 'mne'

    # check backend for validity
    # --------------------------
    if filter_fun is not None or filter_fun_post is not None:
        if backend == 'mne':
            raise ValueError('``filter_fun`` and ``filter_fun_post`` are '
                             "not available for the ``'mne'`` backend.")

    if not has_adjacency:
        if backend == 'numba' and n_dims > 1:
            raise ValueError('Numba backend requires an adjacency matrix for '
                             '> 1D data.')
        elif backend == 'numpy' and n_dims >= 3:
            _borsar_clustering_error()
        elif backend == 'mne' and n_dims == 3:
            # TODO: more informative error
            _borsar_clustering_error()
    else:
        adj_shape = adjacency.shape
        if not adjacency.ndim == 2 or not adj_shape[0] == adj_shape[1]:
            raise ValueError('Adjacency has to be a 2d square matrix. Got '
                             'array of {adj_shape} shape.')
        if not adj_shape[0] == len(data):
            raise ValueError('First data dimension has to correspond to the'
                             ' passed adjacency matrix. First data dimension'
                             f' is {len(data)} long, while adjacency is '
                             f'{adj_shape[0]} x {adj_shape[0]}.')

    if min_adj_ch > 0:
        if backend == 'mne':
            raise ValueError("``min_adj_ch`` is not available for the "
                             "``'mne'`` backend.")
    if backend == 'numpy' and n_dims == 2 and has_adjacency:
        raise ValueError('Currently only "numba" backend can handle 2d data'
                         ' with channel adjacency.')

    if n_dims == 1:
        if backend in ['numba', 'numpy'] and has_adjacency:
            # TODO: more informative error
            _borsar_clustering_error()

    return backend


def _get_cluster_fun(data, adjacency=None, backend='numpy', min_adj_ch=0,
                     filter_fun=None, filter_fun_post=None):
    '''Return the correct clustering function depending on the data shape and
    presence of an adjacency matrix.'''
    has_adjacency = adjacency is not None

    if data.ndim == 3:
        if backend == 'numba':
            from .label_numba import _cluster_3d_numba
            return _cluster_3d_numba
        else:
            return _cluster_3d_numpy
    elif data.ndim == 2 and backend == 'numba':
            from .label_numba import _cluster_2d_numba
            return _cluster_2d_numba
    elif data.ndim < 3 and not has_adjacency:
        if data.ndim == 1 and backend == 'numba':
            from .label_numba import _cluster_1d_numba
            return _cluster_1d_numba
        else:
            return _cluster_1d_or_2d_no_adj


def _borsar_clustering_error():
    supported = get_supported_find_clusters_parameters()
    raise ValueError(
        'borsar has specialized clustering functions only for: '
        'a) three- and two-dimensional data where the first dimension is '
        'spatial (channels or vertices). This spatial dimension requires '
        'an adjacency matrix defining the adjacency relationships; '
        'b) one-dimensional data without spatial dimension.\nCheck all the '
        f'supported options below:\n{supported}')


# TODO : add tail=0 to control for tail selection! or 'pos', 'neg' and 'both'
def find_clusters(data, threshold, adjacency=None, cluster_fun=None,
                  backend='auto', mne_reshape_clusters=True, min_adj_ch=0,
                  filter_fun=None, filter_fun_post=None):
    """Find clusters in data array given cluster membership threshold and
    optionally adjacency matrix.

    Parameters
    ----------
    data : numpy array
        Data array to cluster.
    threshold : float
        Threshold value for cluster membership.
    adjacency : numpy bool array | list, optional
        Boolean adjacency matrix. Can be dense or sparse. None by default,
        which assumes standard lattice adjacency.
    cluster_fun : function, optional
        Clustering function to use. ``None`` by default which selects relevant
        clustering function based on adjacency and number of data dimensions.
    backend : str, optional
        Clustering backend: ``'auto'``, ``'mne'``, ``'borsar'``, ``'numpy'``
        or ``'numba'``. ``'mne'`` backend can be used only for < 3d data.
        ``'borsar'`` leads to selection of numba backend if numba is available,
        otherwise numpy is used. Default is ``'auto'`` which selects the
        relevant backend automatically.
    mne_reshape_clusters : bool, optional
        When ``backend`` is ``'mne'``: whether to reshape clusters back to
        the original data shape after obtaining them from mne. Not used for
        other backends.
    min_adj_ch : int
        Number of minimum adjacent channels/vertices above ``threshold`` for
        given point to be included in a cluster.
    filter_fun : callable | None
        Additional filtering to perform on the boolean mask before clustering.
        Must be a function that receives a boolean matrix and adjacency and
        returns filtered clusters. Can be used for example to remove "pixels"
        that have less than N other neighboring pixels (irrespective of
        ``min_adj_ch``).
    filter_fun_post : callable | None
        Additional filtering to perform on the identified clusters. Must be a
        function that receives cluster ID matrix and adjacency and returns
        the filtered clusters. Can be used for example to automatically reject
        clusters that do not overlap sufficiently with channels of interest or
        that overlap with a known high variance space in the search space that
        is not of interest.

    Returns
    -------
    clusters : list
        List of boolean arrays informing about membership in consecutive
        clusters. For example ``clusters[0]`` informs about data points that
        belong to the first cluster.
    cluster_stats : numpy array
        Array with cluster statistics - usually sum of cluster members' values.
    """
    backend = _check_backend(data, adjacency, backend, min_adj_ch, filter_fun,
                             filter_fun_post)
    if cluster_fun is None:
        cluster_fun = _get_cluster_fun(
            data, adjacency=adjacency, backend=backend, min_adj_ch=min_adj_ch,
            filter_fun=filter_fun, filter_fun_post=filter_fun_post)

    find_func, adjacency, add_arg = _prepare_clustering(
        data, adjacency, cluster_fun, backend, min_adj_ch=min_adj_ch,
        filter_fun=filter_fun, filter_fun_post=filter_fun_post)
    clusters, cluster_stats = find_func(
        data, threshold, adjacency, add_arg, min_adj_ch=min_adj_ch, full=True,
        filter_fun=filter_fun, filter_fun_post=filter_fun_post)

    return clusters, cluster_stats


# TODO: backend auto option should be better!
def _prepare_clustering(data, adjacency, cluster_fun, backend, min_adj_ch=0,
                        filter_fun=None, filter_fun_post=None):
    '''Prepare clustering - perform checks and create necessary variables.'''
    import mne

    # mne_reshape_clusters=True,
    if backend == 'mne':
        if adjacency is not None:
            if not sparse.issparse(adjacency):
                adjacency = sparse.coo_matrix(adjacency)

            if data.ndim > 1:
                adjacency = mne.stats.combine_adjacency(
                    adjacency, *data.shape[1:])

        return _find_clusters_mne, adjacency, None
    else:
        if adjacency is not None and sparse.issparse(adjacency):
            # turn to dense
            adjacency = adjacency.toarray()
        if cluster_fun is None:
            cluster_fun = _get_cluster_fun(data, adjacency=adjacency,
                                           min_adj_ch=min_adj_ch,
                                           backend=backend)
        return _find_clusters_borsar, adjacency, cluster_fun


# TODO: describe the ``full`` argument better
# TODO: remove the ``not_used`` arg later
def _find_clusters_mne(data, threshold, adjacency, not_used, min_adj_ch=0,
                       full=True, filter_fun=None, filter_fun_post=None):
    '''Perform clustering using mne functions.'''
    from mne.stats.cluster_level import (
        _find_clusters, _cluster_indices_to_mask)

    orig_data_shape = data.shape
    if adjacency is not None:
        data = data.ravel()

    clusters, cluster_stats = _find_clusters(
        data, threshold=threshold, tail=0, adjacency=adjacency)

    if full:
        if adjacency is not None:
            if MNE_1_8:
                clusters = _cluster_indices_to_mask(
                    clusters, np.prod(data.shape), False)
            else:
                clusters = _cluster_indices_to_mask(
                    clusters, np.prod(data.shape))
        clusters = [clst.reshape(orig_data_shape) for clst in clusters]

    return clusters, cluster_stats


def _find_clusters_borsar(data, threshold, adjacency, cluster_fun,
                          min_adj_ch=0, full=True, filter_fun=None,
                          filter_fun_post=None):
    if isinstance(threshold, list):
        assert len(threshold) == 2
        pos_threshold, neg_threshold = threshold
    else:
        pos_threshold, neg_threshold = threshold, -threshold

    # positive clusters
    # -----------------
    if pos_threshold is not None:
        pos_clusters, pos_cluster_stats = _find_clusters_borsar_onetail(
            data, pos_threshold, adjacency, cluster_fun,
            tail='pos', min_adj_ch=min_adj_ch, filter_fun=filter_fun,
            filter_fun_post=filter_fun_post
        )
    else:
        pos_clusters, pos_cluster_stats = list(), list()

    # negative clusters
    # -----------------
    if neg_threshold is not None:
        neg_clusters, neg_cluster_stats = _find_clusters_borsar_onetail(
            data, neg_threshold, adjacency, cluster_fun,
            tail='neg', min_adj_ch=min_adj_ch, filter_fun=filter_fun,
            filter_fun_post=filter_fun_post
        )
    else:
        neg_clusters, neg_cluster_stats = list(), list()

    clusters = pos_clusters + neg_clusters
    cluster_stats = np.array(pos_cluster_stats + neg_cluster_stats)

    return clusters, cluster_stats


def _find_clusters_borsar_onetail(data, threshold, adjacency, cluster_fun,
                                  tail='pos', min_adj_ch=0,
                                  filter_fun=None, filter_fun_post=None):
    if tail == 'pos':
        compare_fun = np.greater
    elif tail == 'neg':
        compare_fun = np.less

    mask = compare_fun(data, threshold)

    if filter_fun is not None:
        mask = filter_fun(mask, adjacency=adjacency)

    clusters = cluster_fun(mask, adjacency=adjacency,
                           min_adj_ch=min_adj_ch)

    if filter_fun_post is not None:
        clusters = filter_fun_post(clusters, adjacency=adjacency)

    # TODO - consider numba optimization of this part too:
    cluster_id = np.unique(clusters)
    cluster_id = cluster_id[1:] if 0 in cluster_id else cluster_id
    clusters = [clusters == clst_id for clst_id in cluster_id]
    cluster_stats = [data[clst].sum() for clst in clusters]

    return clusters, cluster_stats


def get_supported_find_clusters_parameters():
    from io import StringIO
    import pandas as pd

    table_text = """n dimensions,channel dimension,min_adj_ch,backend,supported
        1,no,NA,numba,yes
        1,no,NA,numpy,yes
        1,yes,yes,numba,no
        1,yes,no,numba,no
        1,yes,yes,numpy,no
        1,yes,no,numpy,no
        2,no,NA,numba,no
        2,no,NA,numpy,yes
        2,yes,no,numba,yes
        2,yes,no,numpy,no
        2,yes,yes,numba,yes
        2,yes,yes,numpy,no
        3,no,NA,numba,no
        3,no,NA,numpy,no
        3,yes,no,numba,yes
        3,yes,no,numpy,yes
        3,yes,yes,numba,yes
        3,yes,yes,numpy,yes"""
    supported = pd.read_csv(StringIO(table_text), keep_default_na=False)
    return supported
