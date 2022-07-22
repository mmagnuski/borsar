import numpy as np
from scipy import sparse

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
    assert data.dtype == np.bool

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
        # numba works for 2d, 3d with adjacency
        if has_adjacency and n_dims in [2, 3] and has_numba_lib:
            backend = 'numba'
        elif ((has_adjacency and n_dims == 3)
              or (n_dims in [1, 2] and not has_adjacency)):
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
        if backend == 'numba':
            raise ValueError('Numba backend requires an adjacency matrix.')
        elif backend == 'numpy' and n_dims >= 3:
            _borsar_clustering_error()
        elif backend == 'mne' and n_dims == 3:
            # TODO: more informative error
            _borsar_clustering_error()

    if min_adj_ch > 0:
        if backend == 'mne':
            raise ValueError("``min_adj_ch`` is not available for the "
                             "``'mne'`` backend.")
    if backend == 'numpy' and n_dims == 2 and has_adjacency:
        raise ValueError('Currently only "numba" backend can handle 2d data'
                         ' with channel adjacency.')

    if n_dims == 1:
        if backend == 'numba' or (backend == 'numpy' and has_adjacency):
            # TODO: more informative error
            _borsar_clustering_error()

    return backend


def _get_cluster_fun(data, adjacency=None, backend='numpy', min_adj_ch=0,
                     filter_fun=None, filter_fun_post=None):
    '''Return the correct clustering function depending on the data shape and
    presence of an adjacency matrix.'''
    backend = _check_backend(data, adjacency, backend, min_adj_ch, filter_fun,
                             filter_fun_post)
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
        return _cluster_1d_or_2d_no_adj


def _borsar_clustering_error():
    raise ValueError('borsar has specialized clustering functions only'
                     ' for three- and two-dimensional data where the first '
                     'dimension is spatial (channels or vertices). This spat'
                     'ial dimension requires adjacency matrix defining '
                     'adjacency relationships. Your data is either not'
                     'three-dimensional or you did not provide an adja'
                     'cency matrix for the spatial dimension.')


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
        # prepare mne clustering, maybe put this in a separate function?
        try:
            from mne.stats.cluster_level import _setup_connectivity
            arg_name = 'connectivity'
        except ImportError:
            from mne.stats.cluster_level import (_setup_adjacency
                                                 as _setup_connectivity)
            arg_name = 'adjacency'

        if adjacency is not None and isinstance(adjacency, np.ndarray):
            if not sparse.issparse(adjacency):
                adjacency = sparse.coo_matrix(adjacency)

            if data.ndim > 1:
                adjacency = mne.stats.combine_adjacency(
                    adjacency, *data.shape[1:])
            # if adjacency.ndim == 2:
            #     adjacency = _setup_connectivity(
            #         adjacency, np.prod(data.shape), data.shape[0])

        return _find_clusters_mne, adjacency, arg_name
    else:
        if cluster_fun is None:
            cluster_fun = _get_cluster_fun(data, adjacency=adjacency,
                                           min_adj_ch=min_adj_ch,
                                           backend=backend)
        return _find_clusters_borsar, adjacency, cluster_fun


# TODO: describe the ``full`` argument better
def _find_clusters_mne(data, threshold, adjacency, arg_name, min_adj_ch=0,
                       full=True, filter_fun=None, filter_fun_post=None):
    '''Perform clustering using mne functions.'''
    from mne.stats.cluster_level import (
        _find_clusters, _cluster_indices_to_mask)

    orig_data_shape = data.shape
    kwargs = {arg_name: adjacency}
    data = (data.ravel() if adjacency is not None else data)
    clusters, cluster_stats = _find_clusters(
        data, threshold=threshold, tail=0, **kwargs)

    if full:
        if adjacency is not None:
            clusters = _cluster_indices_to_mask(clusters,
                                                np.prod(data.shape))
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
    pos_mask = data > pos_threshold
    if filter_fun is not None:
        pos_mask = filter_fun(pos_mask, adjacency=adjacency)
    pos_clusters = cluster_fun(pos_mask, adjacency=adjacency,
                               min_adj_ch=min_adj_ch)
    if filter_fun_post is not None:
        pos_clusters = filter_fun_post(pos_clusters, adjacency=adjacency)

    # TODO - consider numba optimization of this part too:
    cluster_id = np.unique(pos_clusters)
    cluster_id = cluster_id[1:] if 0 in cluster_id else cluster_id
    pos_clusters = [pos_clusters == id for id in cluster_id]
    cluster_stats = [data[clst].sum() for clst in pos_clusters]

    # negative clusters
    # -----------------
    neg_mask = data < neg_threshold
    if filter_fun is not None:
        neg_mask = filter_fun(neg_mask, adjacency=adjacency)
    neg_clusters = cluster_fun(neg_mask, adjacency=adjacency,
                               min_adj_ch=min_adj_ch)
    if filter_fun_post is not None:
        neg_clusters = filter_fun_post(neg_clusters, adjacency=adjacency)

    # TODO - consider numba optimization of this part too:
    cluster_id = np.unique(neg_clusters)
    cluster_id = cluster_id[1:] if 0 in cluster_id else cluster_id
    neg_clusters = [neg_clusters == id for id in cluster_id]
    cluster_stats = np.array(cluster_stats + [data[clst].sum()
                                              for clst in neg_clusters])
    clusters = pos_clusters + neg_clusters

    return clusters, cluster_stats


def get_supported_find_clusters_parameters():
    from io import StringIO
    import pandas as pd

    table_text = """n dimensions,channel dimension,min_adj_ch,backend,supported
        1,no,NA,numba,no
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
