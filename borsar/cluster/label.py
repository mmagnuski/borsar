import numpy as np
from scipy import sparse

from ..utils import has_numba


# TODO:
# - [ ] compare speed against mne clustering
# - [x] add min_adj_ch (minimum adjacent channels)
# - [x] wait with relabeling till the end (tried that and it was slower)
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
        Matrix boolean array of shape ``(channels, dim2, dim3)``.
    adjacency : numpy array
        Twodimensional boolean matrix with information about channel/vertex
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
        data = _cross_channel_adjacency_3d(data, adjacency, min_adj_ch=min_adj_ch)

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
        ch1_whereidx = np.where(ch1_val)[0]
        if len(ch1_whereidx) == 0:
            continue  # no clusters, no fun...

        # get unchecked neighbours
        neighbours = np.where(adjacency[ch1_idx + 1:, ch1_idx])[0]
        if len(neighbours) > 0:
            neighbours += ch1_idx + 1

            for ngb_idx in neighbours:
                ch2_val = unrolled[ngb_idx]
                for ind in ch1_whereidx:
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


def _get_cluster_fun(data, adjacency=None, backend='numpy', min_adj_ch=0):
    '''Return the correct clustering function depending on the data shape and
    presence of an adjacency matrix.'''
    has_adjacency = adjacency is not None
    if data.ndim == 3 and has_adjacency:
        if backend in ['numba', 'auto']:
            hasnb = has_numba()
            if hasnb:
                from .label_numba import _cluster_3d_numba
                return _cluster_3d_numba
            elif backend == 'numba' and not hasnb:
                raise ValueError('You need numba package to use the "numba" '
                                 'backend.')
            else:
                return _cluster_3d_numpy
        else:
            return _cluster_3d_numpy
    else:
        raise ValueError('borsar has specialised clustering functions only'
                         ' for three dimensional data where the first dimen'
                         'sion is spatial (channels or vertices). This spat'
                         'ial dimension requires adjacency matrix defining '
                         'adjacency relationships. Your data is either not'
                         'three-dimensional or you did not provide an adja'
                         'cency matrix for the spatial dimension')


# TODO : add tail=0 to control for tail selection
def find_clusters(data, threshold, adjacency=None, cluster_fun=None,
                  backend='auto', mne_reshape_clusters=True, min_adj_ch=0):
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
        which assumes standard lattuce adjacency.
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
        When ``backend`` is ``'mne'``: wheteher to reshape clusters back to
        the original data shape after obtaining them from mne. Not used for
        other backends.
    min_adj_ch
        Number of minimum adjacent channels/vertices above ``threshold`` for
        given point to be included in a cluster.

    Returns
    -------
    clusters : list
        List of boolean arrays informing about membership in consecutive
        clusters. For example ``clusters[0]`` informs about data points that
        belong to the first cluster.
    cluster_stats : numpy array
        Array with cluster statistics - usually sum of cluster members' values.
    """
    findfunc, adjacency, addarg = _prepare_clustering(
        data, adjacency, cluster_fun, backend, min_adj_ch=min_adj_ch)
    clusters, cluster_stats = findfunc(data, threshold, adjacency, addarg,
                                       min_adj_ch=min_adj_ch, full=True)

    return clusters, cluster_stats


def _prepare_clustering(data, adjacency, cluster_fun, backend, min_adj_ch=0):
    '''Prepare clustering - perform checks and create necessary variables.'''
    # FIXME - these lines should be put in _get_cluster_fun
    if cluster_fun is None and backend == 'auto':
        backend = 'mne' if data.ndim < 3 and min_adj_ch == 0 else 'auto'

    if data.ndim < 3 and min_adj_ch > 0:
        raise ValueError('currently ``min_adj_ch`` is implemented only for'
                         ' 3d clustering.')

    # mne_reshape_clusters=True,
    if backend == 'mne':
        if min_adj_ch > 0:
            raise ValueError('mne backend does not supprot ``min_adj_ch`` '
                             'filtering')

        try:
            from mne.stats.cluster_level import _setup_connectivity
            argname = 'connectivity'
        except ImportError:
            from mne.stats.cluster_level import (_setup_adjacency
                                                 as _setup_connectivity)
            argname = 'adjacency'

        if adjacency is not None and isinstance(adjacency, np.ndarray):
            if not sparse.issparse(adjacency):
                adjacency = sparse.coo_matrix(adjacency)
            if adjacency.ndim == 2:
                adjacency = _setup_connectivity(adjacency, np.prod(data.shape),
                                                data.shape[0])

        return _find_clusters_mne, adjacency, argname
    else:
        if cluster_fun is None:
            cluster_fun = _get_cluster_fun(data, adjacency=adjacency,
                                           min_adj_ch=min_adj_ch,
                                           backend=backend)
        return _find_clusters_borsar, adjacency, cluster_fun


def _find_clusters_mne(data, threshold, adjacency, argname, min_adj_ch=0,
                       full=True):
    from mne.stats.cluster_level import (
        _find_clusters, _cluster_indices_to_mask)

    orig_data_shape = data.shape
    kwargs = {argname: adjacency}
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
                          min_adj_ch=0, full=True):
    pos_clusters = cluster_fun(data > threshold, adjacency=adjacency,
                               min_adj_ch=min_adj_ch)

    # TODO - consider numba optimization of this part too:
    cluster_id = np.unique(pos_clusters)[1:]
    pos_clusters = [pos_clusters == id for id in cluster_id]
    cluster_stats = [data[clst].sum() for clst in pos_clusters]

    # negative clusters
    # -----------------
    neg_clusters = cluster_fun(data < -threshold, adjacency=adjacency,
                               min_adj_ch=min_adj_ch)

    # TODO - consider numba optimization of this part too:
    cluster_id = np.unique(neg_clusters)[1:]
    neg_clusters = [neg_clusters == id for id in cluster_id]
    cluster_stats = np.array(cluster_stats + [data[clst].sum()
                                              for clst in neg_clusters])
    clusters = pos_clusters + neg_clusters

    return clusters, cluster_stats
