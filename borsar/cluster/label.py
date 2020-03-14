import numpy as np
from scipy import sparse

from ..utils import has_numba


def cluster_3d(data, adjacency):
    '''
    Cluster three-dimensional data given adjacency matrix.

    Parameters
    ----------
    data : numpy array
        Matrix of shape ``(channels, dim2, dim3)``
    adjacency : numpy array
        2d boolean matrix with information about channel adjacency.
        If ``chann_conn[i, j]`` is True that means channel i and j are
        adjacent.

    Returns
    -------
    clusters : array of int
        3d integer matrix with cluster labels.
    '''
    # data has to be bool
    assert data.dtype == np.bool

    # nested import
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

    # unrolled views into clusters for ease of channel comparison:
    unrolled = [clusters[ch, :].ravel() for ch in range(n_chan)]
    # check channel neighbours and merge clusters across channels
    for ch in range(n_chan - 1):  # last chan will be already checked
        ch1 = unrolled[ch]
        ch1_ind = np.where(ch1)[0]
        if len(ch1_ind) == 0:
            continue  # no clusters, no fun...

        # get unchecked neighbours
        neighbours = np.where(adjacency[ch + 1:, ch])[0]
        if len(neighbours) > 0:
            neighbours += ch + 1

            for ngb in neighbours:
                ch2 = unrolled[ngb]
                for ind in ch1_ind:
                    # relabel clusters if adjacent and not the same id
                    if ch2[ind] and not (ch1[ind] == ch2[ind]):
                        c1 = min(ch1[ind], ch2[ind])
                        c2 = max(ch1[ind], ch2[ind])
                        clusters[clusters == c2] = c1
    return clusters


def _get_cluster_fun(data, adjacency=None, backend='numpy'):
    '''Return the correct clustering function depending on the data shape and
    presence of an adjacency matrix.'''
    has_adjacency = adjacency is not None
    if data.ndim == 3 and has_adjacency:
        if backend in ['numba', 'auto']:
            if has_numba():
                from .label_numba import cluster_3d_numba
                return cluster_3d_numba
            elif backend == 'numba':
                raise ValueError('You need numba package to use the "numba" '
                                 'backend.')
            else:
                return cluster_3d
        else:
            return cluster_3d


# TODO : add tail=0 to control for tail selection
def find_clusters(data, threshold, adjacency=None, cluster_fun=None,
                  backend='auto', mne_reshape_clusters=True):
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

    Returns
    -------
    clusters : list
        List of boolean arrays informing about membership in consecutive
        clusters. For example `clusters[0]` informs about data points that
        belong to the first cluster.
    cluster_stats : numpy array
        Array with cluster statistics - usually sum of cluster members' values.
    """
    if cluster_fun is None and backend == 'auto':
        backend = 'mne' if data.ndim < 3 else 'auto'

    if backend == 'mne':
        # mne clustering
        # --------------
        from mne.stats.cluster_level import (
            _find_clusters, _cluster_indices_to_mask, _setup_connectivity)

        # FIXME more checks for adjacency and data when using mne!
        if adjacency is not None and isinstance(adjacency, np.ndarray):
            if not sparse.issparse(adjacency):
                adjacency = sparse.coo_matrix(adjacency)
            if adjacency.ndim == 2:
                adjacency = _setup_connectivity(adjacency, np.prod(data.shape),
                                                data.shape[0])

        orig_data_shape = data.shape
        data = (data.ravel() if adjacency is not None else data)
        clusters, cluster_stats = _find_clusters(
            data, threshold=threshold, tail=0, connectivity=adjacency)

        if mne_reshape_clusters:
            if adjacency is not None:
                clusters = _cluster_indices_to_mask(clusters,
                                                    np.prod(data.shape))
            clusters = [clst.reshape(orig_data_shape) for clst in clusters]
    else:
        # borsar clustering
        # -----------------
        if cluster_fun is None:
            cluster_fun = _get_cluster_fun(data, adjacency=adjacency,
                                           backend=backend)
        # positive clusters
        # -----------------
        pos_clusters = cluster_fun(data > threshold, adjacency=adjacency)

        # TODO - consider numba optimization of this part too:
        cluster_id = np.unique(pos_clusters)[1:]
        pos_clusters = [pos_clusters == id for id in cluster_id]
        cluster_stats = [data[clst].sum() for clst in pos_clusters]

        # negative clusters
        # -----------------
        neg_clusters = cluster_fun(data < -threshold, adjacency=adjacency)

        # TODO - consider numba optimization of this part too:
        cluster_id = np.unique(neg_clusters)[1:]
        neg_clusters = [neg_clusters == id for id in cluster_id]
        cluster_stats = np.array(cluster_stats + [data[clst].sum()
                                                  for clst in neg_clusters])
        clusters = pos_clusters + neg_clusters

    return clusters, cluster_stats
