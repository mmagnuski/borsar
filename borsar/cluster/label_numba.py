import numpy as np
import numba
from numba import jit

from .label import _per_channel_adjacency_3d


def _cluster_3d_numba(data, adjacency=None, min_adj_ch=0):
    """Cluster data using numba-optimized functions.

    WARNING, ``min_adj_ch > 0`` can modify data in-pace! Pass a copy of the
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
    """
    # data has to be bool
    assert data.dtype == np.bool

    if min_adj_ch > 0:
        adj_ch = _check_adj_ch_3d(data, adjacency)
        msk = adj_ch < min_adj_ch
        data[msk] = False  # warning, in-place modification

    clusters = _per_channel_adjacency_3d(data)
    return _relabel_clusters_3d(clusters, adjacency)


def _cluster_2d_numba(data, adjacency=None, min_adj_ch=0):
    """Cluster 2d channels x dim2 data using numba-optimized functions.

    WARNING, ``min_adj_ch > 0`` can modify data in-pace! Pass a copy of the
    data (``data.copy()``) if you don't want that to happen.
    This in-place modification does not happen in ``find_clusters`` function
    which passes ``data > threshold`` to lower-level functions like this one
    (so it creates new data array each time that is used only in lower-level
    clustering function).

    Parameters
    ----------
    data : numpy array
        Matrix boolean array of shape ``(channels, dim2)``.
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
    """
    # data has to be bool
    assert data.dtype == np.bool

    if min_adj_ch > 0:
        adj_ch = _check_adj_ch_2d(data, adjacency)
        msk = adj_ch < min_adj_ch
        data[msk] = False  # warning, in-place modification

    clusters = _per_channel_adjacency_2d_numba(data)
    return _relabel_clusters_2d(clusters, adjacency)


@jit(nopython=True)
def _replace_numba_3d(mat, val1, val2):
    """Numba version of ``mat[mat == val1] = val2`` for 3d arrays.

    About 4.6 faster than normal numpy.
    """
    i1, i2, i3 = mat.shape
    for idx1 in range(i1):
        for idx2 in range(i2):
            for idx3 in range(i3):
                if mat[idx1, idx2, idx3] == val1:
                    mat[idx1, idx2, idx3] = val2
    return mat


@jit(nopython=True)
def _relabel_clusters_3d(clusters, chan_conn):
    """Check channel neighbours and merge clusters across channels."""
    n_chan, n_x, n_y = clusters.shape
    for ch in range(n_chan - 1):  # last channel will be already checked
        # get unchecked neighbours
        neighbours = np.where(chan_conn[ch + 1:, ch])[0]
        if neighbours.shape[0] > 0:
            neighbours += ch + 1
            for idx1 in range(n_x):
                for idx2 in range(n_y):
                    val1 = clusters[ch, idx1, idx2]
                    if val1:
                        for ngb in neighbours:
                            val2 = clusters[ngb, idx1, idx2]
                            if val2 and not (val1 == val2):
                                c1 = min(val1, val2)
                                c2 = max(val1, val2)
                                clusters = _replace_numba_3d(clusters, c2, c1)
                                val1 = c1
    return clusters


@jit(nopython=True)
def _check_adj_ch_3d(clusters, chan_conn):
    """Check number of channel neighbours."""
    n_chan, n_x, n_y = clusters.shape
    adj_ch = np.zeros((n_chan, n_x, n_y), dtype=numba.int32)

    for ch in range(n_chan - 1):  # last channel will be already checked
        # get unchecked neighbours
        neighbours = np.where(chan_conn[ch + 1:, ch])[0]
        if neighbours.shape[0] > 0:
            neighbours = neighbours + ch + 1
            for idx1 in range(n_x):
                for idx2 in range(n_y):
                    val1 = clusters[ch, idx1, idx2]
                    if val1:
                        for ngb in neighbours:
                            val2 = clusters[ngb, idx1, idx2]
                            if val2:
                                adj_ch[ch, idx1, idx2] = (
                                    adj_ch[ch, idx1, idx2] + 1)
                                adj_ch[ngb, idx1, idx2] = (
                                    adj_ch[ngb, idx1, idx2] + 1)
    return adj_ch


@jit(nopython=True)
def _label_1d(data, from_idx=0):
    n_pnts = len(data)
    clusters = np.zeros(n_pnts)
    is_previous = False
    for ix in range(n_pnts):
        if data[ix]:
            if not is_previous:
                is_previous = True
                from_idx += 1
            clusters[ix] = from_idx
        elif is_previous:
            is_previous = False

    return clusters, from_idx



@jit(nopython=True)
def _relabel_clusters_2d(clusters, adjacency):
    """Check channel neighbours and merge clusters across channels."""
    n_chan, n_x = clusters.shape
    for ch in range(n_chan - 1):  # last channel will be already checked
        # get unchecked neighbours
        neighbours = np.where(adjacency[ch + 1:, ch])[0]
        if neighbours.shape[0] > 0:
            neighbours += ch + 1
            for idx1 in range(n_x):
                val1 = clusters[ch, idx1]
                if val1:
                    for ngb in neighbours:
                        val2 = clusters[ngb, idx1]
                        if val2 and not (val1 == val2):
                            c1 = min(val1, val2)
                            c2 = max(val1, val2)
                            clusters = _replace_numba_2d(clusters, c2, c1)
    return clusters


@jit(nopython=True)
def _replace_numba_2d(mat, val1, val2):
    """Numba version of ``mat[mat == val1] = val2`` for 2d arrays.

    About [???] faster than normal numpy.
    """
    i1, i2 = mat.shape
    for idx1 in range(i1):
        for idx2 in range(i2):
            if mat[idx1, idx2] == val1:
                mat[idx1, idx2] = val2
    return mat


@jit(nopython=True)
def _per_channel_adjacency_2d_numba(data):
    '''Identify clusters within channels where each channel contains 1d
    space.'''

    from_idx = 0
    n_chan, n_pnts = data.shape
    clusters_all = np.empty((n_chan, n_pnts), dtype=np.int64)
    # label each channel separately
    for ch in range(n_chan):
        clusters, from_idx = _label_1d(data[ch, :], from_idx=from_idx)
        clusters_all[ch, :] = clusters

    return clusters_all


@jit(nopython=True)
def _check_adj_ch_2d(clusters, adjacency):
    """Check number of channel neighbours."""
    n_chan, n_x = clusters.shape
    adj_ch = np.zeros((n_chan, n_x), dtype=numba.int32)

    for ch in range(n_chan - 1):  # last channel will be already checked
        # get unchecked neighbours
        neighbours = np.where(adjacency[ch + 1:, ch])[0]
        if neighbours.shape[0] > 0:
            neighbours = neighbours + ch + 1
            for idx1 in range(n_x):
                val1 = clusters[ch, idx1]
                if val1:
                    for ngb in neighbours:
                        val2 = clusters[ngb, idx1]
                        if val2:
                            adj_ch[ch, idx1] += 1
                            adj_ch[ngb, idx1] += 1
    return adj_ch
