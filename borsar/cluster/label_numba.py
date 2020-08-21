import numpy as np
import numba
from numba import jit

from .label import _per_channel_adjacency


def cluster_3d_numba(data, adjacency=None, min_adj_ch=0):
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
        adj_ch = _check_adj_ch(data, adjacency)
        msk = adj_ch < min_adj_ch
        data[msk] = False  # warning, in-place modification

    clusters = _per_channel_adjacency(data, adjacency)
    return _relabel_clusters(clusters, adjacency)


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
def _relabel_clusters(clusters, chan_conn):
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
    return clusters


@jit(nopython=True)
def _check_adj_ch(clusters, chan_conn):
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
