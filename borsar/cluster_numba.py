import numpy as np
from numba import jit


def cluster_3d_numba(matrix, chan_conn):
    """Cluster data using numba-optimized functions."""
    # matrix has to be bool
    assert matrix.dtype == np.bool

    # nested import
    from skimage.measure import label

    # label each channel separately
    clusters = np.zeros(matrix.shape, dtype='int')
    max_cluster_id = 0
    n_chan = matrix.shape[0]
    for ch in range(n_chan):
        clusters[ch, :, :] = label(
            matrix[ch, :, :], connectivity=1, background=False)

        # relabel so that layers do not have same cluster ids
        if ch > 0:
            num_clusters = clusters[ch, :, :].max()
            clusters[ch, clusters[ch] > 0] += max_cluster_id
            max_cluster_id += num_clusters

    # unrolled views into clusters for ease of channel comparison:
    return relabel_clusters(clusters, chan_conn)


@jit(nopython=True)
def replace_numba_3d(mat, val1, val2):
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
def relabel_clusters(clusters, chan_conn):
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
                                clusters = replace_numba_3d(clusters, c2, c1)
    return clusters
