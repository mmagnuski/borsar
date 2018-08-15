import numpy as np
from scipy import sparse

from borsar.stats import compute_regression_t


def construct_adjacency_matrix(neighbours, ch_names=None, as_sparse=False):
    '''
    Construct adjacency matrix out of neighbours structure (fieldtrip format).
    '''
    # checks for ch_names
    if ch_names is not None:
        assert isinstance(ch_names, list), 'ch_names must be a list.'
        assert all(map(lambda x: isinstance(x, str), ch_names)), \
            'ch_names must be a list of strings'
    else:
        ch_names = neighbours['label'].tolist()

    n_channels = len(ch_names)
    conn = np.zeros((n_channels, n_channels), dtype='bool')

    for ii, chan in enumerate(ch_names):
        ngb_ind = np.where(neighbours['label'] == chan)[0]

        # safty checks:
        if len(ngb_ind) == 0:
            raise ValueError(('channel {} was not found in neighbours.'
                              .format(chan)))
        elif len(ngb_ind) == 1:
            ngb_ind = ngb_ind[0]
        else:
            raise ValueError('found more than one neighbours entry for '
                             'channel name {}.'.format(chan))

        # find connections and fill up adjacency matrix
        connections = [ch_names.index(ch) for ch in neighbours['neighblabel']
                       [ngb_ind] if ch in ch_names]
        chan_ind = ch_names.index(chan)
        conn[chan_ind, connections] = True
    if as_sparse:
        return sparse.coo_matrix(conn)
    return conn



def cluster_based_regression(data, preds, adjacency=None, n_permutations=1000,
                             stat_threshold=None, alpha_threshold=0.05,
                             progressbar=True, return_distribution=False):

    # data has to have observations as 1st dim and channels/vert as last dim
    from mne.stats.cluster_level import (_setup_connectivity, _find_clusters,
                                         _cluster_indices_to_mask)

    assert preds.ndim == 1 or (preds.ndim == 2) & (preds.shape[1] == 1), (
        '`preds` must be 1d array or 2d array where the second dimension is'
        ' one (only one predictor).')

    if stat_threshold is None:
        from scipy.stats import t
        df = data.shape[0] - 2 # in future: preds.shape[1]
        stat_threshold = t.ppf(1 - alpha_threshold / 2, df)

    # TODO - move this piece of code to utils
    #      - then support tqdm pbar as input
    if progressbar:
        from tqdm import tqdm_notebook
        pbar = tqdm_notebook(total=n_permutations)

    n_obs = data.shape[0]
    if adjacency is not None:
        adjacency = _setup_connectivity(adjacency, np.prod(data.shape[1:]),
                                        data.shape[1])

    pos_dist = np.zeros(n_permutations)
    neg_dist = np.zeros(n_permutations)
    perm_preds = preds.copy()

    # regression on non-permuted data
    t_values = compute_regression_t(data, preds)[1]

    cluster_data = (t_values.ravel() if adjacency is not None
                    else t_values)
    clusters, cluster_stats = _find_clusters(
        cluster_data, threshold=stat_threshold, tail=0, connectivity=adjacency)

    if adjacency is not None:
        clusters = _cluster_indices_to_mask(clusters, np.prod(data.shape[1:]))
    clusters = [clst.reshape(data.shape[1:]) for clst in clusters]

    if not clusters:
        print('No clusters found, permutations are not performed.')
        return t_values, clusters, cluster_stats
    else:
        msg = 'Found {} clusters, computing permutations.'
        print(msg.format(len(clusters)))

    # compute permutations
    for perm in range(n_permutations):
        # permute predictors
        perm_inds = np.random.permutation(n_obs)
        this_perm = perm_preds[perm_inds]
        perm_tvals = compute_regression_t(data, this_perm)

        # cluster
        cluster_data = (perm_tvals[1].ravel() if adjacency is not None
                        else perm_tvals[1])
        _, perm_cluster_stats = _find_clusters(
            cluster_data, threshold=stat_threshold, tail=0,
            connectivity=adjacency)

        # if any clusters were found - add max statistic
        if perm_cluster_stats.shape[0] > 0:
            max_val = perm_cluster_stats.max()
            min_val = perm_cluster_stats.min()

            if max_val > 0: pos_dist[perm] = max_val
            if min_val < 0: neg_dist[perm] = min_val

        if progressbar:
            pbar.update(1)

    # compute permutation probability
    cluster_p = np.array([(pos_dist > cluster_stat).mean() if cluster_stat > 0
                          else (neg_dist < cluster_stat).mean()
                          for cluster_stat in cluster_stats])
    cluster_p *= 2 # because we use two-tail
    cluster_p[cluster_p > 1.] = 1. # probability has to be <= 1.

    # sort clusters by p value
    cluster_order = np.argsort(cluster_p)
    cluster_p = cluster_p[cluster_order]
    clusters = [clusters[i] for i in cluster_order]

    if return_distribution:
        distribution = dict(pos=pos_dist, neg=neg_dist)
        return t_values, clusters, cluster_p, distribution
    else:
        return t_values, clusters, cluster_p
