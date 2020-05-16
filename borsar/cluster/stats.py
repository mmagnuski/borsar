import numpy as np

from .label import find_clusters, _get_cluster_fun
from .stats import compute_regression_t, _handle_preds


# - [x] permute only some predictors
# - [x] better support for `cluster_pred`: idx with respect to `preds`,
#       add intercept
# - [ ] FIXME: consider cluster_pred always adressing preds (you never want
#              cluster the intercept, and if you do you'd need a one sample
#              t test and thus a different permutation scheme)
def cluster_based_regression(data, preds, adjacency=None, n_permutations=1000,
                             stat_threshold=None, alpha_threshold=0.05,
                             cluster_pred=None, backend='auto',
                             progressbar=True, return_distribution=False,
                             stat_fun=None):
    '''Compute cluster-based permutation test with regression as the
    statistical function.

    Parameters
    ----------
    data : numpy array
        N-dimensional numpy array with data to predict with regression. The
        first dimension has to correspond to observations. If ``adjacency`` was
        given the last dimension has to correspond to adjacency space (for
        example channels or vertices).
    preds : numpy array
        Predictors array of shape ``(n_observations, n_predictors)`` to use in
        regression.
    adjacency : numpy array, optional
        Adjacency matrix for the last ``data`` dimension. If ``None`` (default)
        lattice/grid adjacency is used.
    n_permutations : int
        Number of permutations to perferom to get a monte-carlo estimate of the
        null hypothesis distribution. More permutations result in more
        accurrate p values. Default is 1000.
    stat_threshold : float | None
        Cluster inclusion threshold in t value. Only data points exceeding this
        value of the t statistic (either ``t value > stat_threshold`` or
        ``t value < -stat_threshold``) form clusters. Default is ``None``,
        which means that cluster inclusion threshold is set according to
        ``alpha_threshold``. If both ``stat_threshold`` and ``alpha_threshold``
        are set, ``stat_threshold`` takes priority.
    alpha_threshold : float | None
        Cluster inclusion threshold in critical p value. Only data points where
        p value of the predictor effect lower than the critical value form
        clusters. Default is 0.05.
    cluster_pred : int
        Specify which predictor to use in clustering. Must be an integer: a
        zero-based index for the t values matrix returned by
        ``compute_regression_t``. Use values higher than zero - zero index
        indicates the intercept, which should be tested using a different
        permutation scheme than the one used here.
    backend : str
        Clustering backend. The default is 'numpy' but 'numba' can be also
        chosen. 'numba' should be faster for 3d clustering but requires the
        numba package.
    progressbar : bool
        Whether to report the progress of permutations using a progress bar.
        The default is ``True`` which uses tqdm progress bar.
    return_distribution : bool
        Whether to retrun the permutation distribution as an additional, fourth
        output argument.
    stat_fun : None | callable
        Function to compute regression. The function should take two arguments:
        data (data to predict) and preds (predictors to use) and return a
        matrix of regression parameters.

    Returns
    -------
    t_values : numpy array
        Statistical map of t values for the effect of predictor of interest.
    clusters : list of numpy arrays
        List of boolean numpy arrays. Consecutive arrays correspond to boolean
        cluster masks.
    cluster_p : numpy array
        Numpy array of cluster-level p values.
    distributions : dict
        Dictionary of positive null distribution (``distributions['pos']``) and
        negative null distribution (``distributions['neg']``). Returned only if
        ``return_distribution`` was set to ``True``.
    '''
    # data has to have observations as 1st dim and channels/vert as last dim
    # FIXME: add checks for input types
    preds = _handle_preds(preds)

    if stat_threshold is None:
        from scipy.stats import t
        df = data.shape[0] - 2  # in future: preds.shape[1]
        stat_threshold = t.ppf(1 - alpha_threshold / 2, df)

    if stat_fun is None:
        stat_fun = compute_regression_t

    use_3d_clustering = data.ndim > 3 and adjacency is not None

    n_obs = data.shape[0]
    if adjacency is not None and not use_3d_clustering:
        from mne.stats.cluster_level import _setup_connectivity
        adjacency = _setup_connectivity(adjacency, np.prod(data.shape[1:]),
                                        data.shape[1])

    pos_dist = np.zeros(n_permutations)
    neg_dist = np.zeros(n_permutations)
    perm_preds = preds.copy()

    if cluster_pred is None:
        cluster_pred = 1

    # regression on non-permuted data
    t_values = stat_fun(data, preds)[cluster_pred]

    if use_3d_clustering:
        # use 3d clustering
        cluster_fun = _get_cluster_fun(t_values, adjacency=adjacency,
                                       backend=backend)
        # we need to transpose dimensions for 3d clustering
        # FIXME/TODO - this could be eliminated by creating a single unified
        #              clustering function / API
        data_dims = np.array(list(range(data.ndim)))
        data_dims[1], data_dims[-1] = data_dims[-1], 1
        data = data.transpose(data_dims)
        t_values = t_values.transpose(data_dims[1:] - 1)
    else:
        backend = 'mne'
        cluster_fun = None

    clusters, cluster_stats = find_clusters(
        t_values, stat_threshold, adjacency=adjacency, cluster_fun=cluster_fun,
        backend=backend)

    if use_3d_clustering:
        t_values = t_values.transpose(data_dims[1:] - 1)

    if not clusters:
        print('No clusters found, permutations are not performed.')
        return t_values, clusters, cluster_stats
    else:
        msg = 'Found {} clusters, computing permutations.'
        print(msg.format(len(clusters)))

    # TODO - move progressbar code from DiamSar!
    #      - then support tqdm pbar as input
    #      - use autonotebook?
    if progressbar:
        from tqdm import tqdm
        pbar = tqdm(total=n_permutations)

    # compute permutations
    for perm in range(n_permutations):
        # permute predictors
        perm_inds = np.random.permutation(n_obs)
        perm_preds[:, cluster_pred - 1] = preds[perm_inds, cluster_pred - 1]
        perm_tvals = stat_fun(data, perm_preds)[cluster_pred]

        # cluster
        _, perm_cluster_stats = find_clusters(
            perm_tvals, stat_threshold, adjacency=adjacency,
            cluster_fun=cluster_fun, backend=backend,
            mne_reshape_clusters=False)

        # if any clusters were found - add max statistic
        if len(perm_cluster_stats) > 0:
            max_val = perm_cluster_stats.max()
            min_val = perm_cluster_stats.min()

            if max_val > 0:
                pos_dist[perm] = max_val
            if min_val < 0:
                neg_dist[perm] = min_val

        if progressbar:
            pbar.update(1)

    # compute permutation probability
    cluster_p = np.array([(pos_dist > cluster_stat).mean() if cluster_stat > 0
                          else (neg_dist < cluster_stat).mean()
                          for cluster_stat in cluster_stats])
    cluster_p *= 2  # because we use two-tail
    cluster_p[cluster_p > 1.] = 1.  # probability has to be <= 1.

    # sort clusters by p value
    cluster_order = np.argsort(cluster_p)
    cluster_p = cluster_p[cluster_order]
    clusters = [clusters[i] for i in cluster_order]

    if use_3d_clustering:
        clusters = [clst.transpose(data_dims[1:] - 1) for clst in clusters]

    out = t_values, clusters, cluster_p
    if return_distribution:
        distribution = dict(pos=pos_dist, neg=neg_dist)
        out += distribution

    return out
