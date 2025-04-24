import numpy as np

from .label import (_check_backend, _get_cluster_fun, _prepare_clustering,
                    find_clusters)
from ..stats import (compute_regression_t, _handle_preds, _find_stat_fun,
                     _compute_threshold, _compute_threshold_via_permutations)
from ..utils import progressbar


def _set_up_clustering(stat, adjacency=None, backend='auto', min_adj_ch=0):
    '''Helper function that configures clustering function.'''
    backend = _check_backend(stat, adjacency, backend, min_adj_ch)
    cluster_fun = _get_cluster_fun(
        stat, adjacency=adjacency, backend=backend, min_adj_ch=min_adj_ch)
    find_func, adjacency, add_arg = _prepare_clustering(
        stat, adjacency, cluster_fun, backend, min_adj_ch=min_adj_ch)
    return find_func, adjacency, add_arg


def _check_if_any_clusters(clusters):
    if not clusters:
        print('No clusters found, permutations are not performed.')
        if_continue = False
    else:
        msg = 'Found {} clusters, computing permutations.'
        print(msg.format(len(clusters)))
        if_continue = True
    return if_continue


def _update_null(perm_stats, perm_idx, tail='both', pos_dist=None,
                 neg_dist=None):
    if len(perm_stats) > 0:
        if tail in ['both', 'pos']:
            max_val = perm_stats.max()
            if max_val > 0.:
                pos_dist[perm_idx] = max_val

        if tail in ['both', 'neg']:
            min_val = perm_stats.min()
            if min_val < 0.:
                neg_dist[perm_idx] = min_val


def _compare_to_null(clusters, cluster_stats, tail='both', pos_dist=None,
                     neg_dist=None):
    # ignore values that are not in the tail of interest
    if tail == 'pos':
        mask = cluster_stats >= 0
    elif tail == 'neg':
        mask = cluster_stats <= 0
    else:
        mask = None

    if mask is not None:
        cluster_stats = cluster_stats[mask]
        clusters = [clusters[ix] for ix in np.where(mask)[0]]

    cluster_p = np.array([
        (pos_dist >= cluster_stat).mean() if cluster_stat > 0
        else (neg_dist <= cluster_stat).mean()
        for cluster_stat in cluster_stats]
    )

    if tail == 'both':
        cluster_p *= 2  # because we use two-tail
        cluster_p[cluster_p > 1.] = 1.  # probability has to be <= 1.

    # sort clusters by p value
    cluster_order = np.argsort(cluster_p)
    cluster_p = cluster_p[cluster_order]
    clusters = [clusters[i] for i in cluster_order]
    cluster_stats = cluster_stats[cluster_order]

    return clusters, cluster_stats, cluster_p


# - [x] add min_adj_ch parameter passed to find_clusters
# - [ ] add `verbose` parameter
# - [ ] FIXME: add checks for input types
# - [ ] CONSIDER: cluster_pred always addressing preds (you never want
#                 cluster the intercept, and if you do you'd need a one sample
#                 t test and thus a different permutation scheme)
def cluster_based_regression(data, preds, adjacency=None, n_permutations=1000,
                             stat_threshold=None, alpha_threshold=0.05,
                             cluster_pred=None, backend='auto',
                             progressbar=True, return_distribution=False,
                             stat_fun=None, min_adj_ch=0):
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
        Number of permutations to perform to get a monte-carlo estimate of the
        null hypothesis distribution. More permutations result in more
        accurate p values. Default is 1000.
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
        Whether to return the permutation distribution as an additional, fourth
        output argument.
    stat_fun : None | callable
        Function to compute regression. The function should take two arguments:
        ``data`` (data to predict) and ``preds`` (predictors to use) and return
        a matrix of regression parameters.

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
    preds = _handle_preds(preds)
    n_obs = data.shape[0]

    if stat_threshold is None:
        from scipy.stats import t
        df = n_obs - 2
        stat_threshold = t.ppf(1 - alpha_threshold / 2, df)

    if stat_fun is None:
        stat_fun = compute_regression_t

    if cluster_pred is None:
        cluster_pred = 1

    # regression on non-permuted data
    t_values = stat_fun(data, preds)[cluster_pred]

    # set up clustering
    find_func, adjacency, add_arg = _set_up_clustering(
        t_values, adjacency=adjacency, backend=backend, min_adj_ch=min_adj_ch)

    clusters, cluster_stats = find_func(
        t_values, stat_threshold, adjacency, add_arg, min_adj_ch=min_adj_ch,
        full=True)

    if_continue = _check_if_any_clusters(clusters)
    if not if_continue:
        return t_values, clusters, cluster_stats

    pos_dist = np.zeros(n_permutations)
    neg_dist = np.zeros(n_permutations)
    perm_preds = preds.copy()

    # TODO - move progressbar code from DiamSar!
    #      - then support tqdm pbar as input
    if progressbar:
        from tqdm import tqdm
        pbar = tqdm(total=n_permutations)

    # compute permutations
    for perm in range(n_permutations):
        # permute predictors
        perm_inds = np.random.permutation(n_obs)
        perm_preds[:, cluster_pred] = preds[perm_inds, cluster_pred]
        perm_tvals = stat_fun(data, perm_preds)[cluster_pred]

        # cluster
        _, perm_cluster_stats = find_func(
            perm_tvals, stat_threshold, adjacency, add_arg,
            min_adj_ch=min_adj_ch, full=True)

        # if any clusters were found - add max statistic
        _update_null(perm_cluster_stats, perm, tail='both', pos_dist=pos_dist,
                     neg_dist=neg_dist)

        if progressbar:
            pbar.update(1)

    # compute permutation probability
    clusters, _, cluster_p = _compare_to_null(
        clusters, cluster_stats, tail='both', pos_dist=pos_dist,
        neg_dist=neg_dist)

    out = t_values, clusters, cluster_p
    if return_distribution:
        distribution = dict(pos=pos_dist, neg=neg_dist)
        out += (distribution,)

    return out


# TODO: add condition order argument? This may require a large refactoring of
#       the function to allow for 2-step tests (step 1 - within subjects,
#       step 2 - across subjects)
# TODO: move `min_adj_ch` up and add `min_adj`
def permutation_cluster_test_array(data, adjacency, stat_fun=None,
                                   threshold=None, p_threshold=0.05,
                                   paired=False, one_sample=False, tail='both',
                                   n_permutations=1000, n_stat_permutations=0,
                                   progress=True, return_distribution=False,
                                   backend='auto', min_adj_ch=0):
    """Permutation cluster test on array data.

    Parameters
    ----------
    data : np.ndarray | list of np.ndarray
        An array where first two dimensions are ``conditions x observations``
        or list of arrays where each array has observations in the first
        dimension. If the data contains channels it should be in the dimension
        immediately after observations.
    adjacency : 2d boolean array | None
        Array that denotes adjacency between channels (or vertices). If
        ``None`` it is assumed that no channels/vertices are present.
    stat_fun : function | None
        Statistical function to use. It should allow as many arguments as
        conditions and should return one array of computed statistics.
    threshold : float | None
        Cluster entry threshold for the test statistic. If ``None`` (default)
        the ``p_threshold`` argument is used.
    p_threshold : float
        P value threshold to use in cluster entry threshold computation. For
        standard parametric tests (t test, ANOVA) it is computed from
        theoretical test distribution; if ``n_stat_permutations`` is above zero
        the threshold is obtained from percentile of permutation distribution.
    paired : bool
        Whether the permutations should be conducted for paired samples
        scenario (randomization of condition orders within observations).
        Currently the condition orders are randomized even if they are the same
        for all subjects. This argument is also used to automatically pick
        a statistical test if ``stat_fun`` is ``None``.
    one_sample : bool
        Whether the permutations should be conducted for a one sample scenario
        (sign flipping randomization). This argument is also used to
        automatically pick a statistical test if ``stat_fun`` is ``None``.
    tail : str
        Which differences to test. ``'both'`` tests positive and negative
        effects, while ``'pos'`` - only positive.
        NEG is not implemented!
    n_permutations : int
        Number of cluster based permutations to perform. Defaults to ``1000``.
    n_stat_permutations : int
        Whether to compute ``threshold`` using permutations (this is separate
        from cluster-based permutations when the computed thresholds are used).
        If ``n_stat_permutations > 0`` then this many permutations will be used
        to compute statistical cluster-entry thresholds. The threshold is set
        to ``p_threshold`` of the computed permutation distribution.
    progress : bool | str | tqdm progressbar
        Whether to show a progressbar (if boolean) or what kind of progressbar
        to show (``'notebook'`` or ``'text'``). Alternatively a progressbar can
        be passed that will be reset and set to a new maximum.
    return_distribution : bool
        Whether to return the distribution of cluster-based permutations.
        If ``True`` a dictionary of positive and negative cluster statistics
        from permutations is returned.
    backend : str
        Clustering backend to use. Can be ``'auto'``, ``'mne'``, ``'borsar'``
        or ``'numpy'``. Depending on the search space, different backend may be
        optimal. Defaults to ``'auto'`` which selects the backend
        automatically.
    min_adj_ch: int
        Minimum number of adjacent in-cluster channels to retain a point in
        the cluster.

    Returns
    -------
    stat : np.ndarray
        Statistical test results in the search space (same dimensions as
        ``data[0]``, apart from the first one representing observations).
    clusters : list of np.ndarray
        List of clusters. Each cluster is a boolean array of cluster
        membership.
    cluster_p : np.ndarray
        P values for each cluster.
    distribution : dict | None
        Dictionary of cluster statistics from permutations. Only returned if
        ``return_distribution`` is ``True``.
    """

    n_groups = len(data)
    if stat_fun is None:
        stat_fun = _find_stat_fun(n_groups, paired, tail)

    if paired or one_sample:
        n_obs = data[0].shape[0]
        signs_size = tuple([n_obs] + [1] * (data[0].ndim - 1))
    else:
        condition = np.concatenate([np.ones(data[idx].shape[0]) * idx
                                    for idx in range(n_groups)])
        data_unr = np.concatenate(data)

    if one_sample:
        signs = np.array([-1, 1])

    pos_dist = np.zeros(n_permutations)
    if tail == 'both':
        neg_dist = np.zeros(n_permutations)
    else:
        neg_dist = None

    # test on non-permuted data
    stat = stat_fun(*data)

    # compute threshold from stat, use permutation distribution if
    # n_stat_permutations > 0
    if n_stat_permutations > 0:
        threshold = _compute_threshold_via_permutations(
            data, paired, tail, stat_fun, p_threshold, n_stat_permutations,
            progress=progress)
    else:
        threshold = _compute_threshold(data, threshold, p_threshold,
                                       paired, one_sample)

    # use 3d clustering
    find_func, adjacency, add_arg = _set_up_clustering(
        stat, adjacency=adjacency, backend=backend, min_adj_ch=min_adj_ch)

    clusters, cluster_stats = find_func(
        stat, threshold, adjacency, add_arg, min_adj_ch=min_adj_ch,
        full=True)

    if_continue = _check_if_any_clusters(clusters)
    if not if_continue:
        return stat, clusters, cluster_stats

    if paired and n_groups > 2:
        orders = [np.arange(n_groups)]
        for _ in range(n_groups - 1):
            orders.append(np.roll(orders[-1], shift=-1))
        data_all = np.stack(data, axis=0)

    pbar = progressbar(progress, total=n_permutations)

    # compute permutations
    for perm in range(n_permutations):
        # permute data / predictors
        if one_sample:
            # one-sample sign-flip
            idx = np.random.random_integers(0, 1, size=signs_size)
            perm_signs = signs[idx]
            perm_data = [data[0] * perm_signs]
        elif paired and n_groups == 2:
            # this is analogous to one-sample sign-flip but with paired data
            # (we could also perform one sample t test on condition differences
            #  with sign-flip in the permutation step)
            idx1 = np.random.random_integers(0, 1, size=signs_size)
            idx2 = 1 - idx1
            perm_data = list()
            perm_data.append(data[0] * idx1 + data[1] * idx2)
            perm_data.append(data[0] * idx2 + data[1] * idx1)
        elif paired and n_groups > 2:
            ord_idx = np.random.randint(0, n_groups, size=n_obs)
            perm_data = data_all.copy()
            for obs_idx in range(n_obs):
                this_order = orders[ord_idx[obs_idx]]
                perm_data[:, obs_idx] = data_all[this_order, obs_idx]
        elif not paired:
            this_order = condition.copy()
            np.random.shuffle(this_order)
            perm_data = [data_unr[this_order == idx]
                         for idx in range(n_groups)]

        perm_stat = stat_fun(*perm_data)

        _, perm_cluster_stats = find_func(
            perm_stat, threshold, adjacency, add_arg, min_adj_ch=min_adj_ch,
            full=True)

        # if any clusters were found - add max statistic
        _update_null(perm_cluster_stats, perm, tail, pos_dist, neg_dist)

        if progressbar:
            pbar.update(1)

    # compute permutation probability
    clusters, _, cluster_p = _compare_to_null(
        clusters, cluster_stats, tail=tail, pos_dist=pos_dist,
        neg_dist=neg_dist)

    if return_distribution:
        return stat, clusters, cluster_p, dict(pos=pos_dist, neg=neg_dist)
    else:
        return stat, clusters, cluster_p
