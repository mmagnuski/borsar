import numpy as np
from scipy import sparse

from borsar.utils import find_range
from borsar.stats import compute_regression_t, format_pvalue
from borsar._viz3d import plot_cluster_src
from borsar.clusterutils import (_check_stc, _label_from_cluster, _get_clim,
                                 _prepare_cluster_description,
                                 _aggregate_cluster, _get_units)

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
    '''TODO: add DOCs!'''
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

    # TODO - move progressbar code from DiamSar
    #      - then support tqdm pbar as input
    #      - use autonotebook
    if progressbar:
        from tqdm import tqdm
        pbar = tqdm(total=n_permutations)

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


def read_cluster(fname, subjects_dir=None, src=None, info=None):
    '''
    Read standard Clusters .hdf5 file and return Clusters object.
    You need to pass correct subjects_dir and src to `read_cluster` if your
    results are in source space or correct info if your results are in channel
    space.

    Parameters
    ----------
    fname : str
        File path for the file to read.
    subjects_dir : str, optional
        Path to Freesurfer subjects directory.
    src : mne.SourceSpaces, optional
        Source space that the results are reprseneted in.
    info : mne.Info, optional
        Channel space that the results are respresented in.

    Returns
    -------
    clst : Clusters
        Cluster results read from file.
    '''
    from mne.externals import h5io
    # subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)
    data_dict = h5io.read_hdf5(fname)
    clst = Clusters(
        data_dict['clusters'], data_dict['pvals'], data_dict['stat'],
        dimnames=data_dict['dimnames'], dimcoords=data_dict['dimcoords'],
        subject=data_dict['subject'], subjects_dir=subjects_dir, info=info,
        src=src, description=data_dict['description'])
    return clst


class Clusters(object):
    '''
    Container for results of cluster-based tests.

    Parameters
    ----------
    clusters : list of boolean ndarrays | boolean ndarray
        List of boolean masks - one per cluster. The masks should match the
        dimensions of the `stat` ndarray. Each mask descirbes which elements
        are members of given cluster. Alternatively - one boolean array where
        first dimension corresponds to consevutive clusters.
    pvals : list or array of float
        List/array of p values corresponding to consecutive clusters in
        `clusters`.
    stat : ndarray
        Statistical map of the analysis. Usual dimensions are: space,
        space x time, space x frequencies, space x frequencies x
        time where space corresponds to channels or vertices (in the source
        space).
    dimnames : list of str, optional
        List of dimension names. For example `['chan', 'freq']` or `['vert',
        'time']`. The length of `dimnames` has to mach `stat.ndim`.
        If 'chan' dimname is given, you also need to provide `mne.Info`
        corresponding to the channels via info keyword argument.
        If 'vert' dimname is given, you also need to provide forward, subject
        and subjects_dir via respective keyword arguments.
    dimcoords : list of arrays, optional
        List of arrays, where each array contains coordinates (labels) for
        consecutive elements in corresponding dimension. For example if your
        `stat` represents channels by frequencies then a) `dimcoords[0]` should
        have length of `stat.shape[0]` and its consecutive elements should
        represent channel names while b) `dimcoords[1]` should have length of
        `stat.shape[1]` and its consecutive elements should represent centers
        of frequency bins (in Hz).
    info : mne.Info, optional
        When using channel space ('chan' is one of the dimnames) you need to
        provide information about channel position in mne.Info file (for
        example `epochs.info`).
    src : mne.SourceSpaces, optional
        When using source space ('vert' is one of the dimnames) you need to
        pass mne.SourceSpaces
    subject : str, optional
        When using source space ('vert' is one of the dimnames) you need to
        pass a subject name (name of the freesurfer directory with file for
        given subject).
    subjects_dir : str, optional
        When using source space ('vert' is one of the dimnames) you need to
        pass a Freesurfer subjects directory (path to the folder contraining
        subjects as subfolders).
    description : str | dict, optional
        Optional description of the Clusters - for example analysis parameters
        and some other additional details.
    sort_pvals : bool
        Whether to sort clusters by their p-value (ascending). Default: True.
    '''
    def __init__(self, clusters, pvals, stat, dimnames=None, dimcoords=None,
                 info=None, src=None, subject=None, subjects_dir=None,
                 description=None, sort_pvals=True, safety_checks=True):
        # safety checks
        if safety_checks:
            clusters = _clusters_safety_checks(clusters, pvals, stat, dimnames,
                                               dimcoords, description)

            # check channel or source space
            _clusters_chan_vert_checks(dimnames, info, src, subject,
                                       subjects_dir)

            # check polarity of clusters
            polarity = ['neg', 'pos']
            self.cluster_polarity = [polarity[int(stat[cl].mean() > 0)]
                                     for cl in clusters]

        # sort by p values if necessary
        pvals = np.asarray(pvals)
        if sort_pvals:
            psort = np.argsort(pvals)
            if not (psort == np.arange(pvals.shape[0])).all():
                clusters = clusters[psort]
                pvals = pvals[psort]

        # create attributes
        self.subjects_dir = subjects_dir
        self.description = description
        self.dimcoords = dimcoords
        self.clusters = clusters
        self.dimnames = dimnames
        self.subject = subject
        self.pvals = pvals
        self.stat = stat
        self.info = info
        self.stc = None
        self.src = src

    # - [ ] add warning if all clusters removed
    # - [ ] consider select to _not_ work inplace
    def select(self, p_threshold=None, percentage_in=None, n_points_in=None,
               n_points=None, **kwargs):
        '''
        Select clusters by p value threshold or its location in the data space.

        TODO: fix docs
        '''
        if self.clusters is None:
            return self

        # select clusters by p value threshold
        if p_threshold is not None:
            sel = self.pvals < p_threshold
            self = _cluster_selection(self, sel)

        if (len(kwargs) > 0 or n_points_in is not None) and len(self) > 0:
            # kwargs check should be in separate function
            if len(kwargs) > 0:
                _check_dimnames_kwargs(self, **kwargs)
            dim_idx = _index_from_dim(self.dimnames, self.dimcoords, **kwargs)

            dims = np.arange(self.stat.ndim) + 1
            clst_idx = (slice(None),) + dim_idx
            cluster_sel_size = self.clusters[clst_idx].sum(axis=tuple(dims))

            sel = np.ones(len(self), dtype='bool')
            if n_points_in is not None:
                sel = cluster_sel_size >= n_points_in
            if percentage_in is not None:
                cluster_size = self.clusters.sum(axis=tuple(dims))
                sel = ((cluster_sel_size / cluster_size >= percentage_in)
                       & sel)
            self = _cluster_selection(self, sel)
        return self

    # TODO: add deepcopy arg?
    def copy(self):
        '''
        Copy the Clusters object. The lists/arrays are not copied however.
        The SourceSpaces are always copied because they often change when
        plotting.

        Returns
        -------
        clst : Clusters
            Copied Clusters object.
        '''
        clst = Clusters(self.clusters, self.pvals, self.stat, self.dimnames,
                        self.dimcoords, info=self.info, src=self.src,
                        subject=self.subject, subjects_dir=self.subjects_dir,
                        description=self.description,
                        safety_checks=False, sort_pvals=False)
        clst.stc = self.stc if self.stc is None else stc.copy()
        clst.cluster_polarity = self.cluster_polarity
        return clst

    def __len__(self):
        '''Return number of clusters in Clusters.'''
        return len(self.clusters) if self.clusters is not None else 0

    def __iter__(self):
        '''Initialize iteration.'''
        self._current = 0
        return self

    def __next__(self):
        '''
        Get next cluster in iteration. Allows to do things like:
        >>> for clst in clusters:
        >>>     clst.plot()
        '''
        if self._current >= len(self.clusters):
            raise StopIteration
        clst = Clusters(self.clusters[self._current],
                        self.pvals[[self._current]], self.stat, self.dimnames,
                        self.dimcoords, info=self.info, src=self.src,
                        subject=self.subject, subjects_dir=self.subjects_dir,
                        description=self.description,
                        safety_checks=False, sort_pvals=False)
        clst.stc = self.stc # or .copy()?
        clst.cluster_polarity = [self.cluster_polarity[self._current]]
        self._current += 1
        return clst

    def save(self, fname, description=None):
        '''
        Save Clusters to hdf5 file.

        Parameters
        ----------
        fname : str
            Path to save the file in.
        description : str, dict
            Additional description added when saving. When passed overrides
            the description parameter of Clusters.
        '''
        from mne.externals import h5io

        if description is None:
            description = self.description
        else:
            _check_description(description)

        data_dict = {'clusters': self.clusters, 'pvals': self.pvals,
                     'stat': self.stat, 'dimnames': self.dimnames,
                     'dimcoords': self.dimcoords, 'subject': self.subject,
                     'description': description}
        h5io.write_hdf5(fname, data_dict)

    # TODO - consider weighting contribution by stat value
    #      - consider contributions along two dimensions
    def get_contribution(self, cluster_idx=None, along=None, norm=True):
        '''
        Get mass percentage contribution to given clusters along specified
        dimension.

        Parameters
        ----------
        cluster_idx : int | array of int, optional
            Indices of clusters to get contribution of. Default is to calculate
            contribution for all clusters.
        along : int | str, optional
            Dimension along which the clusters contribution should be
            calculated. Default is to calculate along the first dimension.
        norm : bool, optional
            Whether to normalize contributions. Defaults to `True`.

        Returns
        -------
        contrib : array
            One-dimensional array containing float values (percentage
            contributions) if `norm=True` or integer values (number of
            elements) if `norm=False`.
        '''
        if cluster_idx is None:
            cluster_idx = np.arange(len(self))

        along = 0 if along is None else along
        idx = _check_dimname_arg(self, along)

        if isinstance(cluster_idx, (list, np.ndarray)):
            alldims = list(range(self.stat.ndim + 1))
            alldims.remove(0)
            alldims.remove(idx + 1)
            contrib = self.clusters[cluster_idx].sum(axis=tuple(alldims))
            if norm:
                contrib = contrib / contrib.sum(axis=-1, keepdims=True)
        else:
            alldims = list(range(self.stat.ndim))
            alldims.remove(idx)
            contrib = self.clusters[cluster_idx].sum(axis=tuple(alldims))
            if norm:
                contrib = contrib / contrib.sum()
        return contrib

    # TODO: consider continuous vs discontinuous limits
    def get_cluster_limits(self, cluster_idx, retain_mass=0.65,
                           ignore_space=True, check_dims=None, **kwargs):
        '''TODO: add docs'''
        # TODO: add safety checks
        if check_dims is None:
            check_dims = list(range(self.stat.ndim))
        if ignore_space and 0 in check_dims:
            check_dims.remove(0)

        limits = list()
        for idx in range(self.stat.ndim):
            if idx in check_dims:
                dimname = self.dimnames[idx]
                mass = kwargs[dimname] if dimname in kwargs else retain_mass
                contrib = self.get_contribution(cluster_idx, along=dimname)

                # curent method - start at max and extend
                limits.append(_get_mass_range(contrib, mass))
            else:
                limits.append(slice(None))
        return tuple(limits)

    def get_index(self, cluster_idx=None, retain_mass=None, ignore_space=True,
                  **kwargs):
        '''
        Get indices selecting a specified range of data.

        TODO: fix docs
        retain_mass:
        exclude (_contributions): for each dimension include only those elements that
            surpass relative mass (that contribute above certain percentage of mass
            along that dimension)
        '''
        # TODO - reduce this check to existing function
        if len(kwargs) > 0 and self.dimnames is None:
            raise TypeError('To be able to find indexing by dimension names you must'
                            ' create Clusters passing dimnames (and preferrably '
                            'also dimcoords).')
        dims = list([None] * self.stat.ndim)
        dimval = list([None] * self.stat.ndim)
        normal_indexing = kwargs.copy()
        mass_indexing = dict()
        if len(kwargs) > 0:
            for dimname in kwargs.keys():
                dim_idx = self.dimnames.index(dimname)
                dval = kwargs[dimname]
                # TODO - more elaborate checks
                dims[dim_idx] = 'range' if isinstance(dval, list) else 'mass'
                dimval[dim_idx] = dval
                if dims[dim_idx] is 'mass':
                    mass_indexing[dimname] = dval
                    normal_indexing.pop(dimname)

        if self.dimcoords is None:
            idx = tuple(slice(None) for _ in self.stat.shape)
        else:
            idx = _index_from_dim(self.dimnames, self.dimcoords,
                                  **normal_indexing)

        if cluster_idx is not None:
            check_dims = [idx for idx, val in enumerate(idx)
                          if val == slice(None)]
            # check cluster limits only if some dim limits were not specified
            if len(check_dims) > 0:
                idx_mass = self.get_cluster_limits(
                    cluster_idx, ignore_space=ignore_space, **mass_indexing)
                idx = tuple([idx_mass[i] if i in check_dims else idx[i]
                             for i in range(len(idx))])
        return idx

    # maybe rename to `plot mass`?
    def plot_contribution(self, dimname, axis=None):
        '''Plot how different times or frequencies contribute to clusters.'''
        return plot_cluster_contribution(self, dimname, axis=axis)

    def plot(self, cluster_idx=None, aggregate='mean', set_light=True,
             vmin=None, vmax=None, **kwargs):
        '''
        Plot cluster.
        TODO: fix docs - will be copied from function?
        '''
        if self.dimnames is None:
            raise TypeError('To plot the data you need to construct the '
                             'cluster using the dimnames keyword argument.')
        if self.dimnames[0] == 'vert':
            return plot_cluster_src(self, cluster_idx, vmin=vmin, vmax=vmax,
                                    aggregate=aggregate, set_light=set_light,
                                    **kwargs)
        elif self.dimnames[0] == 'chan':
            return plot_cluster_chan(self, cluster_idx, vmin=None, vmax=None,
                                     aggregate=aggregate, **kwargs)


# TODO - add special case for dimension='vert' and 'chan'
def plot_cluster_contribution(clst, dimension, picks=None, axis=None):
    '''
    Plot contribution of clusters along specified dimension.
    TODO - fix docs
    '''
    import matplotlib.pyplot as plt

    # check dimname and return index
    idx = _check_dimname_arg(clst, dimension)

    # check if any clusters
    n_clusters = len(clst)
    if n_clusters == 0:
        raise ValueError('No clusters present in Clusters object.')

    picks = list(range(n_clusters)) if picks is None else picks

    # create freq coords and label
    if clst.dimcoords is None:
        dimcoords = np.arange(clst.stat.shape[idx])
        dimlabel = '{} bins'.format(_get_full_dimname(dimension))
    else:
        dimcoords = clst.dimcoords[idx]
        dimlabel = '{} ({})'.format(_get_full_dimname(dimension),
                                    _get_units(dimension))

    # make sure we have an axes to plot to
    if axis is None:
        axis = plt.axes()

    # plot cluster contribution
    for idx in picks:
        label = 'idx={}, p={:.3f}'.format(idx, clst.pvals[idx])
        contrib = clst.get_contribution(idx, along=dimension, norm=False)
        axis.plot(dimcoords, contrib, label=label)

    axis.legend(loc='best')
    axis.set_xlabel(dimlabel)

    # TODO - reduced dimnames could be: channel-frequency bins
    elements_name = ({'vert': 'vertices', 'chan': 'channels'}
                      [clst.dimnames[0]] if clst.dimcoords is not None
                     else 'elements')
    axis.set_ylabel('Number of ' + elements_name)
    return axis


def plot_cluster_chan(clst, cluster_idx=None, aggregate='mean', vmin=None,
                      vmax=None, **kwargs):
    '''
    Plot cluster in source space.

    Parameters
    ----------
    clst : Clusters
        Clusters object to use in plotting.
    cluster_idx : int
        Cluster index to plot.
    aggregate : str
        TODO: mean, max, weighted
    vmin : float, optional
        Value mapped to minimum in the colormap. Inferred from data by default.
    vmax : float, optional
        Value mapped to maximum in the colormap. Inferred from data by default.
    title : str, optional
        Optional title for the figure.
    **kwargs : additional keyword arguments
        Additional arguments used in aggregation, defining the range to
        aggregate for given dimension. List of two values defines explicit
        range: for example keyword argument `freq=[6, 8]` aggregates the
        6 - 8 Hz range. Float argument between 0. and 1. defines range that is
        dependent on cluster mass. For example `time=0.75` defines time range
        that retains at least 75% of the cluster (calculated along the
        aggregated dimension - in this case time). If no kwarg is passed for
        given dimension then the default value is 0.65 so that range is
        defined to retain at least 65% of the cluster mass.

    Returns
    -------
    topo : mypy.viz.Topo class
        Figure object used in plotting.

    Examples
    --------
    > # to plot the first cluster within 8 - 10 Hz
    > clst.plot(cluster_idx=0, freq=[8, 10])
    > # to plot the second cluster selecting frequencies that make up at least
    > # 70% of the cluster mass:
    > clst.plot(cluster_idx=1, freq=0.7)
    '''
    cluster_idx = 0 if cluster_idx is None else cluster_idx

    # get and aggregate cluster mask and cluster stat
    clst_mask, clst_stat, idx = _aggregate_cluster(
        clst, cluster_idx, mask_proportion=0.5, retain_mass=0.65,
        ignore_space=True, **kwargs)

    # create pysurfer brain
    from mypy.viz import Topo
    vmin, vmax = _get_clim(clst_stat, vmin=vmin, vmax=vmax, pysurfer=False)
    topo = Topo(clst_stat, clst.info, vmin=vmin, vmax=vmax, show=False)
    topo.solid_lines()
    topo.mark_channels(clst_mask)

    return topo


# - [ ] add special functions for handling dims like vert or chan
def _index_from_dim(dimnames, dimcoords, **kwargs):
    '''
    Find axis slices given dimnames, dimaxes and dimname keyword arguments
    with list of `[start, end]` each.

    Parameters
    ----------
    dimnames : list of str
        List of dimension names. For example `['chan', 'freq']` or `['vert',
        'time']`. The length of `dimnames` has to mach `stat.ndim`.
    dimcoords : list of arrays
        List of arrays, where each array contains coordinates (labels) for
        consecutive elements in corresponding dimension.
    **kwargs : additional keywords
        Keywords referring to dimension names and values each being a list of two
        values representing lower and upper limits of dimension selection.

    Returns
    -------
    idx : tuple of slices
        Tuple that can be used to index the data array: `data[idx]`.

    Examples
    --------
    >>> dimnames = ['freq', 'time']
    >>> dimcoords = [np.arange(8, 12), np.arange(-0.2, 0.6, step=0.05)]
    >>> _index_from_dim(dimnames, dimcoords, freq=[9, 11], time=[0.25, 0.5])
    (slice(1, 4, None), slice(9, 15, None))
    '''

    idx = list()
    for dname, dcoord in zip(dimnames, dimcoords):
        if dname not in kwargs:
            idx.append(slice(None))
            continue
        sel_ax = kwargs.pop(dname)
        idx.append(find_range(dcoord, sel_ax))
    return tuple(idx)


# CHECKS
# ------
def _clusters_safety_checks(clusters, pvals, stat, dimnames, dimcoords,
                            description):
    '''Perform basic type and safety checks for Clusters.'''
    # check clusters when it is a list
    if isinstance(clusters, list):
        n_clusters = len(clusters)
        cluster_shapes = [clst.shape for clst in clusters]
        all_shapes_equal = all([clst_shp == cluster_shapes[0]
                                for clst_shp in cluster_shapes])
        if not all_shapes_equal:
            raise ValueError('All clusters have to be of the same '
                             'shape.')
        all_arrays_bool = all([clst.dtype == 'bool' for clst in clusters])
        if not all_arrays_bool:
            raise TypeError('All clusters have to be boolean arrays.')

        clusters = np.stack(clusters, axis=0)

    # check stat
    if not isinstance(stat, np.ndarray):
        raise TypeError('`stat` must be a numpy array.')

    # check clusters shape along stat shape
    if isinstance(clusters, (list, np.ndarray)):
        if not stat.shape == clusters.shape[1:]:
            raise ValueError('Every cluster has to have the same shape as '
                             'stat.')
    else:
        raise TypeError('clusters have to be either a list of arrays or one '
                        'array with first dimension corresponding to '
                        'clusters.')

    if not isinstance(pvals, (list, np.ndarray)):
        raise TypeError('`pvals` has to be a list of floats or numpy array.')
        # check if each element of list is float and array is of dtype float

    if dimnames is not None:
        if not isinstance(dimnames, list):
            raise TypeError('`dimnames` must be a list of dimension names.'
                             'Got {}.'.format(type(dimnames)))
        which_str = np.array([isinstance(el, str) for el in dimnames])
        if not which_str.all():
            other_type = type(dimnames[np.where(~which_str)[0][0]])
            raise TypeError('`dimnames` must be a list od strings, but some '
                            'of the elements in the list you passed are not '
                            'strings, for example: {}'.format(other_type))
        if not len(dimnames) == stat.ndim:
            raise ValueError('Length of `dimnames` must be the same as number'
                             ' of dimensions in `stat`.')
        if ('chan' in dimnames and not dimnames.index('chan') == 0 or
            'vert' in dimnames and not dimnames.index('vert') == 0):
            msg = ('If using channels ("chan" dimension name) or vertices ('
                   'for source space - "vert" dimension name) - it must be '
                   'the first dimension in the `stat` array and therefore the'
                   ' first dimension name in `dimnames`.')
            raise ValueError(msg)
    if dimcoords is not None:
        if not isinstance(dimcoords, list):
            raise TypeError('`dimcoords` must be a list of dimension '
                             'coordinates. Got {}.'.format(type(dimcoords)))

        dims = list(range(len(dimcoords)))
        if dimnames[0] in ['chan', 'vert']:
            dims.pop(0)
        equal_len = [stat.shape[idx] == len(dimcoords[idx]) for idx in dims]
        if not all(equal_len):
            nonequal_len = np.where(equal_len)[0]
            msg = ('The length of each dimension coordinate (except for the '
                   'spatial dimension - channels or vertices) has to be the '
                   'same as the length of the corresponding dimension in '
                   '`stat` array.')
            raise ValueError(msg)
    _check_description(description)
    return clusters


def _check_description(description):
    '''Validate if description is of correct type.'''
    if description is not None:
        if not isinstance(description, (str, dict)):
            raise TypeError('Description has to be either a string or a dict'
                            'ionary, got {}.'.format(type(description)))


def _clusters_chan_vert_checks(dimnames, info, src, subject, subjects_dir):
    '''Safety checks for Clusters spatial dimension.'''
    import mne
    if dimnames is not None and 'chan' in dimnames:
        if info is None or not isinstance(info, mne.Info):
            raise TypeError('You must pass an `mne.Info` in order to use '
                            '"chan" dimension. Use `info` keyword argument.')
    elif dimnames is not None and 'vert' in dimnames:
        if src is None or not isinstance(src, mne.SourceSpaces):
            raise TypeError('You must pass an `mne.SourceSpaces` in order to '
                            'use "vert" dimension. Use `src` keyword'
                            ' argument.')
        if subject is None or not isinstance(subject, str):
            raise TypeError('You must pass a subject string in order to '
                            'use "vert" dimension. Use `subject` keyword'
                            ' argument.')
        if subjects_dir is None:
            subjects_dir = mne.utils.get_subjects_dir()
        if subjects_dir is None or not isinstance(subjects_dir, str):
            raise TypeError('You must pass a `subjects_dir` freesurfer folder'
                            ' name in order to use "vert" dimension. Use '
                            '`subjects_dir` keyword argument.')


def _check_dimname_arg(clst, dimname):
    '''Check dimension name and find its index.'''
    if not isinstance(dimname, (str, int)):
        raise TypeError('Dimension argument has to be string (dimension name) '
                        'or int (dimension index).')
    if isinstance(dimname, str):
        if clst.dimnames is None:
            raise TypeError('Clusters has to have `dimnames` attribute to use '
                            'operations on named dimensions.')
        if dimname not in clst.dimnames:
            raise ValueError('Clusters does not seem to have the dimension you'
                            ' requested. You asked for "{}", while Clusters has '
                            'the following dimensions: {}.'.format(
                                dimname, ', '.join(clst.dimnames)))
        idx = clst.dimnames.index(dimname)
    else:
        if not (dimname >= 0 and dimname < clst.stat.ndim):
            raise ValueError('Dimension, if integer, must be greater or equal '
                             'to 0 and lower than number of dimensions of the '
                             'statistical map. Got {}'.format(dimname))
        idx = dimname
    return idx


def _check_dimnames_kwargs(clst, **kwargs):
    '''Ensure that **kwargs are correct dimnames and dimcoords.'''
    if clst.dimnames is None:
        raise TypeError('Clusters has to have dimnames to use operations '
                        'on named dimensions.')

    for dim in kwargs.keys():
        if dim not in clst.dimnames:
            msg = ('Could not find requested dimension {}. Available '
                   'dimensions: {}.'.format(along, ', '.join(self.dimnames)))
            raise ValueError(msg)


# UTILS
# -----
def _get_full_dimname(dimname):
    '''Return full dimension name.'''
    dct = {'freq': 'frequency', 'time': 'time', 'vert': 'vertices',
           'chan': 'channels'}
    return dct[dimname] if dimname in dct else dimname


def _get_mass_range(contrib, mass):
    '''Find range that retains given mass (sum) of the contributions vector.'''
    contrib_len = contrib.shape[0]
    max_idx = np.argmax(contrib)
    current_mass = contrib[max_idx]
    side_idx = np.array([max_idx, max_idx])
    while current_mass < mass:
        side_idx += [-1, +1]
        vals = [0. if side_idx[0] < 0 else contrib[side_idx[0]],
                0. if side_idx[1] + 1 >= contrib.shape[0]
                else contrib[side_idx[1]]]

        if sum(vals) == 0.:
            side_idx += [+1, -1]
            break
        ord = np.argmax(vals)
        current_mass += contrib[side_idx[ord]]
        one_back = [+1, -1]
        one_back[ord] = 0
        side_idx += one_back
    return slice(side_idx[0], side_idx[1] + 1)


def _cluster_selection(clst, sel):
    '''Select Clusters according to selection vector `sel`'''
    if sel.all():
        return clst
    if sel.dtype == 'bool':
        sel = np.where(sel)[0]

    if len(sel) > 0:
        clst.cluster_polarity = [clst.cluster_polarity[idx]
                                 for idx in sel]
        clst.clusters = clst.clusters[sel]
        clst.pvals = clst.pvals[sel]
    else:
        clst.cluster_polarity = []
        clst.clusters = None
        clst.pvals = None
    return clst
