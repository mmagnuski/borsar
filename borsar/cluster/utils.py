import numpy as np
from scipy import sparse

from ..utils import group_mask, find_range, find_index
from ..stats import format_pvalue
from ..channels import find_channels


# FIXME - currently the reading of neighbours seems to be a bit
#         messy. They should be read into the same format, either
#         FieldTrip struct (for a starter) or a new class Adjacency
#         with sparse matrix and channel/vert names
#         (this adjacency could hold additional adjacency options in
#          future that govern how clustering is done (?))
def construct_adjacency_matrix(neighbours, ch_names=None, as_sparse=False):
    '''
    Construct adjacency matrix out of neighbours structure (fieldtrip format).

    Parameters
    ----------
    neighbours : structured array | dict
        FieldTrip neighbours structure represented either as numpy structured
        array or dictionary. Needs to contain ``'label'`` and ``'neighblabel'``
        fields / keys.
    ch_names : list of str, optional
        List of channel names to use. Defaults to ``None`` which uses all
        channel names in ``neighbours`` in the same order as they appear in
        ``neighbours['label']``.
    as_sparse : bool
        Whether to return the adjacency matrix in sparse format. Defaults to
        ``False`` which returns adjacency matrix in dense format.

    Returns
    -------
    adj : numpy array
        Constructed adjacency matrix.
    '''
    # checks for ch_names
    if ch_names is not None:
        ch_names_from_neighb = False
        assert isinstance(ch_names, list), 'ch_names must be a list.'
        assert all(map(lambda x: isinstance(x, str), ch_names)), \
            'ch_names must be a list of strings'
    else:
        ch_names_from_neighb = True
        ch_names = neighbours['label']
        if isinstance(ch_names, np.ndarray):
            ch_names = ch_names.tolist()

    if (isinstance(neighbours, dict) and 'adjacency' in neighbours
            and isinstance(neighbours['adjacency'], np.ndarray)
            and neighbours['adjacency'].dtype == 'bool'):
        # python dictionary from .hdf5 file
        if ch_names_from_neighb:
            adj = neighbours['adjacency']
        else:
            ch_idx = [neighbours['label'].index(ch) for ch in ch_names]
            adj = neighbours['adjacency'][ch_idx][:, ch_idx]
    else:
        # fieldtrip adjacency structure
        n_channels = len(ch_names)
        adj = np.zeros((n_channels, n_channels), dtype='bool')

        for ch_idx, chan in enumerate(ch_names):
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
            connections = [ch_names.index(ch)
                           for ch in neighbours['neighblabel'][ngb_ind]
                           if ch in ch_names]
            adj[ch_idx, connections] = True

    if as_sparse:
        return sparse.coo_matrix(adj)
    return adj


def _get_units(dimname, fullname=False):
    '''Return unit for specified dimension name.'''
    if not fullname:
        return {'freq': 'Hz', 'time': 's', 'vert': 'vert'}[dimname]
    else:
        return {'freq': 'hertz', 'time': 'seconds',
                'vert': 'vertices'}[dimname]


# TODO: add singular=False to have vertex and vertices possible
def _full_dimname(dimname):
    '''Return unit for specified dimension name.'''
    return {'freq': 'frequency', 'time': 'time', 'vert': 'vertices',
            'chan': 'channels'}[dimname]


def _get_dimcoords(clst, dim_idx, idx=None):
    if idx is None:
        idx = slice(None)

    if (clst.dimcoords[dim_idx] is None
        or isinstance(clst.dimcoords[dim_idx][0], str)):
        numel = clst.stat.shape[dim_idx]
        coords = np.arange(numel)[idx]
    else:
        coords = clst.dimcoords[dim_idx][idx]
    return coords


def _mark_cluster_range(msk, x_values, ax):
    from matplotlib.patches import Rectangle

    alpha = 0.5
    color = [0.75, 0.75, 0.75]

    grp = group_mask(msk)
    ylims = ax.get_ylim()
    y_rng = np.diff(ylims)
    hlf_dist = np.diff(x_values).mean() / 2
    for gr in grp:
        this_x = x_values[gr[0]:gr[1] + 1]
        start = this_x[0] - hlf_dist
        length = np.diff(this_x[[0, -1]]) + hlf_dist * 2
        ptch = Rectangle((start, ylims[0]), length, y_rng, lw=0,
                         facecolor=color, alpha=alpha)
        ax.add_patch(ptch)


def _check_stc(clst):
    '''Make sure Clusters has a list of mne.SourceEstimate in stc attribute.'''
    import mne
    if clst.stc is None:
        vertices = clst.dimcoords[0]
        if vertices is None:
            vert = [clst.src[0]['vertno'], clst.src[1]['vertno']]
        else:
            # this should use _to_data_vert when it is moved from DiamSar
            lh, rh = [vertices[hemi] for hemi in ['lh', 'rh']]
            vert = [clst.src[0]['vertno'][lh], clst.src[1]['vertno'][rh]]

        assert clst.dimnames.index('vert') == 0

        tmin, tstep = 1., 1.
        if len(clst.dimnames) > 1:
            data_single = clst.stat[:, [0]]
        else:
            data_single = clst.stat[:, np.newaxis].copy()

        clst.stc = mne.SourceEstimate(data_single, vertices=vert, tmin=tmin,
                                      tstep=tstep, subject=clst.subject)


# prepare figure colorbar limits
# - [ ] make more universal? there is already some public function for this...
def _get_clim(data, vmin=None, vmax=None, pysurfer=False):
    """Get color limits from data - rounding to steps of 0.5."""
    if vmin is None and vmax is None:
        vmax = np.abs([data.min(), data.max()]).max()
        vmax_round = np.round(vmax)
        if np.abs(np.abs(vmax_round - vmax) - 0.5) < 0.25:
            vmax_round += 0.5 * np.sign(vmax - vmax_round)
        vmin, vmax = -vmax_round, vmax_round
    elif vmin is None:
        vmin = -vmax
    elif vmax is None:
        vmax = -vmin

    if pysurfer:
        return dict(kind='value', lims=[vmin, 0, vmax])
    else:
        return vmin, vmax


def _handle_dims(clst, dims):
    '''Find indices of dimension names.'''
    if dims is None:
        if clst.dimnames[0] in ['chan', 'vert']:
            return [0]
        else:
            raise ValueError("Can't infer the dimensions to plot when the"
                             " first dimension is not 'chan' or 'vert'."
                             " Please use ``dims`` argument.")
    else:
        if isinstance(dims, str):
            dims = [dims]
        dim_idx = [clst.dimnames.index(dim) for dim in dims]
        return dim_idx


# TODO:
# - [x] _aggregate_cluster aggregates by default everything except the spatial
#       dimension. This would be problematic for spaces like [freq, time]
#       consider adding ``dim`` argument. Then ``ignore_space`` could be
#       removed.
# - [~] make sure dimensions are sorted according to ``ignore_dims``
#       (this is done elsewhere - in plotting now, here it might not matter)
# - [ ] beware of changing dimension order for some complex "facny index"
#       operations
def _aggregate_cluster(clst, cluster_idx, ignore_dims=None,
                       mask_proportion=0.5, retain_mass=0.65, **kwargs):
    '''Aggregate cluster mask and cluster stat map.

    Parameters
    ----------
    clst : borsar.Clusters
        Clusters object to use in aggregation.
    cluster_idx : int | list of int
        Cluster index ord indices to aggregate.
    ignore_dims : str | list of str | None
        Dimensions to leave out when aggregating. These dimensions are retained
        and thus present in the output. ``None`` defaults to the spatial
        dimension.
    mask_proportion : float
        When aggregating cluster mask: retain only mask elements that
        participate at least in ``mask_proportion`` part of the aggregated
        space. The default is 0.5.
    retain_mass : float
        If a dimension has to be aggregated but is not specified in
        ``**kwargs`` - define range to aggregate over by retaining at least
        ``retain_mass`` proportion of cluster mass along that dimension.
        FIXME - add note about "see also".
    **kwargs : additional arguments
        Additional arguments used in aggregation, defining the points to
        select (if argument value is a list of float) or the range to
        aggregate for the dimension specified by the argument name. Tuple
        of two values defines explicit range: for example keyword argument
        ``freq=(6, 8)`` aggregates the 6 - 8 Hz range. List of floats
        defines specific points to pick: for example ``time=[0.1, 0.2]``
        selects time points corresponding to 0.1 and 0.2 seconds.
        Float argument between 0. and 1. defines range that is dependent on
        cluster mass or extent. For example ``time=0.75`` defines time
        range that retains at least 75% of the cluster extent (calculated
        along the aggregated dimension - in this case time). If no kwarg is
        passed for given dimension then the default value is ``0.65``.
        This means that the range for such dimension is defined to retain
        at least 65% of the cluster extent.

    Returns
    -------
    clst_mask : bool ndarray
        Aggregated cluster mask.
    clst_stat : ndarray
        Aggregated cluster stats array.
    idx : tuple of indexers
        Indexers for the aggregated dimensions.
        See ``borsar.Cluster.get_index``
    '''
    cluster_idx = ([cluster_idx] if not isinstance(cluster_idx, list)
                   else cluster_idx)
    n_clusters = len(cluster_idx)

    # FIXME - throw an error instead if at least one cluster idx exceeds
    #         number of clusters in the `clst` object
    dim_idx = _handle_dims(clst, ignore_dims)
    if ignore_dims is None and len(dim_idx) > 0:
        ignore_dims = [clst.dimnames[ix] for ix in dim_idx]
    do_aggregation = clst.stat.ndim > 1 and (clst.stat.ndim - len(dim_idx) > 0)

    if cluster_idx[0] is not None:
        cluster_idx = ([None] if len(clst) < max(cluster_idx) + 1
                       else cluster_idx)

    # aggregating multiple clusters is eligible only when the dimname kwargs
    # exhaust the aggregated space and no dimension is set by retained mass
    if n_clusters > 1 and do_aggregation:
        dimnames = clst.dimnames.copy()
        for dim in np.sort(dim_idx)[::-1]:
            dimnames.pop(dim)

        non_exhausted = list()
        for dimname in dimnames:
            if dimname not in kwargs:
                non_exhausted.append(dimname)
            elif isinstance(kwargs[dimname], float):
                non_exhausted.append(dimname)
        if len(non_exhausted) > 0:
            raise ValueError('If aggregating multiple clusters all the '
                             'aggregated dimensions must be fully specified ('
                             'without referring to retained cluster mass). '
                             'Some dimensions were not fully specified: '
                             '{}'.format(', '.join(non_exhausted)))

    # find indexing
    idx = clst.get_index(cluster_idx=cluster_idx[0], ignore_dims=ignore_dims,
                         retain_mass=retain_mass, **kwargs)
    sequences = [x for x in range(len(idx))
                 if isinstance(idx[x], (list, np.ndarray))]
    if len(sequences) > 1:
        # FIXME - separate function
        # we have to use np.ix_ to turn multiple lists/arrays to
        # indexers acceptable by numpy (.oix would be helpful here...)
        seq = [[0]] * len(idx)
        for s_idx in sequences:
            seq[s_idx] = idx[s_idx]

        seq = np.ix_(*seq)
        idx = list(idx)
        for s_idx in sequences:
            idx[s_idx] = seq[s_idx]
        idx = tuple(idx)

    if do_aggregation:
        reduce_axes = tuple(ix for ix in range(0, clst.stat.ndim)
                            if not (isinstance(idx[ix], (list, np.ndarray))
                            or ix in dim_idx))
        # reduce spatial if present, not in dim_idx and list / array
        if (0 not in reduce_axes and clst.dimnames[0] in ['chan', 'vert']
            and 0 not in dim_idx and isinstance(idx[0], (list, np.ndarray))):
            reduce_axes = (0, ) + reduce_axes

        clst_stat = clst.stat[idx].mean(axis=reduce_axes)

        if cluster_idx[0] is not None:
            clst_idx = (slice(None),) + idx
            reduce_mask_axes = tuple(ix + 1 for ix in reduce_axes)
            clst_mask = (clst.clusters[cluster_idx][clst_idx].mean(
                         axis=reduce_mask_axes) >= mask_proportion
                         if cluster_idx[0] is not None else None)
        else:
            clst_mask = None
    else:
        # no aggregation
        # FIXME - what if more clusters?
        clst_stat = clst.stat.copy()
        clst_mask = (clst.clusters[cluster_idx] if cluster_idx[0] is not None
                     else None)

    if len(cluster_idx) == 1 and not clst_mask is None:
        clst_mask = clst_mask[0]

    return clst_mask, clst_stat, idx


# TODO - save labels? this would require saving all parameters
#        of cluter reduction and cluster index
def _label_from_cluster(clst, clst_mask):
    '''Get pysurfer label from cluster mask.'''
    import mne
    clst.stc.data[:, 0] = clst_mask
    labels_l, labels_r = mne.stc_to_label(
        clst.stc, src=clst.src, subjects_dir=clst.subjects_dir, smooth=True)
    if isinstance(labels_l, list):
        clst_label = labels_l[0] if len(labels_l) > 0 else labels_r[0]
    else:
        clst_label = labels_l if labels_l is not None else labels_r
    return clst_label


def _prepare_cluster_description(clst, cluster_idx, idx, reduce_axes=None):
    """Prepare text description about cluster range and p value."""
    if clst.stat.ndim > 1:
        if reduce_axes is None:
            # TODO - start at one only when showing spatial map
            reduce_axes = tuple(range(1, clst.stat.ndim))
        time_label = list()
        for dim in reduce_axes:
            uni = _get_units(clst.dimnames[dim])
            coords = ('{} - {}'.format(*clst.dimcoords[dim][idx[dim]][[0, -1]])
                      if clst.dimcoords is not None else 'all')
            time_label.append('{} {}'.format(coords, uni))
        time_label = ', '.join(time_label) + '\n'
    else:
        time_label = ''

    pval = _format_cluster_pvalues(clst, cluster_idx)
    time_label += pval
    return time_label


def _format_cluster_pvalues(clst, idx):
    '''Format cluster p values into a string.'''
    if clst.pvals is not None:
        pvals = clst.pvals[idx]
        if isinstance(pvals, np.ndarray):
            pval = 'p = ' + ', '.join(['{:.3f}'.format(x).rstrip('0')
                                       for x in pvals])
        else:
            pval = format_pvalue(pvals)
    else:
        pval = 'p = NA'
    return pval


def _get_full_dimname(dimname):
    '''Return full dimension name.'''
    dct = {'freq': 'frequency', 'time': 'time', 'vert': 'vertices',
           'chan': 'channels'}
    return dct[dimname] if dimname in dct else dimname


def _get_mass_range(contrib, mass, adjacent=True):
    '''Find range that retains given mass (sum) of the contributions vector.

    Parameters
    ----------
    contrib : np.ndarray
        Vector of contributions.
    mass : float
        Requested mass to retain.
    adjacent : boolean
        Whether to extend from the maximum point by adjacency or not.

    Returns
    -------
    extent : slice or np.ndarray of int
        Slice (when `adjacency=True`) or indices (when `adjacency=False`)
        retaining the required mass.
    '''
    contrib_len = contrib.shape[0]
    max_idx = np.argmax(contrib)
    current_mass = contrib[max_idx]

    if adjacent:
        side_idx = np.array([max_idx, max_idx])
        while current_mass < mass:
            side_idx += [-1, +1]
            vals = [0. if side_idx[0] < 0 else contrib[side_idx[0]],
                    0. if side_idx[1] + 1 >= contrib_len
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
    else:
        indices = np.argsort(contrib)[::-1]
        cum_mass = np.cumsum(contrib[indices])
        retains_mass = np.where(cum_mass >= mass)[0]
        if len(retains_mass) > 0:
            indices = indices[:retains_mass[0] + 1]
        return np.sort(indices)


def _cluster_selection(clst, sel):
    '''Select Clusters according to selection vector `sel`. Works in-place.

    Parameters
    ----------
    clst : borsar.Clusters
        Clusters to select from.
    sel : list-like of int | array of bool
        Clusters to select.

    Returns
    -------
    clst : borsar.Clusters
        Selected clusters.
    '''
    if isinstance(sel, np.ndarray):
        if sel.dtype == 'bool' and sel.all():
            return clst
        if sel.dtype == 'bool':
            sel = np.where(sel)[0]

    if len(sel) > 0:
        # select relevant fields
        clst.cluster_polarity = [clst.cluster_polarity[idx]
                                 for idx in sel]
        clst.clusters = clst.clusters[sel]
        clst.pvals = clst.pvals[sel]
    else:
        # no cluster selected - returning empty Clusters
        clst.cluster_polarity = []
        clst.clusters = None
        clst.pvals = None
    return clst


def _ensure_correct_info(clst):
    # check if we have channel names:
    has_ch_names = clst.dimcoords[0] is not None
    if has_ch_names:
        from mne import pick_info

        ch_names = clst.dimcoords[0]
        ch_idx = find_channels(clst.info, ch_names)
        clst.info = pick_info(clst.info, ch_idx)


# - [ ] add special functions for handling dims like vert or chan
def _index_from_dim(dimnames, dimcoords, **kwargs):
    '''
    Find axis indices or slices given dimnames, dimcoords and dimname keyword
    arguments of ``dimname=value`` form.

    Parameters
    ----------
    dimnames : list of str
        List of dimension names. For example ``['chan', 'freq']``` or
        ``['vert', 'time']``. The length of `dimnames` has to mach
        ``stat.ndim``.
    dimcoords : list of arrays
        List of arrays, where each array contains coordinates (labels) for
        consecutive elements in corresponding dimension.
    **kwargs : additional keywords
        Keywords referring to dimension names. Value of these keyword has to be
        either:
        * a tuple of two numbers representing lower and upper limits for
          dimension selection.
        * a list of one or more numbers, where each number represents specific
          point in coordinates of the given dimension.

    Returns
    -------
    idx : tuple of slices
        Tuple that can be used to index the data array: `data[idx]`.

    Examples
    --------
    >>> dimnames = ['freq', 'time']
    >>> dimcoords = [np.arange(8, 12), np.arange(-0.2, 0.6, step=0.05)]
    >>> _index_from_dim(dimnames, dimcoords, freq=(9, 11), time=(0.25, 0.5))
    (slice(1, 4, None), slice(9, 15, None))
    >>> _index_from_dim(dimnames, dimcoords, freq=[9, 11], time=(0.25, 0.5))
    ([1, 3], slice(9, 15, None))
    '''

    idx = list()
    for dname, dcoord in zip(dimnames, dimcoords):
        if dname not in kwargs:
            idx.append(slice(None))
            continue
        sel_ax = kwargs.pop(dname)
        if isinstance(sel_ax, tuple) and len(sel_ax) == 2:
            idx.append(find_range(dcoord, sel_ax))
        elif isinstance(sel_ax, list):
            idx.append(find_index(dcoord, sel_ax))
        else:
            raise TypeError('Keyword arguments has to have tuple of length 2 '
                            'or a list, got {}.'.format(type(sel_ax)))
    return tuple(idx)
