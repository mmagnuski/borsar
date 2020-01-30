import numpy as np

from .utils import group_mask
from .stats import format_pvalue


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

    if clst.dimcoords[dim_idx] is not None:
        coords = clst.dimcoords[dim_idx][idx]
    else:
        numel = clst.stat.shape[dim_idx]
        coords = np.arange(numel)[idx]
    return coords


def _label_axis(ax, clst, dim_idx, ax_dim):
    '''Label x or y axis with relevant label according to cluster dimnames, its
    unit and indexed range.'''
    dimname = clst.dimnames[dim_idx]
    if dimname == 'chan':
        label = 'Channels'
    else:
        label = _full_dimname(dimname).capitalize()
        unit = _get_units(dimname)
        label = label + ' ({})'.format(unit)

    if ax_dim == 'x':
        ax.set_xlabel(label, fontsize=12)
        if dimname == 'chan':
            ax.set_xticks([])
    elif ax_dim == 'y':
        ax.set_ylabel(label, fontsize=12)
        if dimname == 'chan':
            ax.set_yticks([])


def _label_topos(topo, dim_kwargs):
    '''Label cluster topoplots with relevant units.'''

    if len(dim_kwargs) == 1:
        # currently works only for one selected dimension
        dim = list(dim_kwargs.keys())[0]
        unit = _get_units(dim)
        values = dim_kwargs[dim]
        labels = [str(v) for v in values]

        if isinstance(values, (list, np.ndarray)):
            assert len(topo) == len(values)
            for tp, lb in zip(topo, labels):
                tp.axes.set_title(lb + ' ' + unit, fontsize=12)

        elif isinstance(values, tuple) and len(values) == 2:
            # range
            ttl = '{} - {} {}'.format(*labels, unit)
            topo.axes.set_title(ttl, fontsize=12)


def _mark_cluster_range(msk, x_values, ax):
    from matplotlib.patches import Rectangle

    alpha = 0.5
    color = [0.75, 0.75, 0.75]

    grp = group_mask(msk)
    ylims = ax.get_ylim()
    y_rng = np.diff(ylims)
    hlf_dist = np.diff(x_values).mean()
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
    # FIXME - what if more clusters?
    idx = clst.get_index(cluster_idx=cluster_idx[0], ignore_dims=ignore_dims,
                         retain_mass=retain_mass, **kwargs)

    if do_aggregation:
        # CHECK/FIXME - why are the dimensions with a list of ndarray
        #               not reduced? because they may be discontinuous?
        reduce_axes = tuple(ix for ix in range(0, clst.stat.ndim)
                            if not (isinstance(idx[ix], (np.ndarray, list))
                                    or ix in dim_idx))
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

    # aggregate masks if more clusters
    # CONSIDER - this could be made optional later,
    # especially with cluster colors
    if clst_mask is not None:
        clst_mask = clst_mask.any(axis=0)

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
    time_label += format_pvalue(clst.pvals[cluster_idx])
    return time_label
