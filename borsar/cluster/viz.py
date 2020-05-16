import numpy as np

from .checks import _check_dimname_arg
from .utils import (_get_full_dimname, _get_units, _get_clim, _handle_dims,
                    _aggregate_cluster, _get_dimcoords, _mark_cluster_range,
                    _full_dimname)


# TODO - add special case for dimension='vert' and 'chan'
def plot_cluster_contribution(clst, dimension, picks=None, axis=None):
    '''
    Plot contribution of clusters along specified dimension.

    Parameters
    ----------
    dimension : str | int
        Dimension along which to calculate contribution.
    picks : list-like | None, optional
        Cluster indices whose contributions should be shown.
    axis : matplotlib Axes | None, optional
        Matplotlib axis to plot in.

    Returns
    -------
    axis : matplotlib Axes
        Axes with the plot.
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
        if dimcoords is None:
            dimcoords = np.arange(clst.stat.shape[idx])
        dimlabel = '{} ({})'.format(_get_full_dimname(dimension),
                                    _get_units(dimension))

    # make sure we have an axes to plot to
    if axis is None:
        axis = plt.gca()

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


# FIXME - allow for channel sorting (by region and y position)
def plot_cluster_chan(clst, cluster_idx=None, dims=None, vmin=None, vmax=None,
                      mark_clst_prop=0.65, mark_kwargs=None,
                      cluster_colors=None,**kwargs):
    '''Plot cluster in sensor space.

    Parameters
    ----------
    clst : Clusters
        Clusters object to use in plotting.
    cluster_idx : int | list
        Cluster index or list of cluster indices to plot.
    dims : str | list of str | None
        Dimensions to visualize. By default (``None``) only spatial dimension
        is plotted.
    vmin : float, optional
        Value mapped to minimum in the colormap. Inferred from data by default.
    vmax : float, optional
        Value mapped to maximum in the colormap. Inferred from data by default.
    mark_clst_prop : float
        Mark elements that exceed this proportion in the reduced cluster range.
        For example if 4 frequency bins are reduced using ``freq=(8, 12)``
        then if ``mark_clst_prop`` is ``0.5`` only channels contributing
        at least 2 frequency bins (4 bins * 0.5 proportion) in this range
        will be marked in the topomap.
    mark_kwargs : dict | None, optional
        Keyword arguments for ``Topo.mark_channels`` used to mark channels
        participating in selected cluster. For example:
        ``mark_kwargs={'markersize': 3}`` to change the size of the markers.
        ``None`` defaults to ``{'markersize: 5'}``.
    cluster_colors : list | None, optional
        List of cluster colors if plotting multiple clusters.
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
    topo : borsar.viz.Topo class
        Figure object used in plotting.

    Examples
    --------
    > # to plot the first cluster within 8 - 10 Hz
    > clst.plot(cluster_idx=0, freq=(8, 10))
    > # to plot the second cluster selecting frequencies that make up at least
    > # 70% of the cluster mass:
    > clst.plot(cluster_idx=1, freq=0.7)
    > # plot time-frequency representation of the first cluster:
    > clst.plot(cluster_idx=0, dims=['freq', 'time'])
    '''
    # TODO - if cluster_idx is not None and there is no such cluster - error
    #      - if no clusters and None - plot without highlighting
    cluster_idx = 0 if cluster_idx is None else cluster_idx

    # split kwargs into plotfun_kwargs and dim_kwargs
    plotfun_kwargs, dim_kwargs = dict(), dict()
    for k, v in kwargs.items():
        if k not in clst.dimnames:
            plotfun_kwargs[k] = v
        else:
            dim_kwargs[k] = v

    # consider dims
    dim_idx = _handle_dims(clst, dims)
    dims = [clst.dimnames[ix] for ix in dim_idx]

    # TODO - aggregate should work when no clusters
    # get and aggregate cluster mask and cluster stat
    # CONSIDER ? add retain_mass and mask_proportion to args?
    clst_mask, clst_stat, idx = _aggregate_cluster(
        clst, cluster_idx, ignore_dims=dims, mask_proportion=mark_clst_prop,
        retain_mass=0.65, **dim_kwargs)
    n_elements = clst_stat.shape[1] if clst_stat.ndim > len(dim_idx) else 1
    vmin, vmax = _get_clim(clst_stat, vmin=vmin, vmax=vmax,
                           pysurfer=False)

    if len(cluster_idx) > 1 and cluster_colors is None:
        clst_mask = clst_mask.any(axis=0)

    # Viz rules:
    # ----------
    # 1. if 1 spatial dim is plotted -> Topo
    # 2. if 2 dims are plotted -> heatmap
    # 3. if 1 non-spatial dim -> line
    if len(dim_idx) == 1:
        if dims[0] == 'chan':

            # topographical plot
            # ------------------

            # create Topo object
            from borsar.viz import Topo
            topo = Topo(clst_stat, clst.info, vmin=vmin, vmax=vmax, show=False,
                        **plotfun_kwargs)
            topo.solid_lines()

            # FIXME: labels axes also when resulting from idx reduction
            _label_topos(clst, topo, dim_kwargs, idx)

            # mark cluster channels
            _mark_topo_channels(topo, clst_mask, mark_kwargs, cluster_colors)

            return topo
        else:
            # line plot
            # ---------

            # show a line plot of the effect
            import matplotlib.pyplot as plt
            x_axis = _get_dimcoords(clst, dim_idx[0], idx[dim_idx[0]])
            fig, ax = plt.subplots()
            ax.plot(x_axis, clst_stat, **plotfun_kwargs)
            _label_axis(ax, clst, dim_idx[0], ax_dim='x')
            _mark_cluster_range(clst_mask, x_axis, ax)
            return ax
    elif len(dim_idx) == 2:
        # heatmap
        # -------

        from ..viz import heatmap
        outlines = True
        if clst_mask is None:
            clst_mask = np.zeros(clst_stat.shape, dtype='bool')

        if not clst_mask.any():
            outlines = False

        # make sure the dimension order is correct
        if not (np.sort(dim_idx) == dim_idx).all():
            clst_stat = clst_stat.T
            if clst_mask is not None:
                clst_mask = clst_mask.transpose(clst_mask.ndim - 1,
                                                clst_mask.ndim - 2)

        # get dimension coords
        x_axis = _get_dimcoords(clst, dim_idx[1], idx[dim_idx[1]])
        y_axis = _get_dimcoords(clst, dim_idx[0], idx[dim_idx[0]])

        if clst_mask.ndim > 2 and outlines and cluster_colors is not None:
            line_kwargs = plotfun_kwargs.get('line_kwargs', dict())
            line_kwargs['color'] = cluster_colors
            plotfun_kwargs['line_kwargs'] = line_kwargs

        axs = heatmap(clst_stat, mask=clst_mask, outlines=outlines,
                      x_axis=x_axis, y_axis=y_axis, vmin=vmin, vmax=vmax,
                      **plotfun_kwargs)

        # add dimension labels
        _label_axis(axs[0], clst, dim_idx[1], ax_dim='x')
        _label_axis(axs[0], clst, dim_idx[0], ax_dim='y')

        return axs
    else:
        raise ValueError("Can't plot more than two dimensions.")


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


def _label_topos(clst, topo, dim_kwargs, idx):
    '''Label cluster topoplots with relevant units.'''

    n_kwargs = len(dim_kwargs)
    if n_kwargs == 1:
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
    elif n_kwargs == 0:
        # if idx for non-spatial dimensions is not (None, None, None)
        good_idx = [isinstance(ix, slice) and not ix == slice(None)
                    for ix in idx]
        good_idx = np.where(good_idx)[0]
        good_idx = good_idx[good_idx > 0]
        if len(good_idx) > 0:
            range = idx[good_idx[0]]
            dim = clst.dimnames[good_idx[0]]
            unit = _get_units(dim)
            values = clst.dimcoords[good_idx[0]][range]
            labels = [str(v) for v in values[[0, -1]]]
            ttl = '{} - {} {}'.format(*labels, unit)
            topo.axes.set_title(ttl, fontsize=12)


def _mark_topo_channels(topo, clst_mask, mark_kwargs, cluster_colors):
    '''Mark topo channels contributing to clusters.'''
    if clst_mask is None or not clst_mask.any():
        return

    n_topos = len(topo)
    if mark_kwargs is not None:
        if 'markersize' not in mark_kwargs:
            mark_kwargs.update({'markersize': 5})
    else:
        mark_kwargs = dict(markersize=5)

    multi_clusters = clst_mask.ndim > 1 + int(n_topos > 1)
    if multi_clusters:
        n_clusters = clst_mask.shape[0]
        for clst_idx in range(n_clusters):
            mark_kwargs['markerfacecolor'] = cluster_colors[clst_idx]
            if n_topos == 1:
                topo.mark_channels(clst_mask[clst_idx], **mark_kwargs)
            else:
                for idx, tp in enumerate(topo):
                    tp.mark_channels(clst_mask[clst_idx, :, idx],
                                     **mark_kwargs)
    else:
        if n_topos == 1:
            topo.mark_channels(clst_mask, **mark_kwargs)
        else:
            for idx, tp in enumerate(topo):
                tp.mark_channels(clst_mask[:, idx], **mark_kwargs)
