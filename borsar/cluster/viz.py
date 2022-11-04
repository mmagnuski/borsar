import numpy as np

from .checks import _check_dimname_arg
from .utils import (_get_units, _get_clim, _handle_dims, _aggregate_cluster,
                    _get_dimcoords, _mark_cluster_range, _full_dimname,
                    _human_readable_dimlabel)
from ..viz import Topo


# - [x] use **args passed to clst.plot()
# - [ ] add intensity label to line/topo/heatmap plot
def plot_cluster_contribution(clst, dims, picks=None, axis=None, **kwargs):
    '''
    Plot contribution of clusters along specified dimension.

    Parameters
    ----------
    dims: str | list of str
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

    # check if any clusters
    n_clusters = len(clst)
    if n_clusters == 0:
        raise ValueError('No clusters present in Clusters object.')

    # select full range for unspecified dimensions
    # --------------------------------------------
    specified_dims = dims if isinstance(dims, list) else [dims]
    dim_idx = _check_dimname_arg(clst, specified_dims[0])
    unspecified_dims = [dim for dim in clst.dimnames
                        if dim not in specified_dims
                        or dim not in kwargs]
    if len(unspecified_dims) > 0:
        for unspec in unspecified_dims:
            dim_idx = _check_dimname_arg(clst, unspec)
            if unspec in ['chan', 'vert']:
                all = np.arange(clst.stat.shape[dim_idx])
            else:
                frm, to = clst.dimcoords[dim_idx][[0, -1]]
                all = (frm, to)
            kwargs[unspec] = all

    # plot
    # ----
    picks = list(range(n_clusters)) if picks is None else picks
    ax = plot_cluster_chan(clst, picks, dims=dims, plot_contribution=True,
                           retain_mass=1., axis=axis, cmap='viridis',
                           **kwargs)

    # create "intensity" label
    # ------------------------
    labeldims = kwargs.get('labeldims', True)
    if labeldims:
        nonreduced_dims = [dim for dim in clst.dimnames if dim not in dims]
        dimnames = [_full_dimname(dim, singular=True)
                    for dim in nonreduced_dims]
        binlabel = 'Number of {} bins'.format('-'.join(dimnames))
        if isinstance(ax, tuple):
            # heatmap with colorbar
            cbar = ax[1]
            cbar.set_label(binlabel)
        elif isinstance(ax, Topo):
            # only if contains colorbar - which will be added to Topo
            # in some time
            pass
        elif (isinstance(dims, str) or (isinstance(dims, list)
              and len(dims) == 1)):
            # line plot - label y axis
            ax.set_ylabel(binlabel)

            # make sure y axis min is 0
            ylims = list(ax.get_ylim())
            if not ylims[0] == 0:
                ylims[0] = 0
                ax.set_ylim(ylims)

    return ax


# FIXME - allow for channel sorting (by region and y position)
# FIXME - change mark_clst_prop to mask_proportion, mark_proportion,
#         mark_cluster?
def plot_cluster_chan(clst, cluster_idx=None, dims=None, vmin=None, vmax=None,
                      mark_clst_prop=0.5, mark_kwargs=None,
                      cluster_colors=None, plot_contribution=False,
                      retain_mass=0.65, **kwargs):
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
        will be marked in the topomap. Defaults to ``0.5``.
    mark_kwargs : dict | None, optional
        Keyword arguments for ``Topo.mark_channels`` used to mark channels
        participating in selected cluster. For example:
        ``mark_kwargs={'markersize': 3}`` to change the size of the markers.
        ``None`` defaults to ``{'markersize: 5'}``.
    cluster_colors : list | None, optional
        List of cluster colors if plotting multiple clusters.
    plot_contribution : bool
        Whether to plot cluster contribution instead of statistical map.
    retain_mass : float
        How to define ranges for dimensions that are not specified in kwargs.
        Limits for these dimensions are automatically selected to contain at
        least ``retain_mass`` proportion of cluster volume. Defaults to
        ``0.65``.
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
    #      - if None - plot without highlighting
    #      - if auto' (default) : autoselect cluster
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
    # CONSIDER ? add retain_mass to args?
    clst_mask, clst_stat, idx = _aggregate_cluster(
        clst, cluster_idx, ignore_dims=dims, mask_proportion=mark_clst_prop,
        retain_mass=retain_mass, mask_sum=plot_contribution, **dim_kwargs)

    # remove singleton dimensions from clst_mask, merge clusters
    multi_clusters = False
    if clst_mask is not None:
        singletons = np.where(np.array(clst_mask.shape) == 1)[0]
        listlikes = (list, np.ndarray)
        if len(singletons) > 0:
            clst_mask = np.squeeze(clst_mask, axis=tuple(singletons))
            stat_singletons = (singletons - 1)[singletons > 0]
            if len(stat_singletons) > 0:
                clst_stat = np.squeeze(clst_stat, axis=tuple(stat_singletons))

        # merge clusters if necessary
        if (cluster_colors is None and isinstance(cluster_idx, listlikes)
            and len(cluster_idx) > 1):
            if not plot_contribution:
                clst_mask = clst_mask.any(axis=0)
            else:
                multi_clusters = True

    show = clst_mask if plot_contribution else clst_stat
    vmin, vmax = _get_clim(show, vmin=vmin, vmax=vmax,
                           pysurfer=False)
    vmin = 0 if plot_contribution else vmin
    labeldims = True
    if 'labeldims' in plotfun_kwargs:
        labeldims = plotfun_kwargs.pop('labeldims')

    # Viz rules:
    # ----------
    # 1. if 1 spatial dim is plotted -> Topo
    # 2. if 2 dims are plotted -> heatmap
    # 3. if 1 non-spatial dim -> line
    if len(dim_idx) == 1:
        if dims[0] == 'chan':

            # topographical plot
            # ------------------

            # make sure 'axis' kwarg is also accepted
            if 'axis' in plotfun_kwargs:
                ax = plotfun_kwargs.pop('axis')
                plotfun_kwargs['axes'] = ax

            # for plotting contribution of multiple clusters we currently
            # just sum up the individual cluster contributions
            if multi_clusters:
                show = clst_mask.sum(axis=0)

            # create Topo object
            from borsar.viz import Topo
            topo = Topo(show, clst.info, vmin=vmin, vmax=vmax, show=False,
                        **plotfun_kwargs)
            topo.solid_lines()

            # mark cluster channels
            if not plot_contribution:
                _mark_topo_channels(topo, clst_mask, mark_kwargs,
                                    cluster_colors, cluster_idx)

            # FIXME: labels axes also when resulting from idx reduction
            if labeldims:
                _label_topos(clst, topo, dim_kwargs, idx)

            # if 'axes' not in plotfun_kwargs:
                # we created the axes so we can reposition them for better
                # topo titles visibility
                # _move_axes_to(topo.axes, y=0.05)

            return topo
        else:
            # line plot
            # ---------

            # show a line plot of the effect
            import matplotlib.pyplot as plt
            x_axis = _get_dimcoords(clst, dim_idx[0], idx[dim_idx[0]])

            # FIXME: temporary plot_contribution compatibility
            if 'cmap' in plotfun_kwargs:
                plotfun_kwargs.pop('cmap')

            # FIXME: temporary plot_contribution compatibility
            if 'axis' in plotfun_kwargs:
                ax = plotfun_kwargs.pop('axis')
                if ax is None:
                    fig, ax = plt.subplots()
            else:
                fig, ax = plt.subplots()

            if multi_clusters:
                show = show.T

            ax.plot(x_axis, show, **plotfun_kwargs)

            if labeldims:
                _label_axis(ax, clst, dim_idx[0], ax_dim='x')
            if not plot_contribution:
                _mark_cluster_range(clst_mask, x_axis, ax)
            return ax
    elif len(dim_idx) == 2:
        # heatmap
        # -------

        from ..viz import heatmap

        # make sure the dimension order is correct
        if not (np.sort(dim_idx) == dim_idx).all():
            clst_stat = clst_stat.T
            if clst_mask is not None:
                clst_mask = clst_mask.transpose(clst_mask.ndim - 1,
                                                clst_mask.ndim - 2)
            show = clst_mask if plot_contribution else clst_stat

        if plot_contribution:
            outlines = False
            if multi_clusters:
                show = clst_mask.sum(axis=0)
            clst_mask = None
        else:
            outlines = True
            if clst_mask is None:
                clst_mask = np.zeros(clst_stat.shape, dtype='bool')

            # CHECK/FIX: this should not be needed, heatmap should be smart
            # enough to not draw outlines then
            if not clst_mask.any():
                outlines = False

        # get dimension coords
        x_axis = _get_dimcoords(clst, dim_idx[1], idx[dim_idx[1]])
        y_axis = _get_dimcoords(clst, dim_idx[0], idx[dim_idx[0]])

        if (clst_mask is not None and clst_mask.ndim > 2 and outlines
            and cluster_colors is not None):
            line_kwargs = plotfun_kwargs.get('line_kwargs', dict())
            line_kwargs['color'] = cluster_colors
            plotfun_kwargs['line_kwargs'] = line_kwargs

        axs = heatmap(show, mask=clst_mask, outlines=outlines,
                      x_axis=x_axis, y_axis=y_axis, vmin=vmin, vmax=vmax,
                      **plotfun_kwargs)

        # add dimension labels
        if labeldims:
            heatmap_ax = axs[0] if isinstance(axs, (list, tuple)) else axs
            _label_axis(heatmap_ax, clst, dim_idx[1], ax_dim='x')
            _label_axis(heatmap_ax, clst, dim_idx[0], ax_dim='y')

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


# - [ ] improve by combining both n_kwargs branches so that it works for all
#       dims; dims could be listed separated by commas or newlines...
def _label_topos(clst, topo, dim_kwargs, idx):
    '''Label cluster topoplots with relevant units.'''

    ndims = len(idx)
    idx = [np.squeeze(ix) + 0 if isinstance(ix, np.ndarray) else ix
           for ix in idx]
    multi_point_idx = [ix for ix in range(ndims)
                       if isinstance(idx[ix], (list, np.ndarray))
                       and len(idx[ix]) > 1
                       and clst.dimnames[ix] not in ['chan', 'vert']]

    label_parts = list()
    for ix in range(1, ndims):
        sel = idx[ix]
        if ix in multi_point_idx or (isinstance(sel, slice)
                                     and sel == slice(None)):
            # FIXME - later if all times or frequencies are selected, this
            #         may be noted in the dimlabel
            continue

        dim = clst.dimnames[ix]
        unit = _get_units(dim)
        coords = clst.dimcoords[ix]
        values = coords[sel]

        label = _human_readable_dimlabel(values, idx[ix], coords, unit)
        label_parts.append(label)
    label = '\n'.join(label_parts)

    if len(multi_point_idx) == 1:
        multi_dim = multi_point_idx[0]
        multi_idx = idx[multi_dim]
        assert len(topo) == len(multi_idx)

        dimname = clst.dimnames[multi_dim]
        unit = _get_units(dimname)
        coords = clst.dimcoords[multi_dim]
        values = coords[multi_idx]
        multi_label = _human_readable_dimlabel(values, multi_idx, coords, unit)

        for tp, lbl in zip(topo, multi_label):
            if len(label) > 0:
                lbl += '\n' + label
            tp.axes.set_title(lbl, fontsize=12)

    elif len(multi_point_idx) == 0:
        topo.axes.set_title(label, fontsize=12)


def _mark_topo_channels(topo, clst_mask, mark_kwargs, cluster_colors,
                        cluster_idx):
    '''Mark topo channels contributing to clusters.'''
    if clst_mask is None or not clst_mask.any():
        return

    n_topos = len(topo)
    if mark_kwargs is not None:
        if 'markersize' not in mark_kwargs:
            mark_kwargs.update({'markersize': 5})
    else:
        mark_kwargs = dict(markersize=5)

    multi_clusters = (isinstance(cluster_idx, (list, np.ndarray))
                      and len(cluster_idx) > 1 and cluster_colors is not None)
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


def _move_axes_to(axes, x=None, y=None):
    # maybe .ravel() ndarray axes
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    if x is None and y is None:
        return axes

    for ax in axes:
        pos = list(ax.get_position().bounds)
        if x is not None:
            pos[0] = x
        if y is not None:
            pos[1] = y

        ax.set_position(pos)
    return axes
