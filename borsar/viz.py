import numpy as np
import matplotlib.pyplot as plt

from .channels import get_ch_pos


class Topo(object):
    '''High-level object that allows for convenient topographic plotting.

    Parameters
    ----------
    values : numpy array
        Values to plot topographically. Has to be of shape ``(n_channels,)`` or
        ``(n_channels, n_topos)`` for multiple topographies to show.
    info : mne Info instance
        Info object containing channel positions.
    axes : matplotlib Axes, optional
        Axes to plot in. Creates new by default. The axes handle is later
        available in ``.axes`` attribute. If ``values`` is two dimensional
        then ``axes`` has to be a list of matplotlib Axes of length equal
        to the size of the second dimension (``values.shape[1]``).
    **kwargs : any additional keyword arguments
        Additional keyword arguments are passed to `mne.viz.plot_topomap`.

    Returns
    -------
    topo : borsar.viz.Topo
        Topo object that exposes various useful methods like `remove_levels`
        or `mark_channels`.

    Example
    -------
    topo = Topo(values, info, axis=ax)
    topo.remove_levels(0)
    topo.solid_lines()
    topo.set_linewidth(1.5)
    topo.mark_channels([4, 5, 6], markerfacecolor='r', markersize=12)
    '''

    def __init__(self, values, info, side=None, **kwargs):
        from mne.viz.topomap import plot_topomap

        self.info = info
        self.values = values

        # handle multiple axes
        multi_axes = values.ndim > 1
        if multi_axes:
            n_topos = values.shape[1]

        # FIXME: split axis checking and plotting into separate methods
        has_axis = 'axes' in kwargs.keys()
        if has_axis:
            axes = kwargs['axes']
            if multi_axes:
                from mne.viz.utils import _validate_if_list_of_axes
                _validate_if_list_of_axes(axes, obligatory_len=values.shape[1])
                plt.sca(axes[0])
            else:
                plt.sca(axes)
            self.axes = axes
        elif multi_axes:
            fig, axes = plt.subplots(ncols=n_topos)
            self.axes = axes.tolist()

        part = _infer_topo_part(info)
        if part is not None:
            info, kwargs = _construct_topo_part(info, part, kwargs)

        # plot using mne's `plot_topomap`
        if multi_axes:
            im, lines, chans = list(), list(), list()
            self.marks = [list() for idx in range(n_topos)]

            kwargs.update({'show': False})
            for topo_idx in range(n_topos):
                this_im, this_lines = plot_topomap(values[:, topo_idx], info,
                                                   axes=self.axes[topo_idx],
                                                   **kwargs)
                # get channel objects and channel positions from topo
                this_chans, chan_pos = _extract_topo_channels(this_im.axes)

                im.append(this_im)
                lines.append(this_lines)
                chans.append(this_chans)
            self.chan_pos = chan_pos
        else:
            im, lines = plot_topomap(values, info, **kwargs)

            self.marks = list()
            self.axes = axes if has_axis else im.axes

            # get channel objects and channel positions from topo
            self.chans, self.chan_pos = _extract_topo_channels(im.axes)

        self.img = im
        self.lines = lines
        self.fig = im[0].figure if isinstance(im, list) else im.figure


    def remove_levels(self, lvl):
        '''
        Remove countour lines at specified levels.

        Parameters
        ----------
        lvl : value | list of values
            Remove contour lines at these levels.
        '''
        if not isinstance(lvl, list):
            lvl = [lvl]

        iter_lines = (self.lines if isinstance(self.lines, list)
                      else [self.lines])

        for lines in iter_lines:
            for l in lvl:
                remove_lines = np.where(lines.levels == l)[0]
                for rem_ln in remove_lines:
                    lines.collections[rem_ln].remove()
                for pop_ln in np.flipud(np.sort(remove_lines)):
                    lines.collections.pop(pop_ln)

    def solid_lines(self):
        '''Turn all contour lines to solid style (no dashed lines).'''
        self.set_linestyle('-')

    def set_linestyle(self, *args, **kwargs):
        '''
        Set specific linestyle to all contour lines.

        Parameters
        ----------
        *args : arguments
            Arguments are passed to `set_linestyle` of each line.
        **kwargs : keyword arguments
            Keyword arguments are passed to `set_linestyle` of each line.
        '''
        iter_lines = (self.lines if isinstance(self.lines, list)
                      else [self.lines])
        for lines in iter_lines:
            for line in lines.collections:
                line.set_linestyle(*args, **kwargs)

    def set_linewidth(self, lw):
        '''
        Set contour lines linewidth.

        Parameters
        ----------
        lw : int | float
            Desired line width of the contour lines.
        '''
        iter_lines = (self.lines if isinstance(self.lines, list)
                      else [self.lines])
        for lines in iter_lines:
            for line in lines.collections:
                line.set_linewidths(lw)

    def mark_channels(self, chans, **kwargs):
        '''
        Highlight specified channels with markers.

        Parameters
        ----------
        chans : numpy array of int or bool
            Channels to highlight. You can use either:
            * an integer array with channel indices
            * a list with channel indices
            * boolean array of shape ``(n_channels,)``.
            When ``Topo`` created multiple topographies and you want a
            different selection of channels highlighted in each use either:
            * a list of lists of indices
            * a list of numpy arrays, where each array is either int or bool
            * a boolean array of ``(n_channels, n_topos)`` shape.
        **kwargs : additional keyword arguments
            Any additional keyword arguments are passed as arguments to
            `plt.plot`. It is useful for defining marker properties like
            `marker`, `markerfacecolor`, `markeredgecolor`, `linewidth` or
            `markersize`.
        '''
        default_marker = dict(marker='o', markerfacecolor='w', markersize=8,
                              markeredgecolor='k', linewidth=0)
        default_marker.update(kwargs)

        # FIXME: add len(topo) and make topo iterable
        # mark channels and save marks in `self.marks` list
        n_topos = len(self.axes) if isinstance(self.axes, list) else 1
        iter_marks = (self.marks if n_topos > 1 else [self.marks])
        iter_axes = (self.axes if n_topos > 1 else [self.axes])

        # make sure channel selection is iterable
        if (isinstance(chans, (list, tuple)) and not
            isinstance(chans[0], (list, tuple, np.ndarray))):
                chans = [chans]
        elif isinstance(chans, np.ndarray):
            chans = (np.tile(chans, (n_topos, 1)).T if chans.ndim == 1
                     else chans.T)

        for ax, marks, msk in zip(iter_axes, iter_marks, chans):
            this_marks = ax.plot(
                self.chan_pos[msk, 0], self.chan_pos[msk, 1], **default_marker)
            marks.append(this_marks)


def _extract_topo_channels(ax):
    '''
    Extract channel positions from mne topoplot axis.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis containing the topoplot.

    Returns
    -------
    chans : `matplotlib.patches.Circle` or `matplotlib.collections.PathCollection`
        Matplotlib object representing channels. Some older mne versions use
        `plt.scatter` to draw channels so the channels are marked with
        `mpl.patches.Circle`. At other times `mpl.collections.PathCollection`
        is being used.
    chan_pos : numpy array
        Numpy array of shape (n_channels, 2) representing channel positions.
        First column contains the x position and second column the y position.
    '''
    import matplotlib as mpl

    # first, look for circles
    circles = ax.findobj(mpl.patches.Circle)
    if len(circles) > 0:
        chans = circles
        chan_pos = np.array([ch.center for ch in chans])
    else:
        # if there are no circles: look for PathCollection
        path_collection = ax.findobj(mpl.collections.PathCollection)
        if len(path_collection) > 0:
            chans = path_collection[0]
            chan_pos = chans.get_offsets()
        else:
            msg = ('Could not find matplotlib objects representing channels. '
                   'Looked for `matplotlib.patches.Circle` and `matplotlib.'
                   'collections.PathCollection`.')
            raise RuntimeError(msg)

    return chans, chan_pos


def _infer_topo_part(info):
    """Infer whether a specific part of the topography should be shown.

    For example when only channels on the left are shown, the right side of the
    topography should be masked.
    This function will be less useful once convex-hull masking is available in
    mne-python.
    """
    ch_pos = get_ch_pos(info)
    all_x_above_0 = (ch_pos[:, 0] >= 0.).all()
    all_y_above_0 = (ch_pos[:, 1] >= 0.).all()
    side = ''
    if all_x_above_0:
        side += 'right'
    elif (ch_pos[:, 0] <= 0.).all():
        side += 'left'

    if all_y_above_0:
        side = 'frontal' if len(side) == 0 else '_'.join([side, 'frontal'])

    side = None if len(side) == 0 else side
    return side


def _construct_topo_part(info, part, kwargs):
    """Mask part of the topography."""
    from mne.viz.topomap import _check_outlines, _find_topomap_coords

    # create head circle
    use_skirt = kwargs.get('outlines', None) == 'skirt'
    radius = 0.5 if not use_skirt else 0.65 # this does not seem to change much
    ll = np.linspace(0, 2 * np.pi, 101)
    head_x = np.cos(ll) * radius
    head_y = np.sin(ll) * radius
    mask_outlines = np.c_[head_x, head_y]

    # create mask
    if 'right' in part:
        below_zero = mask_outlines[:, 0] < 0
        removed_len = below_zero.sum()
        filling = np.zeros((removed_len, 2))
        filling[:, 1] = np.linspace(radius, -radius, num=removed_len)
        mask_outlines[below_zero, :] = filling
    elif 'left' in part:
        above_zero = mask_outlines[:, 0] > 0
        removed_len = above_zero.sum()
        filling = np.zeros((removed_len, 2))
        filling[:, 1] = np.linspace(-radius, radius, num=removed_len)
        mask_outlines[above_zero, :] = filling
    if 'frontal' in part:
        below_zero = mask_outlines[:, 1] < 0
        removed_len = below_zero.sum()
        filling = np.zeros((removed_len, 2))
        lo = 0. if 'right' in part else -radius
        hi = 0. if 'left' in part else radius
        filling[:, 0] = np.linspace(lo, hi, num=removed_len)
        mask_outlines[below_zero, :] = filling

    head_pos = dict(center=(0., 0.))
    picks = range(len(info['ch_names']))
    pos = _find_topomap_coords(info, picks=picks)

    # TODO currently uses outlines='head', but should change later
    outlines = kwargs.get('outlines', 'head')
    pos, outlines = _check_outlines(pos, outlines=outlines,
                                    head_pos=head_pos)

    # scale pos to min - max of the circle (the 0.425 value was hand-picked)
    scale_factor = 0.425 if not use_skirt else 0.565
    scale_x = scale_factor / pos[:, 0].max()
    scale_y = scale_factor / np.abs(pos[:, 1]).max()
    pos[:, 0] *= scale_x
    pos[:, 1] *= scale_y

    outlines['mask_pos'] = (mask_outlines[:, 0], mask_outlines[:, 1])
    kwargs.update(dict(outlines=outlines, head_pos=head_pos))

    info = pos
    return info, kwargs
