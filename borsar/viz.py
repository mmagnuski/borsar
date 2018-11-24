import numpy as np
import matplotlib.pyplot as plt


class Topo(object):
    '''High-level object that allows for convenient topographic plotting.

    Parameters
    ----------
    values : numpy array
        Values to topographically plot.
    info : mne Info instance
        Info object containing channel positions.
    axes : matplotlib Axes, optional
        Axes to plot in. Creates new by default. The axes handle is later
        available in `.axis` attribute.
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

        has_axis = 'axes' in kwargs.keys()
        if has_axis:
            self.axis = kwargs['axes']
            plt.sca(self.axis)

        part = _infer_topo_part(info)
        if part is not None:
            info, kwargs = _construct_topo_part(info, part, kwargs)

        # plot using mne's `plot_topomap`
        im, lines = plot_topomap(values, info, **kwargs)

        self.img = im
        self.lines = lines
        self.marks = list()
        self.fig = im.figure
        if not has_axis:
            self.axis = im.axes

        # get channel objects and channel positions from topo
        self.chans, self.chan_pos = _extract_topo_channels(im.axes)

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
        for l in lvl:
            remove_lines = np.where(self.lines.levels == l)[0]
            for rem_ln in remove_lines:
                self.lines.collections[rem_ln].remove()
            for pop_ln in np.flipud(np.sort(remove_lines)):
                self.lines.collections.pop(pop_ln)

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
        for ln in self.lines.collections:
            ln.set_linestyle(*args, **kwargs)

    def set_linewidth(self, lw):
        '''
        Set contour lines linewidth.

        Parameters
        ----------
        lw : int | float
            Desired line width of the contour lines.
        '''
        for ln in self.lines.collections:
            ln.set_linewidths(lw)

    def mark_channels(self, chans, **kwargs):
        '''
        Highlight specified channels with markers.

        Parameters
        ----------
        chans : numpy array of int or bool
            Channels to highlight. Integer array with channel indices or
            boolean array of shape (n_channels,).
        **kwargs : additional keyword arguments
            Any additional keyword arguments are passed as arguments to
            `plt.plot`. It is useful for defining marker properties like
            `marker`, `markerfacecolor`, `markeredgecolor`, `linewidth` or
            `markersize`.
        '''
        default_marker = dict(marker='o', markerfacecolor='w', markersize=8,
                              markeredgecolor='k', linewidth=0)
        for k in kwargs.keys():
            default_marker[k] = kwargs[k] # or just .update(kwargs)

        # mark channels and save marks in `self.marks` list
        marks = self.axis.plot(
            self.chan_pos[chans, 0], self.chan_pos[chans, 1], **default_marker)
        self.marks.append(marks)


def _extract_topo_channels(ax):
    '''
    Extract channels positions from mne topoplot axis.

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
    ch_pos = get_ch_pos(info)
    all_x_above_0 = (ch_pos[:, 0] >= 0.).all()
    all_y_above_0 = (ch_pos[:, 1] >= 0.).all()
    side = ''
    if all_x_above_0:
        side += 'right'

    if all_y_above_0:
        side = 'frontal' if len(side) == 0 else '_'.join(side, 'frontal')

    side = None if len(side) == 0 else side
    return side


def _construct_topo_part(info, part='right', kwargs):
    from mne.viz.topomap import _check_outlines, _find_topomap_coords

    # create head circle
    radius = 0.5
    ll = np.linspace(0, 2 * np.pi, 101)
    head_x = np.cos(ll) * radius
    head_y = np.sin(ll) * radius
    mask_outlines = np.c_[head_x, head_y]

    # create mask
    if 'right' in side:
        below_zero = mask_outlines[:, 0] < 0
        removed_len = below_zero.sum()
        filling = np.zeros((removed_len, 2))
        filling[:, 1] = np.linspace(0.5, -0.5, num=removed_len)
        mask_outlines[below_zero, :] = filling
    if side == 'right_frontal':
        below_zero = mask_outlines[:, 1] < 0
        removed_len = below_zero.sum()
        filling = np.zeros((removed_len, 2))
        filling[:, 0] = np.linspace(0., 0.5, num=removed_len)
        mask_outlines[below_zero, :] = filling

    head_pos = dict(center=(0., 0.))
    picks = range(len(info['ch_names']))
    pos = _find_topomap_coords(info, picks=picks)

    # TODO currently uses outlines='head', but should change later
    pos, outlines = _check_outlines(pos, outlines='head',
                                    head_pos=head_pos)
    outlines['mask_pos'] = (mask_outlines[:, 0], mask_outlines[:, 1])
    kwargs.update(dict(outlines=outlines, head_pos=head_pos))

    # scale pos to min - max of the circle (the 0.425 value was hand-picked)
    scale_x = 0.425 / pos[:, 0].max()
    scale_y = 0.425 / np.abs(pos[:, 1]).max()
    pos[:, 0] *= scale_x
    pos[:, 1] *= scale_y
    info = pos
    return info, kwargs
