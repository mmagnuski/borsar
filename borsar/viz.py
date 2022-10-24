from copy import copy
import numpy as np
import matplotlib.pyplot as plt

from .channels import get_ch_pos
from ._heatmap import heatmap
from ._vizutils import add_colorbar_to_axis, color_limits


# * add possibility to index topo object
# * loop + tp.marks[0].set_markersize(4) -> method?

# CONSIDER:
# - [ ] save vmin and vmax and let modify?
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

    Examples
    --------
    topo = Topo(values, info, axis=ax)
    topo.remove_levels(0)
    topo.solid_lines()
    topo.set_linewidth(1.5)
    topo.mark_channels([4, 5, 6], markerfacecolor='r', markersize=12)
    '''

    def __init__(self, values, info, side=None, **kwargs):
        from ._mne_020_modified import plot_topomap

        self.info = info
        self.values = values

        # FIXME: should squeezing really be considered?
        self._squeezed = False
        if self.values.ndim > 1:
            if self.values.shape[1] == 1:
                self._squeezed = True
        else:
            self.values = self.values[:, np.newaxis]

        # default extrapolate='local'
        if 'extrapolate' not in kwargs:
            kwargs['extrapolate'] = 'local'

        self._check_axes(kwargs)

        kwargs.update({'show': False})
        if 'axes' in kwargs:
            kwargs.pop('axes')

        # plot using mne's `plot_topomap`
        im, lines, chans, head = list(), list(), list(), list()

        for topo_idx in range(self.n_topos):
            this_im, this_lines, interp = plot_topomap(
                self.values[:, topo_idx], info, axes=self.axes[topo_idx],
                **kwargs)

            # get channel objects and channel positions from topo
            this_chans, chan_pos = _extract_topo_channels(this_im.axes)
            head_lines = self.axes[topo_idx].findobj(plt.Line2D)[:4]

            im.append(this_im)
            lines.append(this_lines)
            chans.append(this_chans)
            head.append(head_lines)

        self.chan_pos = chan_pos
        self.interpolator = interp
        self.mask_patch = this_im.get_clip_path()
        self.chan = chans if self.multi_axes else chans[0]
        self.img = im if self.multi_axes else im[0]
        self.head = head if self.multi_axes else head[0]
        self.lines = lines if self.multi_axes else lines[0]
        self.marks = [list() for idx in range(self.n_topos)]
        self.fig = im[0].figure if isinstance(im, list) else im.figure

        if not self.multi_axes:
            self.marks = self.marks[0]
            self.axes = self.axes[0]

    def remove_levels(self, lvl):
        '''
        Remove contour lines at specified levels.

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
        for topo in self:
            for line in topo.lines.collections:
                line.set_linestyle(*args, **kwargs)

        # changing linestyle to solid often goes without changes in interactive
        # mode, we have to force a redraw:
        self.fig.canvas.draw()

    # TODO: keywords: contours=x, outline=y,
    def set_linewidth(self, contours=None, outlines=None):
        '''
        Set contour lines line width.

        Parameters
        ----------
        contours : None | int | float
            Desired line width of the contour lines. The contour line width
            is not changed if ``None``. Defaults to ``None``.
        outline : None | int | float
            Desired line width of the head outline. The outline line width is
            not changed if ``None``. Defaults to ``None``.
        '''
        for topo in self:
            if contours is not None:
                for line in topo.lines.collections:
                    line.set_linewidths(contours)
            if outlines is not None:
                for line in topo.head:
                    line.set_linewidth(outlines)

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
            different selection of channels highlighted in each, iterate
            through topos with a for loop and ``.mark_channels()`` for each
            individually.
        **kwargs : additional keyword arguments
            Any additional keyword arguments are passed as arguments to
            `plt.plot`. It is useful for defining marker properties like
            `marker`, `markerfacecolor`, `markeredgecolor`, `linewidth` or
            `markersize`.
        '''
        default_marker = dict(marker='o', markerfacecolor='w', markersize=8,
                              markeredgecolor='k', linewidth=0)
        default_marker.update(kwargs)

        # mark channels and save marks in `self.marks` list
        for topo in self:
            this_marks = topo.axes.plot(
                topo.chan_pos[chans, 0], topo.chan_pos[chans, 1],
                **default_marker)[0]
            topo.marks.append(this_marks)

    def zoom(self, xlim=None, ylim=None):
        '''Change the x and y limits of the topography image.

        Takes care of line clipping after changing limits.

        Parameters
        ----------
        xlim : tuple
            New x-axis limits. Defaults to ``None``, which does not change the
            x-axis limits.
        ylim : tuple
            New y-axis limits. Defaults to ``None``, which does not change the
            y-axis limits.
        '''
        if xlim is None and ylim is None:
            return

        for topo in self:
            ax = topo.axes
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            if hasattr(topo, 'head'):
                [line.set_clip_on(True) for line in topo.head]
            [line.set_clip_on(True) for line in topo.lines]

    def update(self, values):
        '''
        Change data presented in the topography. Useful especially in
        interactive applications as it should be faster than clearing the
        axis and drawing the different topography from scratch.

        Parameters
        ----------
        values : numpy array
            Values to plot topographically. Has to be of shape
            ``(n_channels,)``. If ``Topo`` contains multiple topographies
            each should be updated independently by looping through the
            ``Topo`` object and using ``.update()`` on each element.

        Examples
        --------
        # single topography:
        topo = Topo(values, info)
        topo.update(other_values)

        # multiple topographies
        topos = Topo(values2d, info)

        for idx, this_topo in enumerate(topos):
            this_topo.update(other_values[:, idx])
        '''

        # .update() works only for single-axis Topo
        if self.multi_axes:
            raise NotImplementedError('.update() is not implemented for multi-'
                                      'axis Topo. To update the data in such'
                                      ' case you should use a for loop through'
                                      ' Topo and use .update() on each element'
                                      ' independently.')

        # FIXME - topo.update() is not particularly fast, profile later
        interp = self.interpolator
        new_image = interp.set_values(values)()
        self.img.set_data(new_image)

        # update contour lines by removing the old ...
        for l in self.lines.collections:
            l.remove()

        # ... and drawing new ones
        # FIXME - make line properties (lw) remembered
        linewidth, n_contours = 1, 6
        self.lines = self.axes.contour(interp.Xi, interp.Yi, new_image,
                                       n_contours, colors='k',
                                       linewidths=linewidth / 2.)

        # reapply clipping to the contours
        patch = self.mask_patch
        for l in self.lines.collections:
            l.set_clip_path(patch)

    def __len__(self):
        '''Return number of topomaps in Topo.'''
        return self.n_topos

    def __iter__(self):
        '''Initialize iteration.'''
        self._current = 0
        return self

    def __next__(self):
        '''
        Get next topomap in iteration. Allows to do things like:
        >>> for tp in topo:
        >>>     tp.update(values)
        '''
        if self._current >= len(self):
            raise StopIteration

        topo = copy(self)
        topo.n_topos = 1
        topo.multi_axes = False
        topo._current = 0

        if self.multi_axes:
            topo.img = topo.img[self._current]
            topo.head = topo.head[self._current]
            topo.chan = topo.chan[self._current]
            topo.lines = topo.lines[self._current]
            topo.marks = topo.marks[self._current]
            topo.axes = topo.axes[self._current]

        self._current += 1
        return topo

    def _check_axes(self, kwargs):
        '''Handle axes checking for Topo.'''

        # handle multiple axes
        self.multi_axes = self.values.ndim > 1 and self.values.shape[1] > 1
        if self.multi_axes:
            self.n_topos = self.values.shape[1]
        else:
            self.n_topos = 1

        has_axis = 'axes' in kwargs.keys() and kwargs['axes'] is not None
        if has_axis:
            # axis was passed, check if valid
            axes = kwargs['axes']
            if self.multi_axes:
                # multiple topos and axes were passed, check if axes correct
                from mne.viz.utils import _validate_if_list_of_axes
                _validate_if_list_of_axes(axes, obligatory_len=self.n_topos)
                self.axes = axes
            else:
                # one topo and axes were passed, should check axes
                if self._squeezed and isinstance(axes, list):
                    axes = axes[0]
                    kwargs['axes'] = axes
                self.axes = [axes]
        else:
            # no axes passed, create figure and axes
            if self.multi_axes:
                # create a row of topographies
                n_per_topo = 2.5
                fig_size = (n_per_topo * self.n_topos, n_per_topo)
                fig, axes = plt.subplots(ncols=self.n_topos, figsize=fig_size)
                self.axes = axes.tolist()
            else:
                fig, axes = plt.subplots()
                self.axes = [axes]


def _extract_topo_channels(ax):
    '''
    Extract channel positions from mne topoplot axis.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis containing the topoplot.

    Returns
    -------
    chans : ``matplotlib.patches.Circle`` or
        ``matplotlib.collections.PathCollection`` matplotlib object
        representing channels. Some older mne versions use ``plt.scatter`` to
        draw channels so the channels are marked with ``mpl.patches.Circle``.
        At other times ``mpl.collections.PathCollection`` is being used.
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
            n_points = [path.get_offsets().shape[0]
                        for path in path_collection]
            chan_idx = np.argmax(n_points)
            chans = path_collection[chan_idx]
            chan_pos = chans.get_offsets()
        else:
            msg = ('Could not find matplotlib objects representing channels. '
                   'Looked for `matplotlib.patches.Circle` and `matplotlib.'
                   'collections.PathCollection`.')
            raise RuntimeError(msg)

    return chans, chan_pos
