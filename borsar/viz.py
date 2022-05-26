from copy import copy
import numpy as np
import matplotlib.pyplot as plt

from .channels import get_ch_pos
from ._heatmap import heatmap
from ._vizutils import add_colorbar_to_axis, color_limits


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
        from._mne_modified import plot_topomap, has_new_mne

        self.info = info
        self.values = values

        # FIXME: should squeezing really be considered?
        self._squeezed = False
        if self.values.ndim > 1:
            if self.values.shape[1] == 1:
                self._squeezed = True
        else:
            self.values = self.values[:, np.newaxis]

        # default outlines='skirt' and extrapolate='head':
        if 'outlines' not in kwargs:
            if not has_new_mne:
                kwargs['outlines'] = 'skirt'
        if 'extrapolate' not in kwargs:
            if not has_new_mne:
                kwargs['extrapolate'] = 'head'

        self._check_axes(kwargs)
        if not has_new_mne:
            # TODO - these functions will have to be modified and made
            # compatible with new mne versions (>= 0.20)
            part = _infer_topo_part(info)
            info, kwargs = _construct_topo_part(info, part, kwargs)

        kwargs.update({'show': False})
        if 'axes' in kwargs:
            kwargs.pop('axes')

        # plot using mne's `plot_topomap`
        im, lines, chans = list(), list(), list()

        for topo_idx in range(self.n_topos):
            this_im, this_lines, interp = plot_topomap(
                self.values[:, topo_idx], info, axes=self.axes[topo_idx],
                **kwargs)

            # get channel objects and channel positions from topo
            this_chans, chan_pos = _extract_topo_channels(this_im.axes)

            im.append(this_im)
            lines.append(this_lines)
            chans.append(this_chans)

        self.chan_pos = chan_pos
        self.interpolator = interp
        self.mask_patch = this_im.get_clip_path()
        self.chan = chans if self.multi_axes else chans[0]
        self.img = im if self.multi_axes else im[0]
        self.lines = lines if self.multi_axes else lines[0]
        self.marks = [list() for idx in range(self.n_topos)]
        self.fig = im[0].figure if isinstance(im, list) else im.figure

        if not self.multi_axes:
            self.marks = self.marks[0]
            self.axes = self.axes[0]

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
        for topo in self:
            for line in topo.lines.collections:
                line.set_linestyle(*args, **kwargs)

    def set_linewidth(self, lw):
        '''
        Set contour lines linewidth.

        Parameters
        ----------
        lw : int | float
            Desired line width of the contour lines.
        '''
        for topo in self:
            for line in topo.lines.collections:
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

    def update(self, values):
        '''
        Change data presented in the topography. Useful especially in
        interactive applications as it should be faster than clearing the
        axis and drawing the different topography from scratch.

        Parameters
        ----------
        values : umpy array
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

        # FIXME - topo.update() is not particularily fast, profile later
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

        # reapply clipping to the countours
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
                figsize = (n_per_topo * self.n_topos, n_per_topo)
                fig, axes = plt.subplots(ncols=self.n_topos, figsize=figsize)
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

    # additional checks...
    y_limits = ch_pos[:, 1].min(), ch_pos[:, 1].max()
    y_range = y_limits[1] - y_limits[0]
    all_y_sufficiently_high = y_limits[0] > y_range * -0.3

    side = ''
    if all_x_above_0:
        side += 'right'
    elif (ch_pos[:, 0] <= 0.).all():
        side += 'left'

    if all_y_above_0 or all_y_sufficiently_high:
        side = 'frontal' if len(side) == 0 else '_'.join([side, 'frontal'])

    side = None if len(side) == 0 else side
    return side


def _construct_topo_part(info, part, kwargs):
    """Mask part of the topography."""
    from mne.viz.topomap import _find_topomap_coords

    # project channels to 2d
    picks = range(len(info['ch_names']))
    pos = _find_topomap_coords(info, picks=picks)

    # create head circle and other shapes
    # -----------------------------------
    use_skirt = kwargs.get('outlines', None) == 'skirt'
    radius = np.pi / 2 if use_skirt else max(np.linalg.norm(pos, axis=1).max(),
                                             np.pi / 2)
    ll = np.linspace(0, 2 * np.pi, 101)
    head_x = np.cos(ll) * radius
    head_y = np.sin(ll) * radius

    nose_x = np.array([0.18, 0, -0.18]) * radius
    nose_y = np.array([radius - .004, radius * 1.15, radius - .004])
    ear_x = np.array([.497, .510, .518, .5299, .5419, .54, .547,
                      .532, .510, .489]) * (radius / 0.5)
    ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932,
                      -.1313, -.1384, -.1199]) * (radius / 0.5)

    outlines_dict = dict(head=(head_x, head_y), nose=(nose_x, nose_y),
                         ear_left=(ear_x, ear_y),
                         ear_right=(-ear_x, ear_y))
    outlines_dict['autoshrink'] = False

    # create mask properties
    mask_outlines = np.c_[head_x, head_y]
    head_pos = dict(center=(0., 0.))
    # what does head_pos['scale'] do?

    mask_scale = (max(1.25, np.linalg.norm(pos, axis=1).max() / radius)
                  if use_skirt else 1)
    mask_outlines *= mask_scale
    outlines_dict['clip_radius'] = (mask_scale * radius,) * 2
    outlines_dict['mask_pos'] = (mask_outlines[:, 0], mask_outlines[:, 1])

    # modify mask pizza-style
    # -----------------------
    if isinstance(part, str):
        if 'right' in part:
            lowest = pos[:, 0].min() - radius * 0.2
            below_lowest = mask_outlines[:, 0] < lowest
            min_x = mask_outlines[~below_lowest, 0].min()
            mask_outlines[below_lowest, 0] = min_x
        elif 'left' in part:
            highest = pos[:, 0].max() + radius * 0.2
            above_highest = mask_outlines[:, 0] > highest
            max_x = mask_outlines[~above_highest, 0].max()
            mask_outlines[above_highest, 0] = max_x

        if 'frontal' in part:
            lowest_y = pos[:, 1].min() - radius * 0.2
            below_lowest = mask_outlines[:, 1] < lowest_y
            min_y = mask_outlines[~below_lowest, 1].min()
            mask_outlines[below_lowest, 1] = min_y

    outlines_dict['mask_pos'] = (mask_outlines[:, 0], mask_outlines[:, 1])
    kwargs.update(dict(outlines=outlines_dict, head_pos=head_pos))
    return pos, kwargs
