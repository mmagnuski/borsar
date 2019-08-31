import numpy as np
import matplotlib.pyplot as plt

from .channels import get_ch_pos


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

    Example
    -------
    topo = Topo(values, info, axis=ax)
    topo.remove_levels(0)
    topo.solid_lines()
    topo.set_linewidth(1.5)
    topo.mark_channels([4, 5, 6], markerfacecolor='r', markersize=12)
    '''

    def __init__(self, values, info, side=None, **kwargs):
        from._mne_modified import plot_topomap

        self.info = info
        self.values = values

        # FIXME: should squeezing really be considered?
        self._squeezed = False
        if self.values.ndim > 1:
            if self.values.shape[1] == 1:
                self._squeezed = True
        else:
            self.values = self.values[:, np.newaxis]

        self._check_axes(kwargs)
        part = _infer_topo_part(info)
        info, kwargs = _construct_topo_part(info, part, kwargs)

        # plot using mne's `plot_topomap`
        im, lines, chans = list(), list(), list()
        self.marks = [list() for idx in range(self.n_topos)]

        kwargs.update({'show': False})
        if 'axes' in kwargs:
            kwargs.pop('axes')

        for topo_idx in range(self.n_topos):
            this_im, this_lines, interp, patch = plot_topomap(
                self.values[:, topo_idx], info, axes=self.axes[topo_idx],
                **kwargs)
            # get channel objects and channel positions from topo
            this_chans, chan_pos = _extract_topo_channels(this_im.axes)

            im.append(this_im)
            lines.append(this_lines)
            chans.append(this_chans)

        self.chan_pos = chan_pos
        self.img = im if self.multi_axes else im[0]
        self.lines = lines if self.multi_axes else lines[0]
        self.interpolator = interp
        self.fig = im[0].figure if isinstance(im, list) else im.figure
        self.mask_patch = patch
        if not self.multi_axes:
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
        n_channels = len(self.chan_pos)
        iter_types = (list, tuple, np.ndarray)
        n_topos = len(self.axes) if isinstance(self.axes, iter_types) else 1
        iter_marks = (self.marks if n_topos > 1 else [self.marks])
        iter_axes = (self.axes if n_topos > 1 else [self.axes])

        # make sure channel selection is iterable
        if (isinstance(chans, (list, tuple)) and not
            isinstance(chans[0], iter_types)):
            chans = [chans]
        elif isinstance(chans, np.ndarray):
            chans = (np.tile(chans, (n_topos, 1)) if chans.ndim == 1
                     else chans.T if chans.shape[0] == n_channels
                     else chans)

        for ax, marks, msk in zip(iter_axes, iter_marks, chans):
            this_marks = ax.plot(
                self.chan_pos[msk, 0], self.chan_pos[msk, 1], **default_marker)
            marks.append(this_marks)

    def update(self, values):
        '''FIXME.'''

        # FIXME - topo.update() is not particularily fast, will need to profile
        # FIXME - fix for situation with multiple topos...
        interp = self.interpolator
        new_image = interp.set_values(values)()
        self.img.set_data(new_image)

        # update contour lines by removing old ...
        for l in self.lines.collections:
            l.remove()

        # ... and drawing new ones
        # FIXME - make line properties (lw) remembered
        linewidth = 1
        n_contours = 6
        self.lines = self.axes.contour(interp.Xi, interp.Yi, new_image,
                                       n_contours, colors='k',
                                       linewidths=linewidth / 2.)

        # apply clipping to the countours
        patch = self.mask_patch
        for l in self.lines.collections:
            l.set_clip_path(patch)

    def _check_axes(self, kwargs):
        # handle multiple axes
        self.multi_axes = self.values.ndim > 1 and self.values.shape[1] > 1
        if self.multi_axes:
            self.n_topos = self.values.shape[1]
        else:
            self.n_topos = 1

        has_axis = 'axes' in kwargs.keys()
        if has_axis:
            axes = kwargs['axes']
            if self.multi_axes:
                # multiple topos and axes were passed, check if axes correct
                from mne.viz.utils import _validate_if_list_of_axes
                _validate_if_list_of_axes(axes, obligatory_len=self.n_topos)
                plt.sca(axes[0])  # FIXME - this may not be needed in future
            else:
                # one topo and axes were passed, should check axes
                if self._squeezed and isinstance(axes, list):
                    axes = axes[0]
                    kwargs['axes'] = axes
                plt.sca(axes)  # FIXME - this may not be needed in future
                self.axes = [axes]
        else:
            if self.multi_axes:
                # create a row of topographies
                fig, axes = plt.subplots(ncols=self.n_topos)
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

    outlines_dict['mask_pos'] = (mask_outlines[:, 0], mask_outlines[:, 1])
    kwargs.update(dict(outlines=outlines_dict, head_pos=head_pos))
    return pos, kwargs
