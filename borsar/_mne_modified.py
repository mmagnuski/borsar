import warnings
import numpy as np
from mne import Info
from mne.io.pick import _pick_data_channels, pick_info, channel_type
from mne.defaults import _handle_default
from mne.viz.utils import plt_show, _setup_vmin_vmax
from mne.viz.topomap import (_check_outlines, _prepare_topomap, _autoshrink,
                             _plot_sensors, _draw_outlines, _GridData,
                             _find_topomap_coords)


def plot_topomap(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,
                 res=64, axes=None, names=None, show_names=False, mask=None,
                 mask_params=None, outlines='head',
                 contours=6, image_interp='bilinear', show=True,
                 head_pos=None, onselect=None, extrapolate='box'):
    """Plot a topographic map as image.
    Parameters
    ----------
    data : array, shape (n_chan,)
        The data values to plot.
    pos : array, shape (n_chan, 2) | instance of Info
        Location information for the data points(/channels).
        If an array, for each data point, the x and y coordinates.
        If an Info object, it must contain only one data type and
        exactly `len(data)` data channels, and the x/y coordinates will
        be inferred from this Info object.
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data). Defaults to None.
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. If callable, the output
        equals vmax(data). Defaults to None.
    cmap : matplotlib colormap | None
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r'.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses). If True (default), circles
        will be used.
    res : int
        The resolution of the topomap image (n pixels along each side).
    axes : instance of Axes | None
        The axes to plot to. If None, the current axes will be used.
    names : list | None
        List of channel names. If None, channel names are not plotted.
    show_names : bool | callable
        If True, show channel names on top of the map. If a callable is
        passed, channel names will be formatted using the callable; e.g., to
        delete the prefix 'MEG ' from all channel names, pass the function
        lambda x: x.replace('MEG ', ''). If `mask` is not None, only
        significant sensors will be shown.
        If `True`, a list of names must be provided (see `names` keyword).
    mask : ndarray of bool, shape (n_channels, n_times) | None
        The channels to be marked as significant at a given time point.
        Indices set to `True` will be considered. Defaults to None.
    mask_params : dict | None
        Additional plotting parameters for plotting significant sensors.
        Default (None) equals::
           dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=4)
    outlines : 'head' | 'skirt' | dict | None
        The outlines to be drawn. If 'head', the default head scheme will be
        drawn. If 'skirt' the head scheme will be drawn, but sensors are
        allowed to be plotted outside of the head circle. If dict, each key
        refers to a tuple of x and y positions, the values in 'mask_pos' will
        serve as image mask, and the 'autoshrink' (bool) field will trigger
        automated shrinking of the positions due to points outside the outline.
        Alternatively, a matplotlib patch object can be passed for advanced
        masking options, either directly or as a function that returns patches
        (required for multi-axes plots). If None, nothing will be drawn.
        Defaults to 'head'.
    contours : int | array of float
        The number of contour lines to draw. If 0, no contours will be drawn.
        If an array, the values represent the levels for the contours. The
        values are in uV for EEG, fT for magnetometers and fT/m for
        gradiometers. Defaults to 6.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    show : bool
        Show figure if True.
    head_pos : dict | None
        If None (default), the sensors are positioned such that they span
        the head circle. If dict, can have entries 'center' (tuple) and
        'scale' (tuple) for what the center and scale of the head should be
        relative to the electrode locations.
    onselect : callable | None
        Handle for a function that is called when the user selects a set of
        channels by rectangle selection (matplotlib ``RectangleSelector``). If
        None interactive selection is disabled. Defaults to None.
    extrapolate : str
        If 'box' (default) extrapolate to four points placed to form a square
        encompassing all data points, where each side of the square is three
        times the range of the data in the respective dimension. If 'head'
        extrapolate to the edges of the head circle (or to the edges of the
        skirt if ``outlines='skirt'``). If 'local' extrapolate only to nearby
        points (approximately to points closer than median inter-electrode
        distance).
        .. versionadded:: 0.18
    Returns
    -------
    im : matplotlib.image.AxesImage
        The interpolated data.
    cn : matplotlib.contour.ContourSet
        The fieldlines.
    """
    return _plot_topomap(data, pos, vmin, vmax, cmap, sensors, res, axes,
                         names, show_names, mask, mask_params, outlines,
                         contours, image_interp, show,
                         head_pos, onselect, extrapolate)


def _plot_topomap(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,
                  res=64, axes=None, names=None, show_names=False, mask=None,
                  mask_params=None, outlines='head',
                  contours=6, image_interp='bilinear', show=True,
                  head_pos=None, onselect=None, extrapolate='box'):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    data = np.asarray(data)

    if isinstance(pos, Info):  # infer pos from Info object
        picks = _pick_data_channels(pos)  # pick only data channels
        pos = pick_info(pos, picks)

        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = {channel_type(pos, idx)
                   for idx, _ in enumerate(pos["chs"])}
        info_help = ("Pick Info with e.g. mne.pick_info and "
                     "mne.io.pick.channel_indices_by_type.")
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. "
                             + info_help)
        elif len(pos["chs"]) != data.shape[0]:
            raise ValueError("Number of channels in the Info object and "
                             "the data array does not match. " + info_help)
        else:
            ch_type = ch_type.pop()

        if any(type_ in ch_type for type_ in ('planar', 'grad')):
            # deal with grad pairs
            from mne.channels.layout import (_merge_grad_data, find_layout,
                                             _pair_grad_sensors)
            picks, pos = _pair_grad_sensors(pos, find_layout(pos))
            data = _merge_grad_data(data[picks]).reshape(-1)
        else:
            picks = list(range(data.shape[0]))
            pos = _find_topomap_coords(pos, picks=picks)

    if data.ndim > 1:
        raise ValueError("Data needs to be array of shape (n_sensors,); got "
                         "shape %s." % str(data.shape))

    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = ("Electrode positions should be specified as a 2D array with "
                "shape (n_channels, 2). Each row in this matrix contains the "
                "(x, y) position of an electrode.")
    if pos.ndim != 2:
        error = ("{ndim}D array supplied as electrode positions, where a 2D "
                 "array was expected").format(ndim=pos.ndim)
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = ("The supplied electrode positions matrix contains 3 columns. "
                 "Are you trying to specify XYZ coordinates? Perhaps the "
                 "mne.channels.create_eeg_layout function is useful for you.")
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)

    if len(data) != len(pos):
        raise ValueError("Data and pos need to be of same length. Got data of "
                         "length %s, pos of length %s" % (len(data), len(pos)))

    norm = min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)
    if cmap is None:
        cmap = 'Reds' if norm else 'RdBu_r'

    pos, outlines = _check_outlines(pos, outlines, head_pos)
    assert isinstance(outlines, dict)

    ax = axes if axes else plt.gca()
    _prepare_topomap(pos, ax)

    _use_default_outlines = any(k.startswith('head') for k in outlines)

    if _use_default_outlines:
        # prepare masking
        _autoshrink(outlines, pos, res)

    mask_params = _handle_default('mask_params', mask_params)

    # find mask limits
    xlim = np.inf, -np.inf,
    ylim = np.inf, -np.inf,
    mask_ = np.c_[outlines['mask_pos']]
    xmin, xmax = (np.min(np.r_[xlim[0], mask_[:, 0]]),
                  np.max(np.r_[xlim[1], mask_[:, 0]]))
    ymin, ymax = (np.min(np.r_[ylim[0], mask_[:, 1]]),
                  np.max(np.r_[ylim[1], mask_[:, 1]]))

    # interpolate the data, we multiply clip radius by 1.06 so that pixelated
    # edges of the interpolated image would appear under the mask
    head_radius = (None if extrapolate == 'local' else
                   outlines['clip_radius'][0] * 1.06)
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    interp = _GridData(pos, extrapolate, head_radius).set_values(data)
    Zi = interp.set_locations(Xi, Yi)()

    # plot outline
    patch_ = None
    if 'patch' in outlines:
        patch_ = outlines['patch']
        patch_ = patch_() if callable(patch_) else patch_
        patch_.set_clip_on(False)
        ax.add_patch(patch_)
        ax.set_transform(ax.transAxes)
        ax.set_clip_path(patch_)
    if _use_default_outlines:
        from matplotlib import patches
        patch_ = patches.Ellipse((0, 0),
                                 2 * outlines['clip_radius'][0],
                                 2 * outlines['clip_radius'][1],
                                 clip_on=True,
                                 transform=ax.transData)

    # plot interpolated map
    im = ax.imshow(Zi, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   aspect='equal', extent=(xmin, xmax, ymin, ymax),
                   interpolation=image_interp)

    # This tackles an incomprehensible matplotlib bug if no contours are
    # drawn. To avoid rescalings, we will always draw contours.
    # But if no contours are desired we only draw one and make it invisible.
    linewidth = mask_params['markeredgewidth']
    no_contours = False
    if isinstance(contours, (np.ndarray, list)):
        pass  # contours precomputed
    elif contours == 0:
        contours, no_contours = 1, True
    if (Zi == Zi[0, 0]).all():
        cont = None  # can't make contours for constant-valued functions
    else:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore')
            cont = ax.contour(Xi, Yi, Zi, contours, colors='k',
                              linewidths=linewidth / 2.)
    if no_contours and cont is not None:
        for col in cont.collections:
            col.set_visible(False)

    if patch_ is not None:
        im.set_clip_path(patch_)
        if cont is not None:
            for col in cont.collections:
                col.set_clip_path(patch_)

    pos_x, pos_y = pos.T
    if sensors is not False and mask is None:
        _plot_sensors(pos_x, pos_y, sensors=sensors, ax=ax)
    elif sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)
        idx = np.where(~mask)[0]
        _plot_sensors(pos_x[idx], pos_y[idx], sensors=sensors, ax=ax)
    elif not sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)

    if isinstance(outlines, dict):
        _draw_outlines(ax, outlines)

    if show_names:
        if names is None:
            raise ValueError("To show names, a list of names must be provided"
                             " (see `names` keyword).")
        if show_names is True:
            def _show_names(x):
                return x
        else:
            _show_names = show_names
        show_idx = np.arange(len(names)) if mask is None else np.where(mask)[0]
        for ii, (p, ch_id) in enumerate(zip(pos, names)):
            if ii not in show_idx:
                continue
            ch_id = _show_names(ch_id)
            ax.text(p[0], p[1], ch_id, horizontalalignment='center',
                    verticalalignment='center', size='x-small')

    if onselect is not None:
        ax.RS = RectangleSelector(ax, onselect=onselect)
    plt_show(show)
    return im, cont, interp, patch_
