import numpy as np

import mne
from mne.utils import fill_doc, _check_sphere
from mne.viz.topomap import _plot_topomap

from packaging import version
has_0_20_5 = version.parse(mne.__version__) > version.parse('0.20.4')
_BORDER_DEFAULT = 'mean'
_EXTRAPOLATE_DEFAULT = 'head' if has_0_20_5 else 'box'


# TODOs:     %(topomap_ch_type)s in docs did not work last time (was not
#                                present in docdict)
@fill_doc
def plot_topomap(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,
                 res=64, axes=None, names=None, show_names=False, mask=None,
                 mask_params=None, outlines='head',
                 contours=6, image_interp='bilinear', show=True,
                 onselect=None, extrapolate=_EXTRAPOLATE_DEFAULT,
                 sphere=None, border=_BORDER_DEFAULT,
                 ch_type='eeg'):
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
    %(topomap_outlines)s
    contours : int | array of float
        The number of contour lines to draw. If 0, no contours will be drawn.
        If an array, the values represent the levels for the contours. The
        values are in µV for EEG, fT for magnetometers and fT/m for
        gradiometers. Defaults to 6.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    show : bool
        Show figure if True.
    onselect : callable | None
        Handle for a function that is called when the user selects a set of
        channels by rectangle selection (matplotlib ``RectangleSelector``). If
        None interactive selection is disabled. Defaults to None.
    %(topomap_extrapolate)s

        .. versionadded:: 0.18
    %(topomap_sphere)s
    %(topomap_border)s

    Returns
    -------
    im : matplotlib.image.AxesImage
        The interpolated data.
    cn : matplotlib.contour.ContourSet
        The fieldlines.
    """
    sphere = _check_sphere(sphere)
    if has_0_20_5:
        return _plot_topomap(data, pos, vmin, vmax, cmap, sensors, res, axes,
                             names, show_names, mask, mask_params, outlines,
                             contours, image_interp, show, onselect,
                             extrapolate, sphere=sphere, border=border,
                             ch_type=ch_type)
    else:
        head_pos = None
        return _plot_topomap(data, pos, vmin, vmax, cmap, sensors, res, axes,
                             names, show_names, mask, mask_params, outlines,
                             contours, image_interp, show,
                             head_pos, onselect, extrapolate,
                             sphere=sphere, border=border)
