import mne
from mne.utils import fill_doc, _check_sphere
from mne.viz.topomap import _plot_topomap

from packaging import version
mne_version = version.parse(mne.__version__)
has_0_21 = mne_version >= version.parse('0.21.dev0')
has_1_1 = mne_version >= version.parse('1.1.dev0')
has_1_2 = mne_version >= version.parse('1.1.dev0')

_BORDER_DEFAULT = 'mean'
_EXTRAPOLATE_DEFAULT = 'head' if has_0_21 else 'box'


# TODOs:     %(topomap_ch_type)s in docs did not work last time (was not
#                                present in docdict)
@fill_doc
def plot_topomap(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,
                 res=64, axes=None, names=None, show_names=False, mask=None,
                 mask_params=None, outlines='head', contours=6,
                 image_interp='bilinear', show=True, onselect=None,
                 extrapolate=_EXTRAPOLATE_DEFAULT, sphere=None,
                 border=_BORDER_DEFAULT, ch_type='eeg', cnorm=None):
    """Plot a topographic map as image.
    (docs as in mne)
    """
    sphere = _check_sphere(sphere)
    if has_1_2 or has_1_1:
        if image_interp == 'bilinear':
            image_interp = 'cubic'

    if has_1_2:
        return _plot_topomap(
            data, pos, vmin=vmin, vmax=vmax, cmap=cmap, sensors=sensors,
            res=res, axes=axes, names=names, mask=mask,
            mask_params=mask_params, outlines=outlines, contours=contours,
            image_interp=image_interp, show=show, onselect=onselect,
            extrapolate=extrapolate, sphere=sphere, border=border,
            ch_type=ch_type, cnorm=cnorm)
    elif has_1_1:
        return _plot_topomap(data, pos, vmin, vmax, cmap, sensors, res, axes,
                             names, show_names, mask, mask_params, outlines,
                             contours, image_interp, show, onselect,
                             extrapolate, sphere=sphere, border=border,
                             ch_type=ch_type)
    elif has_0_21:
        return _plot_topomap(data, pos, vmin, vmax, cmap, sensors, res, axes,
                             names, show_names, mask, mask_params, outlines,
                             contours, image_interp, show, onselect,
                             extrapolate, sphere=sphere, border=border,
                             ch_type=ch_type)
    else:
        head_pos = None
        return _plot_topomap(data, pos, vmin, vmax, cmap, sensors, res, axes,
                             names, show_names, mask, mask_params, outlines,
                             contours, image_interp, show, head_pos, onselect,
                             extrapolate, sphere=sphere, border=border)
