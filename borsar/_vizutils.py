import numpy as np
import matplotlib.pyplot as plt


# - [ ] default source if not given
def add_colorbar_to_axis(axis, source, side='right', size='8%', pad=0.1):
    '''Add colorbar to given axis.
    FIXME!
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axis)
    cax = divider.append_axes(side, size=size, pad=pad)
    cbar = plt.colorbar(source, cax=cax)
    return cbar


def color_limits(data):
    '''Set color limits from data.

    Parameters
    ----------
    data : numpy array
        Data to set colorlimits for.

    Returns
    -------
    vmin : float
        Minimum value for the colormap.
    vmax : float
        Maximum value for the colormap.
    '''
    if data.dtype == 'bool':
        return 0., 1.

    vmax = np.abs([np.nanmin(data), np.nanmax(data)]).max()
    return -vmax, vmax
