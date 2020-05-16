import numpy as np
import matplotlib.pyplot as plt

from ._vizutils import add_colorbar_to_axis, color_limits


def _masked_image(img, mask=None, alpha=0.75, mask_color=(0.5, 0.5, 0.5),
                  axis=None, **imshow_kwargs):
    '''Create imshow image with alpha mask.'''
    defaults = {'interpolation': 'none', 'origin': 'lower'}
    defaults.update(imshow_kwargs)

    if axis is None:
        fig, axis = plt.subplots()

    # plot images
    main_img = axis.imshow(img, **defaults)
    if mask is not None:
        mask_img = _add_image_mask(mask, alpha=alpha, mask_color=mask_color,
                                   axis=axis, **defaults)
        return main_img, mask_img
    else:
        return main_img


def _add_image_mask(mask, alpha=0.75, mask_color=(0.5, 0.5, 0.5),
                    axis=None, **imshow_kwargs):
    if axis is None:
        axis = plt.gca()
    else:
        # update imshow_kwargs so that image does not change
        if 'aspect' not in imshow_kwargs:
            imshow_kwargs['aspect'] = axis.get_aspect()
        if 'extent' not in imshow_kwargs:
            imshow_kwargs['extent'] = axis.images[0].get_extent()
        if 'origin' not in imshow_kwargs:
            imshow_kwargs['origin'] = axis.images[0].origin

    # create RGBA mask:
    mask_img = np.array(list(mask_color) + [0.]).reshape((1, 1, 4))
    mask_img = np.tile(mask_img, list(mask.shape) + [1])
    mask_img[~mask, -1] = alpha

    # plot images
    return axis.imshow(mask_img, **imshow_kwargs)


# - [ ] multiple masks, multiple outline_colors, multiple alpha?
def heatmap(array, mask=None, axis=None, x_axis=None, y_axis=None,
            outlines=False, colorbar=True, cmap='RdBu_r', alpha=0.75,
            vmin=None, vmax=None, line_kwargs=dict(), **kwargs):
    '''Plot heatmap with defaults meaningful for big heatmaps like
    time-frequency representations.

    Parameters
    ----------
    array : 2d numpy array
        The array to be plotted as heatmap.
    mask : 2d boolean array
        Matrix specifying which pixels to unmask. Masking is done with
        transparency.
    axis : matplotlib axis
        Axis to draw in.
    x_axis : 1d array
        X axis coordinates - 1d array of x axis bin names.
    y_axis : 1d array
        Y axis coordinates - 1d array of y axis bin names.
    outlines : boolean
        whether to draw outlines of the clusters defined by the mask.
    colorbar : boolean
        Whether to add a colorbar to the image.
    cmap : str
        Colormap to use. Defaults to ``'RdBu_r'``.
    alpha : float
        Mask transparency.
    vmin : float | None
        Minimum value for the colormap.
    vmax : float | None
        Maximum value for the colormap.
    line_kwargs : dict
        Dictionary of additional parameters for outlines.

    Returns
    -------
    axis : maplotlib axis
        The axis drawn to.
    cbar : matplotlib colorbar
        The handle to the colorbar.
    '''
    if vmin is None and vmax is None:
        vmin, vmax = color_limits(array)
    n_rows, n_cols = array.shape

    x_axis = np.arange(n_cols) if x_axis is None else x_axis
    y_axis = np.arange(n_rows) if y_axis is None else y_axis

    # set extents
    x_step = np.diff(x_axis)[0]
    y_step = np.diff(y_axis)[0]
    ext = [*(x_axis[[0, -1]] + [-x_step / 2, x_step / 2]),
           *(y_axis[[0, -1]] + [-y_step / 2, y_step / 2])]


    mask_ = mask.any(axis=0) if mask.ndim == 3 else mask
    out = _masked_image(array, mask=mask_, vmin=vmin, vmax=vmax,
                        cmap=cmap, aspect='auto', extent=ext,
                        interpolation='nearest', origin='lower',
                        axis=axis, alpha=alpha, **kwargs)

    img = out if mask is None else out[0]

    # add outlines if necessary
    if outlines:
        if 'color' not in line_kwargs.keys():
            line_kwargs['color'] = 'w'

        mask = mask[np.newaxis, :] if mask.ndim == 2 else mask
        n_masks = mask.shape[0]

        if not isinstance(line_kwargs['color'], list):
            line_kwargs['color'] = [line_kwargs['color']] * n_masks

        this_line_kwargs = line_kwargs.copy()
        for mask_idx in range(n_masks):
            this_line_kwargs['color'] = line_kwargs['color'][mask_idx]
            outlines = _create_cluster_contour(mask[mask_idx], extent=ext)
            for x_line, y_line in outlines:
                img.axes.plot(x_line, y_line, **this_line_kwargs)

    if colorbar:
        cbar = add_colorbar_to_axis(img.axes, img)
        # cbar.set_label('t values')
        return img.axes, cbar
    else:
        return img.axes


# TODOs:
# - [x] docstring
# - [ ] rename to create_contour?
# - [ ] check timing and compare against numba version
#       numba would require some changes, we'd have to remove all the dicts
#
# separate cluter_contour?
# - [ ] cluster mode (returns a list or dict mapping cluster ids to list of
#       cluster contours) - so that each cluster can be marked by a different
#       color.
def _create_cluster_contour(mask, extent=None):
    '''Create contour lines for clusters in a boolean matrix.

    Parameters
    ----------
    mask : numpy array
        Two dimensional boolean numpy array.
    extent : iterable, optional
        The extents of the image: ``[x_min, x_max, y_min, y_max]`` - just as
        the extent argument in ``matplotlib.pyplot.imshow``.

    Returns
    -------
    contours : list
        List of contours, one per cluster. Each controur is a list of two numpy
        arrays: ``[x_contours, y_contours]``.
    '''
    from scipy.ndimage import correlate

    orig_mask_shape = mask.shape
    mask_int = np.pad(mask.astype('int'), ((1, 1), (1, 1)), 'constant')
    kernels = {'upper': np.array([[-1], [1], [0]]),
               'lower': np.array([[0], [1], [-1]]),
               'left': np.array([[-1, 1, 0]]),
               'right': np.array([[0, 1, -1]])}
    lines = {k: (correlate(mask_int, v) == 1).astype('int')
             for k, v in kernels.items()}

    search_order = {'upper': ['right', 'left', 'upper'],
                    'right': ['lower', 'upper', 'right'],
                    'lower': ['left', 'right', 'lower'],
                    'left': ['upper', 'lower', 'left']}
    movement_direction = {'upper': [0, 1], 'right': [1, 0],
                          'lower': [0, -1], 'left': [-1, 0]}
    search_modifiers = {'upper_left': [-1, 1], 'right_upper': [1, 1],
                        'lower_right': [1, -1], 'left_lower': [-1, -1]}
    finish_modifiers = {'upper': [-0.5, 0.5], 'right': [0.5, 0.5],
                        'lower': [0.5, -0.5], 'left': [-0.5, -0.5]}

    # current index - upmost upper line
    upper_lines = np.where(lines['upper'])
    outlines = list()

    while len(upper_lines[0]) > 0:
        current_index = np.array([x[0] for x in upper_lines])
        closed_shape = False
        current_edge = 'upper'
        edge_points = [tuple(current_index + [-0.5, -0.5])]
        direction = movement_direction[current_edge]

        while not closed_shape:
            new_edge = None
            ind = tuple(current_index)

            # check the next edge
            for edge in search_order[current_edge]:
                modifier = '_'.join([current_edge, edge])
                has_modifier = modifier in search_modifiers
                if has_modifier:
                    modifier_value = search_modifiers[modifier]
                    test_ind = tuple(current_index + modifier_value)
                else:
                    test_ind = ind

                if lines[edge][test_ind] == 1:
                    new_edge = edge
                    lines[current_edge][ind] = -1
                    break
                elif lines[edge][test_ind] == -1:  # -1 means 'visited'
                    closed_shape = True
                    new_edge = 'finish'
                    lines[current_edge][ind] = -1
                    break

            if not new_edge == current_edge:
                edge_points.append(tuple(
                    current_index + finish_modifiers[current_edge]))
                direction = modifier_value if has_modifier else [0, 0]
                current_edge = new_edge
            else:
                direction = movement_direction[current_edge]

            current_index += direction

        # TODO: this should be done at runtime
        x = np.array([l[1] for l in edge_points])
        y = np.array([l[0] for l in edge_points])
        outlines.append([x, y])
        upper_lines = np.where(lines['upper'] > 0)

    _correct_all_outlines(outlines, orig_mask_shape, extent=extent)
    return outlines


def _correct_all_outlines(outlines, orig_mask_shape, extent=None):
    '''Performs various corrections on outlines.'''
    if extent is not None:
        orig_ext = [-0.5, orig_mask_shape[1] - 0.5,
                    -0.5, orig_mask_shape[0] - 0.5]
        orig_ranges = [orig_ext[1] - orig_ext[0],
                       orig_ext[3] - orig_ext[2]]
        ext_ranges = [extent[1] - extent[0],
                      extent[3] - extent[2]]
        scales = [ext_ranges[0] / orig_ranges[0],
                  ext_ranges[1] / orig_ranges[1]]

    def find_successive(vec):
        vec = vec.astype('int')
        two_consec = np.where((vec[:-1] + vec[1:]) == 2)[0]
        return two_consec

    for current_outlines in outlines:
        x_lim = (0, orig_mask_shape[1])
        y_lim = (0, orig_mask_shape[0])

        x_above = current_outlines[0] > x_lim[1]
        x_below = current_outlines[0] < x_lim[0]
        y_above = current_outlines[1] > y_lim[1]
        y_below = current_outlines[1] < y_lim[0]

        x_ind, y_ind = list(), list()
        for x in [x_above, x_below]:
            x_ind.append(find_successive(x))
        for y in [y_above, y_below]:
            y_ind.append(find_successive(y))

        all_ind = np.concatenate(x_ind + y_ind)

        if len(all_ind) > 0:
            current_outlines[1] = np.insert(current_outlines[1],
                                            all_ind + 1, np.nan)
            current_outlines[0] = np.insert(current_outlines[0],
                                            all_ind + 1, np.nan)
        # compensate for padding
        current_outlines[0] = current_outlines[0] - 1.
        current_outlines[1] = current_outlines[1] - 1.

        if extent is not None:
            current_outlines[0] = ((current_outlines[0] + 0.5) * scales[0]
                                   + extent[0])
            current_outlines[1] = ((current_outlines[1] + 0.5) * scales[1]
                                   + extent[2])
