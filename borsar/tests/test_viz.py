import os.path as op
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import signal
from scipy.stats import distributions
from matplotlib.patches import Rectangle

import mne
import pytest

from borsar.channels import select_channels
from borsar.freq import compute_rest_psd
from borsar.utils import find_range, _get_test_data_dir
from borsar.viz import Topo, _extract_topo_channels, heatmap, highlight
from borsar._heatmap import _create_cluster_contour, _add_image_mask
from borsar._vizutils import color_limits


data_dir = _get_test_data_dir()
fname = op.join(data_dir, 'rest_sample_data-raw.fif')
raw = mne.io.read_raw_fif(fname)
psd, freq = compute_rest_psd(raw, tmin=1., tmax=3., winlen=1.)


def test_topo():
    alpha = find_range(freq, (8, 12))
    alpha_topo = psd[:, alpha].mean(axis=-1)

    # currently a smoke test
    topo = Topo(alpha_topo, raw.info, show=False)
    topo.set_linewidth(contours=0.5, outlines=2.5)
    topo.solid_lines()
    topo.remove_levels(0.)
    topo.mark_channels([1, 2, 3, 6, 8], markersize=10.)
    ch, pos = _extract_topo_channels(topo.axes)

    # test 1 x channels vector
    topo = Topo(alpha_topo[:, np.newaxis], raw.info, show=False)
    fig, ax = plt.subplots()
    topo = Topo(alpha_topo[:, np.newaxis], raw.info, axes=[ax], show=False)

    fig, ax = plt.subplots()
    with pytest.raises(RuntimeError):
        ch, pos = _extract_topo_channels(ax)

    # earlier mne versions used scatter for points
    ax.scatter(pos[:, 0], pos[:, 1])
    ch, pos2 = _extract_topo_channels(ax)
    assert (pos == pos2).all()

    topo = Topo(alpha_topo, raw.info, axes=ax, show=False)
    assert topo.axes == ax
    assert topo.fig == fig

    # various selections
    for selection in ['frontal', 'asy_frontal', 'asy_all']:
        select = select_channels(raw, select=selection)
        select = ([select] if isinstance(select, np.ndarray) else
                  [select['left'], select['right']])
        for sel in select:
            sel_info = mne.pick_info(raw.info, sel=sel)
            topo = Topo(alpha_topo[sel], sel_info, show=False)

    # test topo update
    topo = Topo(alpha_topo, raw.info, show=False)
    topo.update(alpha_topo[::-1])

    # make sure border='mean' works
    topo = Topo(alpha_topo, raw.info, extrapolate='head',
                border='mean', show=False)
    assert topo.interpolator.border == 'mean'
    # add a test for one point checking if its value is the mean of neighbours


def test_multi_topo():
    n_channels = len(raw.ch_names)
    freq_ranges = find_range(freq, [(3, 7), (8, 12), (15, 25)])
    freq_topos = np.concatenate(
        [psd[:, slc].mean(axis=-1, keepdims=True)
         for slc in freq_ranges], axis=1)
    tp = Topo(freq_topos, raw.info)

    # test changing line width
    tp.set_linewidth(contours=0.35)

    linewidths = list()
    for lines in tp.lines:
        for line in lines.collections:
            linewidths.append(line.get_linewidths()[0])
    assert (np.array(linewidths) == 0.35).all()

    # other tests
    tp.solid_lines()
    tp.remove_levels(0.)
    tp.mark_channels([1, 2, 3, 6, 8], markersize=10.)

    # test different marks per topo:
    fig, axes = plt.subplots(ncols=3)
    tp = Topo(freq_topos, raw.info, axes=axes)
    mark_idxs = [[0, 1], [3, 5], [9, 10, 13]]

    for this_topo, mrk in zip(tp, mark_idxs):
        this_topo.mark_channels(mrk, markerfacecolor='g')

    for ax, mrk in zip(tp.axes, mark_idxs):
        last_line = ax.findobj(plt.Line2D)[-1]
        mark_pos = np.stack(last_line.get_data(), axis=1)
        assert (mark_pos == tp.chan_pos[mrk, :]).all()

    # boolean channel marking
    mark_idxs = [[2, 5, 12], [3], [8, 23, 27]]
    ifmark = np.zeros((n_channels, len(mark_idxs)), dtype='bool')
    for idx, mrk in enumerate(mark_idxs):
        ifmark[mrk, idx] = True

    for this_topo, mrk in zip(tp, ifmark.T):
        this_topo.mark_channels(mrk, markerfacecolor='r')

    for ax, mrk in zip(tp.axes, mark_idxs):
        last_line = ax.findobj(plt.Line2D)[-1]
        mark_pos = np.stack(last_line.get_data(), axis=1)
        assert (mark_pos == tp.chan_pos[mrk, :]).all()

    # one 1d array
    tp.mark_channels(np.array([8, 12, 21, 31]), markersize=5)

    # make sure that iterating works and updates base Topo
    topo = Topo(freq_topos, raw.info)

    for tp, mrk in zip(topo, mark_idxs):
        tp.mark_channels(mrk)

    # make sure topo.marks is updated
    for tp, mrk in zip(topo, mark_idxs):
        mark_pos = np.stack(tp.marks[0].get_data(), axis=1)
        assert (mark_pos == tp.chan_pos[mrk, :]).all()


def test_outlines():
    # test 01
    img = np.random.rand(10, 10) > 0.65
    plt.imshow(img)
    cntr = _create_cluster_contour(img)
    for cnt in cntr:
        x, y = cnt
        plt.plot(x, y, color='w', lw=2)

    # test 02
    data = np.zeros((5, 5), dtype='bool')
    data[[1, 2, 2, 3], [0, 0, 1, 1]] = True
    data[[1, 1, 2, 3, 2], [2, 3, 3, 3, 4]] = True
    cntr = _create_cluster_contour(data)

    correct_cntr = [[
        np.array([-0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, -0.5, -0.5]),
        np.array([0.5, 0.5, 1.5, 1.5, 3.5, 3.5, 2.5, 2.5, 0.5])],
        [np.array([1.5, 3.5, 3.5, 4.5, np.nan, 4.5, 3.5, 3.5, 2.5, 2.5, 1.5,
                   1.5]), np.array([0.5, 0.5, 1.5, 1.5, np.nan, 2.5, 2.5, 3.5,
                                    3.5, 1.5, 1.5, 0.5])]]

    for c1, c2 in zip(cntr, correct_cntr):
        np.testing.assert_equal(c1, c2)

    # TODO - add test for outlines with extent
    cntr = _create_cluster_contour(data, extent=(0, 10, 5.25, 7.75))


def test_topo_simulated_data():
    x_pos = np.arange(-3, 3.1, step=0.5)
    y_pos = np.arange(-3, 3.1, step=0.5)
    n_elem = x_pos.shape[0]

    x_pos = np.tile(x_pos, (n_elem, 1))
    y_pos = np.tile(y_pos[:, None], (1, n_elem))

    xy_pos = np.stack([x_pos.ravel(), y_pos.ravel()], axis=1)

    # gaussian effect at 0, 0
    distances = np.linalg.norm(xy_pos, axis=1)
    norm = distributions.norm()
    y_val = norm.pdf(distances)

    # plot topo and test zoom
    within_limits = distances <= 3.
    tp = Topo(y_val[within_limits], xy_pos[within_limits, :], sphere=2.5)
    tp.zoom(xlim=(0, 3.5), ylim=(0, 3.5))

    assert tp.axes.get_xlim() == (0., 3.5)
    assert tp.axes.get_ylim() == (0., 3.5)

    # optional matplotlib plotting
    # y_val_mat = y_val.copy()
    # y_val_mat[~within_limits] = np.nan
    # y_val_mat = y_val_mat.reshape((n_elem, n_elem))
    # plt.imshow(y_val_mat, extent=[-3.25, 3.25, -3.25, 3.25])
    # plt.scatter(xy_pos[:, 0], xy_pos[:, 1], color='k')


def test_topo_grid():
    "Test that nrows and ncols are respected by Topo."
    pos = (np.random.rand(10, 2) - 0.5) * 0.33
    data = np.random.rand(10, 8)

    def test_grid_shape(topo, nrows, ncols):
        gridspec = topo.axes[0].get_gridspec()
        assert gridspec.nrows == nrows
        assert gridspec.ncols == ncols

    topo = Topo(data, pos)
    test_grid_shape(topo, 2, 6)
    plt.close(topo.axes[0].figure)

    topo = Topo(data, pos, nrows=3)
    test_grid_shape(topo, 3, 3)
    plt.close(topo.axes[0].figure)

    topo = Topo(data, pos, ncols=2)
    test_grid_shape(topo, 4, 2)
    plt.close(topo.axes[0].figure)

    topo = Topo(data, pos, nrows=5, ncols=5)
    test_grid_shape(topo, 5, 5)
    plt.close(topo.axes[0].figure)

    data = np.random.rand(10, 40)
    topo = Topo(data, pos)
    test_grid_shape(topo, 6, 7)
    plt.close(topo.axes[0].figure)


def test_heatmap():
    data = np.random.random((5, 6))
    x = np.linspace(10, 12, num=6)
    y = np.linspace(3, 9, num=5)

    # currently just smoke tests
    heatmap(data)
    heatmap(data, x_axis=x)
    out1 = heatmap(data, x_axis=x, y_axis=y)
    out2 = heatmap(data, x_axis=x, y_axis=y, colorbar=False)
    assert isinstance(out2, plt.Axes)
    assert isinstance(out1[0], plt.Axes)
    assert isinstance(out1[1], mpl.colorbar.Colorbar)

    plt.close('all')
    msk = data > 0.5
    heatmap(data, x_axis=x, y_axis=y, mask=msk)
    heatmap(data, x_axis=x, y_axis=y, mask=msk, outlines=True)

    plt.imshow(data)
    _add_image_mask(msk)
    plt.close('all')

    msk = np.stack([data < 0.25, data > 0.75], axis=0)
    heatmap(data, x_axis=x, y_axis=y, mask=msk, outlines=True)

    img = heatmap(data, mask=msk, outlines=True, colorbar=False,
                  line_kwargs={'color': ['w', 'r']})

    # assert that line colors have both white and red colors
    chldr = img.axes.get_children()
    lines = [chld for chld in chldr if 'Line2D' in str(chld)]
    line_colors = [l.get_color() for l in lines]
    if msk[0].any():
        assert 'w' in line_colors
    if msk[1].any():
        assert 'r' in line_colors

    plt.close('all')


def test_utils():
    clim = color_limits(np.random.randint(0, high=2, dtype='bool'))
    assert clim == (0., 1.)


def compare_box_and_slice(x, box, slc):
    '''Function used to compare slice limits and box range of a rectangle.'''
    halfsample = np.diff(x).mean() / 2
    correct_limits = x[slc][[0, -1]] + [-halfsample, halfsample]

    bbox_limits = box.get_bbox().get_points()
    bbox_x_limits = bbox_limits[:, 0]
    print(bbox_x_limits)
    print(correct_limits)

    return np.allclose(correct_limits, bbox_x_limits)


def test_highlight():
    x = np.arange(0, 10, step=0.05)
    n_times = len(x)

    y = np.random.random(n_times)

    # simple usage
    # ------------
    line = plt.plot(x, y)
    highlight(x, slice(10, 40))

    ax = line[0].axes
    rectangles = ax.findobj(Rectangle)
    assert len(rectangles) == 2
    plt.close(ax.figure)

    # two slices, setting color and alpha
    # -----------------------------------
    line = plt.plot(x, y)
    use_alpha, use_color = 0.5, [0.75] * 3
    slices = [slice(10, 40), slice(60, 105)]
    highlight(x, slices, alpha=use_alpha, color=use_color)

    ax = line[0].axes
    rectangles = ax.findobj(Rectangle)
    assert len(rectangles) == 3

    # check box color and box alpha
    rgba = rectangles[0].get_facecolor()
    assert (rgba[:3] == np.array(use_color)).all()
    assert rgba[-1] == use_alpha

    # compare slices and rectangles:
    for box, slc in zip(rectangles, slices):
        assert compare_box_and_slice(x, box, slc)
    plt.close(ax.figure)

    # two slices, using bottom_bar
    # ----------------------------
    line = plt.plot(x, y)

    slices = [slice(10, 40), slice(60, 105)]
    highlight(x, slices, bottom_bar=True)

    ax = line[0].axes
    rectangles = ax.findobj(Rectangle)
    assert len(rectangles) == 5

    for idx, col in enumerate([0.95, 0, 0.95, 0]):
        rect_color = rectangles[idx].get_facecolor()[:3]
        assert (rect_color == np.array([col] * 3)).all()
    plt.close(ax.figure)


def test_highlight2():
    # create random smoothed data
    x = np.linspace(-0.5, 2.5, num=1000)
    y = np.random.rand(1000)

    gauss = signal.windows.gaussian(75, 12)
    smooth_y = signal.correlate(y, gauss, mode='same')

    lines = plt.plot(x, smooth_y)
    ax = lines[0].axes

    maxval = smooth_y.max()
    mark = smooth_y > maxval * 0.9

    # highlight with bottom_bar
    highlight(highlight=mark, ax= ax, bottom_bar=True)

    # make sure there is equal number of higher and shorter bars
    rectangles = ax.findobj(plt.Rectangle)[::-1]
    heights = np.array([r.get_height() for r in rectangles])
    shorter, higher = min(heights), max(heights)

    which_shorter = heights == shorter
    n_shorter = (which_shorter).sum()
    which_longer = heights == higher
    n_longer = (which_longer).sum()
    assert n_shorter == n_longer

    # make sure the colors agree and match the longer / higher bars correctly
    colors = np.stack([r.get_facecolor() for r in rectangles],
                      axis=0)[:, :3]

    light_gray = np.array([0.95, 0.95, 0.95])[None, :]
    black = np.array([0., 0., 0.])[None, :]
    assert ((colors == light_gray).all(axis=1) == which_longer).all()
    assert ((colors == black).all(axis=1) == which_shorter).all()
