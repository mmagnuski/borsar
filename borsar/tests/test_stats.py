import time
import numpy as np
import statsmodels.api as sm
from borsar.stats import compute_regression_t


def test_compute_regression_t():
    # compare regression_t to statsmodels
    data = np.random.random((25, 120, 64))
    preds = np.random.random((25, 3))

    n_preds = preds.shape[1] + 1
    n_channels, n_times = data.shape[1:]

    t0 = time.clock()
    preds_sm = sm.add_constant(preds)
    t_vals_sm = np.zeros((n_preds, n_channels, n_times))
    for ch in range(n_channels):
        for tm in range(n_times):
            mdl = sm.OLS(data[:, ch, tm], preds_sm).fit()
            for pred in range(n_preds):
                t_vals_sm[pred, ch, tm] = mdl.tvalues[pred]
    elapsed_sm = time.clock() - t0

    t0 = time.clock()
    t_vals_borsar = compute_regression_t(data, preds)
    elapsed_borsar = time.clock() - t0

    assert t_vals_sm.shape == t_vals_borsar.shape
    np.testing.assert_allclose(t_vals_borsar, t_vals_sm, rtol=1e-9)

    # make sure we are at least 90 times faster than statsmodels loop
    assert elapsed_borsar * 90 < elapsed_sm

    # make sure preds are turned from 1d to 2d
    data = np.random.random((35, 2))
    preds = np.random.random(35)
    t_vals_borsar = compute_regression_t(data, preds)
    assert t_vals_borsar.shape == (2, 2)

    # make sure p_values are correct
    data = np.random.random((15, 10))
    preds = np.random.random((15, 3))
    n_points = data.shape[1]
    n_preds = preds.shape[1] + 1

    preds_sm = sm.add_constant(preds)
    p_vals_sm = np.zeros((n_preds, n_points))
    for pnt in range(n_points):
        mdl = sm.OLS(data[:, pnt], preds_sm).fit()
        for pred in range(n_preds):
            p_vals_sm[pred, pnt] = mdl.pvalues[pred]

    _, p_vals_borsar = compute_regression_t(data, preds, return_p=True)
    np.testing.assert_allclose(p_vals_borsar, p_vals_sm, rtol=1e-9)
