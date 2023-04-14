import time
import numpy as np
import statsmodels.api as sm
from borsar.stats import compute_regression_t, format_pvalue


def test_compute_regression_t():
    # create test data
    data = np.random.random((25, 120, 64))
    preds = np.random.random((25, 3))
    n_preds = preds.shape[1] + 1
    n_channels, n_times = data.shape[1:]

    # calculate regression t-values with statsmodels loop
    t0 = time.perf_counter()
    preds_sm = sm.add_constant(preds)
    t_vals_sm = np.zeros((n_preds, n_channels, n_times))
    for ch in range(n_channels):
        for tm in range(n_times):
            mdl = sm.OLS(data[:, ch, tm], preds_sm).fit()
            for pred in range(n_preds):
                t_vals_sm[pred, ch, tm] = mdl.tvalues[pred]
    elapsed_sm = time.perf_counter() - t0

    # calculate regression t-values with borsar
    t0 = time.perf_counter()
    t_vals_borsar = compute_regression_t(data, preds)
    elapsed_borsar = time.perf_counter() - t0

    # make sure t-values are almost the same (same up to ~8 decimal places)
    # (most of the time it is the same up to 9 decimal places but not always)
    assert t_vals_sm.shape == t_vals_borsar.shape
    np.testing.assert_allclose(t_vals_borsar, t_vals_sm, rtol=1e-8)

    # make sure we are at least 2 times faster than statsmodels loop
    # (although in most cases this will be about 50-100 times faster)
    # (the factor of 2 is used because the speedup is much less evident
    #  on Circle CI than on a local machine - likely due to the computational
    #  resources on free Circle plan)
    assert elapsed_borsar * 2 < elapsed_sm

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


def test_compute_regression_t_residuals():

    data = np.random.rand(56, 20, 20)
    preds = np.random.rand(56)

    _, resid = compute_regression_t(data, preds, return_residuals=True)

    preds_sm = sm.add_constant(preds)
    for ix in range(20):
        for iy in range(20):
            model = sm.OLS(data[:, ix, iy], preds_sm)
            result = model.fit()

            np.testing.assert_allclose(result.resid, resid[:, ix, iy])


def test_format_p_value():
    assert format_pvalue(0.13) == 'p = 0.130'
    assert format_pvalue(0.035) == 'p = 0.035'
    assert format_pvalue(0.001) == 'p = 0.001'
    assert format_pvalue(0.0025) == 'p = 0.003'
    assert format_pvalue(0.00025) == 'p < 0.001'
    assert format_pvalue(0.000015) == 'p < 0.0001'
    assert format_pvalue(0.000000000015) == 'p < 1e-10'
