import numpy as np
import scipy


# - [ ] add option to return residuals
def compute_regression_t(data, preds, return_p=False):
    '''Compute regression t values for whole multidimensional data space.

    Parameters
    ----------
    data : numpy array of shape (observations, ...)
        Data to perform regression on. Frist dimension represents observations
        (for example trials or subjects).
    preds : numpy array of shape (observations, predictors)
        Predictor array to use in regression.
    return_p : bool
        If True - also return p values. Defaults to False.

    Returns
    -------
    t_vals : numpy array of shape (predictors, ...)
        T values for all predictors for the original data space.
    p_vals : numpy array of shape (predictors, ...)
        P values for all predictors for the original data space.
    '''
    n_obs = data.shape[0]
    preds = _handle_preds(preds)

    n_preds = preds.shape[1]
    assert n_obs == preds.shape[0], ('preds must have the same number of rows'
        ' as the size of first data dimension (observations).')

    df = n_obs - n_preds
    original_shape = data.shape
    data = data.reshape((original_shape[0], np.prod(original_shape[1:])))
    coefs, _, _, _ = scipy.linalg.lstsq(preds, data)
    prediction = (preds[:, :, np.newaxis] * coefs[np.newaxis, :]
                  ).sum(axis=1)
    MSE = (((data - prediction) ** 2).sum(axis=0, keepdims=True) / df)
    SE = np.sqrt(MSE * np.diag(np.linalg.pinv(preds.T @ preds))[:, np.newaxis])
    t_vals = (coefs / SE).reshape([n_preds, *original_shape[1:]])

    if return_p:
        from scipy.stats import t
        p_vals = t.cdf(-np.abs(t_vals), df) * 2.
        return t_vals, p_vals
    else:
        return t_vals


def format_pvalue(pvalue):
    '''Format p value according to APA rules.'''
    if pvalue >= .001:
        return 'p = {:.3f}'.format(pvalue)
    else:
        powers = 10 ** np.arange(-3, -26, step=-1, dtype='float')
        which_power = np.where(pvalue < powers)[0][-1]
        if which_power < 2:
            return 'p < {}'.format(['0.001', '0.0001'][which_power])
        else:
            return 'p < {}'.format(str(powers[which_power]))


def _handle_preds(preds):
    '''Reshape predictors, add constant term if not present.'''
    assert preds.ndim < 3, '`preds` must be 1d or 2d array.'
    if preds.ndim == 1:
        preds = np.atleast_2d(preds).T

    # add constant term
    if not (preds[:, 0] == 1).all():
        n_obs = preds.shape[0]
        preds = np.concatenate([np.ones((n_obs, 1)), preds], axis=1)

    return preds
