import numpy as np
import scipy


def compute_regression_t(data, preds):
    '''Compute regression t values for whole multidimensional data space.

    Parameters
    ----------
    data : numpy array of shape (observations, ...)
        Data to perform regression on. Frist dimension represents observations
        (for example trials or subjects).
    preds : numpy array of shape (observations, predictors)
        Predictor array to use in regression.

    Returns
    -------
    t_vals : numpy array of shape (predictors, ...)
        T values for all predictors for the original data space.
    '''
    n_obs = data.shape[0]
    if preds.ndim == 1:
        preds = np.atleast_2d(preds).T
    # add constant term
    if not (preds[:, 0] == 1).all():
        preds = np.concatenate([np.ones((n_obs, 1)), preds], axis=1)

    n_preds = preds.shape[1]
    assert n_obs == preds.shape[0], ('preds must have the same number of rows'
        ' as the size of first data dimension (observations).')

    original_shape = data.shape
    data = data.reshape((original_shape[0], np.prod(original_shape[1:])))
    coefs, _, _, _ = scipy.linalg.lstsq(preds, data)
    prediction = (preds[:, :, np.newaxis] * coefs[np.newaxis, :]
                  ).sum(axis=1)
    MSE = (((data - prediction) ** 2).sum(axis=0, keepdims=True)
           / (n_obs - n_preds))
    SE = np.sqrt(MSE * np.diag(np.linalg.inv(preds.T @ preds))[:, np.newaxis])
    t_vals = (coefs / SE).reshape([n_preds, *original_shape[1:]])
    return t_vals
