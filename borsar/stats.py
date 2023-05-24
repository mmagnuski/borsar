import numpy as np
import scipy


# - [x] return residuals
# - [ ] consider returning  a dictionary of additional info if required
#       (for example coefficients and SE)
def compute_regression_t(data, preds, return_p=False, return_residuals=False,
                         return_all=False):
    '''Compute regression t values for whole multidimensional data space.

    Parameters
    ----------
    data : numpy array of shape (observations, ...)
        Data to perform regression on. First dimension represents observations
        (for example trials or subjects).
    preds : numpy array of shape (observations, predictors)
        Predictor array to use in regression.
    return_p : bool
        If ``True`` - also return p values. Defaults to ``False``.
    return_residuals : bool
        If ``True`` - also return regression residuals. Defaults to ``False``.
    return_all : bool
        If ``True`` - return all outputs as a dictionary. Defaults to
        ``False``. The outputs include:
        * ``'coefs'`` - regression coefficients
        * ``'SE'`` - standard errors
        * ``'t'`` - t values
        * ``'p'`` - p values (only if ``return_p`` is ``True``)
        * ``'resid'`` - regression residuals
        * ``'df'`` - degrees of freedom
        * ``'predicted'`` - predicted values

    Returns
    -------
    t_vals : numpy array
        T values for all predictors for the original data space. The first
        dimension of the array corresponds to consecutive predictors.
    p_vals : numpy array of shape (predictors, ...)
        P values for all predictors for the original data space. Returned only
        if ``return_p`` is True.
    residuals : numpy array
        Array of model residuals. Has the same shape as input data.
    '''
    n_obs = data.shape[0]
    preds = _handle_preds(preds)

    n_preds = preds.shape[1]
    assert n_obs == preds.shape[0], ('preds must have the same number of rows'
        ' as the size of first data dimension (observations).')

    df = n_obs - n_preds
    original_shape = data.shape
    data = data.reshape((original_shape[0], np.prod(original_shape[1:])))
    coefs, _, _, _ = np.linalg.lstsq(preds, data, rcond=None)
    prediction = (preds[:, :, np.newaxis] * coefs[np.newaxis, :]
                  ).sum(axis=1)
    residuals = data - prediction
    MSE = ((residuals ** 2).sum(axis=0, keepdims=True) / df)
    SE = np.sqrt(MSE * np.diag(np.linalg.pinv(preds.T @ preds))[:, np.newaxis])
    t_vals = (coefs / SE).reshape([n_preds, *original_shape[1:]])

    if return_all:
        out = {'coefs': coefs.reshape([n_preds, *original_shape[1:]]),
               'SE': SE.reshape([n_preds, *original_shape[1:]]),
               't': t_vals,
               'df': df,
               'predicted': prediction.reshape(original_shape)}
    else:
        out = (t_vals,)

    if return_p:
        from scipy.stats import t
        p_vals = t.cdf(-np.abs(t_vals), df) * 2.

        if return_all:
            out['p'] = p_vals
        else:
            out += (p_vals,)

    if return_residuals and not return_all:
        residuals = residuals.reshape(original_shape)
        out += (residuals,)

    # make sure we return a tuple only if we have more than one output
    if len(out) == 1 and not return_all:
        return out[0]
    else:
        return out


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
            return 'p < {:.0e}'.format(powers[which_power])


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


def _find_stat_fun(n_groups, paired, tail):
    '''Find relevant stat_fun given ``n_groups``, ``paired`` and ``tail``.'''
    if n_groups > 2 and tail == 'both':
        raise ValueError('Number of compared groups is > 2, but tail is set'
                         ' to "both". If you want to use ANOVA, set tail to'
                         ' "pos".')
    if n_groups > 2 and not tail == 'both':
        if paired:
            # repeated measures ANOVA
            return rm_anova_stat_fun
        else:
            from scipy.stats import f_oneway

            def stat_fun(*args):
                fval, _ = f_oneway(*args)
                return fval
            return stat_fun
    elif n_groups == 2:
        if paired:
            from scipy.stats import ttest_rel

            def stat_fun(*args):
                tval, _ = ttest_rel(*args)
                return tval
            return stat_fun
        else:
            from mne.stats import ttest_ind_no_p as stat_fun
            return stat_fun
    else:
        # one group
        from mne.stats import ttest_1samp_no_p as stat_fun
        return stat_fun



# FIXME: streamline/simplify permutation reshaping and transposing
# FIXME: time and see whether a different solution is better
# FIXME: splitting across jobs could be smarter (chunks of permutations, not
# one by one)
def _compute_threshold_via_permutations(data, paired, tail, stat_fun,
                                        p_threshold=0.05, n_permutations=1000,
                                        progress=True,
                                        return_distribution=False,
                                        n_jobs=1):
    '''
    Compute significance thresholds using permutations.

    Assumes ``n_conditions x n_observations x ...`` data array.
    Note that the permutations are implemented via shuffling of the condition
    labels, not randomization of independent condition orders.
    '''
    from .utils import progressbar

    if paired:
        # concatenate condition dimension if needed
        if isinstance(data, (list, tuple)):
            data = np.stack(data, axis=0)

        dims = np.arange(data.ndim)
        dims[:2] = [1, 0]
        dim_shape = data.shape
        n_cond, n_obs = dim_shape[:2]
        data_unr = data.transpose(*dims).reshape(
            n_cond * n_obs, *data.shape[2:])

        # compute permutations of the stat
        if n_jobs == 1:
            stats = np.zeros(shape=(n_permutations, *data.shape[2:]))
            pbar = progressbar(progress, total=n_permutations)
            for perm_idx in range(n_permutations):
                stats[perm_idx] = _paired_perm(
                    data_unr, stat_fun, n_cond, n_obs, dim_shape, dims,
                    pbar=pbar
                )
        else:
            from joblib import Parallel, delayed
            stats = Parallel(n_jobs=n_jobs)(
                delayed(_paired_perm)(data_unr, stat_fun, n_cond, n_obs,
                                      dim_shape, dims)
                for perm_idx in range(n_permutations)
            )
            stats = np.stack(stats, axis=0)
    else:
        n_cond = len(data)
        condition = np.concatenate([np.ones(data[idx].shape[0]) * idx
                                    for idx in range(n_cond)])
        data_unr = np.concatenate(data)

        if n_jobs == 1:
            stats = np.zeros(shape=(n_permutations, *data[0].shape[1:]))
            pbar = progressbar(progress, total=n_permutations)
            for perm_idx in range(n_permutations):
                stats[perm_idx] = _unpaired_perm(
                    data_unr, stat_fun, condition, n_cond, pbar=pbar
                )
        else:
            from joblib import Parallel, delayed
            stats = Parallel(n_jobs=n_jobs)(
                delayed(_unpaired_perm)(data_unr, stat_fun, condition, n_cond)
                for perm_idx in range(n_permutations)
            )
            stats = np.stack(stats, axis=0)

    # now check threshold
    if tail == 'pos':
        percentile = 100 - p_threshold * 100
        threshold = np.percentile(stats, percentile, axis=0)
    elif tail == 'neg':
        percentile = p_threshold * 100
        threshold = np.percentile(stats, percentile, axis=0)
    elif tail == 'both':
        percentile_neg = p_threshold / 2 * 100
        percentile_pos = 100 - p_threshold / 2 * 100
        threshold = [np.percentile(stats, perc, axis=0)
                     for perc in [percentile_pos, percentile_neg]]
    else:
        raise ValueError(f'Unrecognized tail "{tail}"')

    if not return_distribution:
        return threshold
    else:
        return threshold, stats


def _compute_threshold(data, threshold, p_threshold, paired,
                       one_sample):
    '''Find significance threshold analytically.'''
    if threshold is None:
        from scipy.stats import distributions
        n_groups = len(data)
        n_obs = [len(x) for x in data]

        if n_groups < 3:
            len1 = len(data[0])
            len2 = (len(data[1]) if (len(data) > 1 and data[1] is not None)
                    else 0)
            df = (len1 - 1 if paired or one_sample else len1 + len2 - 2)
            threshold = distributions.t.ppf(1 - p_threshold / 2., df=df)
        else:
            # ANOVA F
            n_obs = data[0].shape[0] if paired else sum(n_obs)
            dfn = n_groups - 1
            dfd = n_obs - n_groups
            threshold = distributions.f.ppf(1. - p_threshold, dfn, dfd)
    return threshold


def _paired_perm(data_unr, stat_fun, n_cond, n_obs, dim_shape, dims,
                 pbar=None):
    rnd = (np.random.random(size=(n_cond, n_obs))).argsort(axis=0)
    idx = (rnd + np.arange(n_obs)[None, :] * n_cond).T.ravel()
    this_data = data_unr[idx].reshape(
        n_obs, n_cond, *dim_shape[2:]).transpose(*dims)
    stat = stat_fun(*this_data)

    if pbar is not None:
        pbar.update(1)

    return stat


def _unpaired_perm(data_unr, stat_fun, condition, n_cond, pbar=None):
    rnd = condition.copy()
    np.random.shuffle(rnd)
    this_data = [data_unr[rnd == idx] for idx in range(n_cond)]
    stat = stat_fun(*this_data)

    if pbar is not None:
        pbar.update(1)

    return stat


def rm_anova_stat_fun(*args):
    '''Stat fun that does one-way repeated measures ANOVA.'''
    from mne.stats import f_mway_rm

    data = np.stack(args, axis=1)
    n_factors = data.shape[1]

    fval, _ = f_mway_rm(data, factor_levels=[n_factors],
                        return_pvals=False)

    if data.ndim > 3:
        fval = fval.reshape(data.shape[2:])
    return fval
