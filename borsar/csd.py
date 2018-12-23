import numpy as np


def current_source_density(inst, G, H, smoothing=1.0e-5, head_radius=1.):
    '''
    Compute current source density for given mne object instance.

    Note that this function works in-place.

    Parameters
    ----------
    inst : mne object instance
        Raw or Epochs data.
    G : numpy matrix
        G matrix.
    H : numpy matrix
        H matrix.
    smoothing : float
        CSD smoothing. Defaults to 1.0e-5.
    head_radius : float
        Radius of the head sphere.

    Returns
    -------
    inst : mne object instance
        The data with CSD reference.
    '''
    import mne
    from mne.utils import _get_inst_data

    data = _get_inst_data(inst)
    got_epochs = isinstance(inst, mne.Epochs)
    if got_epochs:
        n_epochs, n_channels, n_times = data.shape
    else:
        n_channels, n_times = data.shape

    # ensure the data is average referenced
    inst.set_eeg_reference('average', projection=False)

    # apply current source density
    data = _current_source_density(data, G, H, smoothing, head_radius)

    if not isinstance(inst, mne.Evoked):
        inst._data = data
    else:
        inst.data = data

    return inst


def _current_source_density(data, G, H, smoothing=1.0e-5, head_radius=1.):
    '''Python implementation of CSD.m from Matlab CSD toolbox.'''
    # FIXME add checks for G and H sizes

    n_channels, n_times = data.shape

    # average reference
    data -= data.mean(axis=0, keepdims=True)

    # check if bads exist and interpolate if necessary (or throw an error)

    # add lambda smoothing to the diagonal
    G = G + np.diag(np.ones(G.shape[0]) * smoothing)
    G_inv = np.linalg.inv(G)
    G_inv_row_sum = G_inv.sum(axis=1)
    G_inv_sum = G_inv_row_sum.sum()

    for idx in range(n_times):
        Cp = np.dot(G_inv, data[:, idx])
        c0 = Cp.sum() / G_inv_sum
        C = Cp - np.dot(c0, G_inv_row_sum.T)

        # for ch_idx in range(n_channels):
        #     data[ch_idx, idx] = (C * H[ch_idx].T).sum() / head_radius
        data[:, idx] = np.dot(C, H) / head_radius

    return data
