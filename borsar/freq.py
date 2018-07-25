import numpy as np
from .utils import find_range
from .channels import select_channels, get_ch_names


def format_psds(psds, freq, info, average_freq=(8, 13),
                selection='asy_frontal', log_transform=True):
    '''
    psds should be subjects x channels x frequencies

    Returns
    -------
    psds
    ch_names
    '''
    if average_freq:
        rng = find_range(freq, average_freq)
        psds = psds[..., rng].mean(axis=-1)

    ch_names = get_ch_names(info)
    sel = select_channels(info, selection)

    if log_transform:
        psds = np.log(psds)

    if 'asy' in selection:
        # compute asymmetry
        lft = psds[:, sel['left']]
        rgt = psds[:, sel['right']]
        psds = rgt - lft
        ch_names = ['{}-{}'.format(ch1, ch2) for ch1, ch2 in
                    zip(np.array(ch_names)[sel['left']],
                        np.array(ch_names)[sel['right']])]
    else:
        psds = psds[:, sel]
        ch_names = list(np.array(ch_names)[sel])

    return psds, ch_names
