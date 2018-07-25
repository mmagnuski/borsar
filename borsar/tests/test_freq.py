import numpy as np
import mne
from borsar.freq import format_psds
from borsar.channels import select_channels


fname = op.join('.', 'data', 'rest_sample_data-raw.fif')
raw, events = mne.io.read_raw_fif(fname)


def test_format_psds():
    # prepare data
    ch_names = raw.ch_names
    info = mne.create_info(ch_names, 250., ch_types='eeg',
                           montage='easycap-M1')
    freqs = np.array([4, 6, 8, 10])
    psd_data = np.random.random((5, 64, 4))

    # test 1
    sel = select_channels(info, select='frontal')
    psds, _ = format_psds(psd_data, freqs, info, average_freq=(8, 10),
                          selection='frontal')
    assert (psds == np.log(psd_data[..., 2:].mean(axis=-1)[:, sel])).all()

    # test 2
    sel = select_channels(info, select='asy_pairs')
    psds, _ = format_psds(psd_data, freqs, info, average_freq=None,
                          selection='asy_pairs')
    should_be = (np.log(psd_data[:, sel['right']]) -
                 np.log(psd_data[:, sel['left']]))
    assert (psds == should_be).all()
