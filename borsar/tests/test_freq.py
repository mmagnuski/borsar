import os.path as op
import numpy as np
import mne

from borsar.utils import create_fake_raw
from borsar.freq import format_psds
from borsar.channels import select_channels


fname = op.join('.', 'borsar', 'data', 'rest_sample_data-raw.fif')
raw = mne.io.read_raw_fif(fname)


def test_compute_rest_psd():
    # test on fake raw with 3 channels
    raw = create_fake_raw(n_channels=3, n_samples=26, sfreq=4.)
    events = np.array([[5, 0, 1], [10, 0, 2], [16, 0, 1], [21, 0, 2]])
    zigzag = np.array([-1, 1, -1, 1])
    raw._data[1, 5:9] = zigzag
    raw._data[1, 16:20] = zigzag
    raw._data[2, 10:14] = zigzag * 2
    raw._data[2, 21:25] = zigzag * 2

    psd, freq = compute_rest_psd(raw, events=events, tmin=0., tmax=1.5,
                                 winlen=1., step=1.)
    assert psd[1][1, -1] > psd[1][0, -1]
    assert psd[2][2, -1] > psd[2][0, -1]
    assert psd[2][2, -1] > psd[1][1, -1]

    psd2, freq2 = compute_rest_psd(raw, events=events, event_id=1, tmin=0.,
                                   tmax=1.5, winlen=1., step=1.)
    assert (psd2 == psd[1]).all()
    assert (freq2 == freq).all()


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
