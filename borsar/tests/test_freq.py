import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne

from borsar.utils import (create_fake_raw, download_test_data,
                          _get_test_data_dir)
from borsar.freq import compute_rest_psd, compute_psd
from borsar.channels import select_channels


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


def test_psd_class():
    data_dir = _get_test_data_dir()
    download_test_data()

    raw_fname = op.join(data_dir, 'DiamSar_023_rest_raw.fif')
    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    event_id = {'S 11': 11}
    events, _ = mne.events_from_annotations(raw, event_id=event_id)
    psd = compute_psd(raw, tmin=0.5, tmax=20.5, winlen=2.,
                      step=0.5, events=events, event_id=[11])

    # make sure plotting does not error
    psd.plot(dB=False, fmax=40, show=False)
    psd.plot(fmax=40, average=True, show=False);

    topo = psd.plot_topomap(fmin=10, fmax=12)
    assert isinstance(topo.axes, plt.Axes)

    topo = psd.plot_topomap(fmin=[5, 11], fmax=[7, 13])
    assert len(topo.axes) == 2

    assert len(psd.ch_names) == len(raw.ch_names)
    assert psd.data.shape[-1] == len(psd.freq)
    assert psd.data.shape[-2] == len(raw.ch_names)

    avg_psd_arr = psd.average(fmin=10, fmax=12)
    assert avg_psd_arr.ndim == 1
    assert avg_psd_arr.shape[0] == len(raw.ch_names)

    psd.crop(fmin=10, fmax=15)
    assert (psd.freq[0] - 10) < 0.5
    assert (psd.freq[-1] - 15) < 0.5
