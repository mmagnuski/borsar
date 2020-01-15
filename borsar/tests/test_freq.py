import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pytest
import mne

from borsar.utils import (create_fake_raw, download_test_data,
                          _get_test_data_dir)
from borsar.freq import compute_rest_psd, compute_psd, PSD
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

    psd, freqs = compute_rest_psd(raw, events=events, tmin=0., tmax=1.5,
                                 winlen=1., step=1.)
    assert psd[1][1, -1] > psd[1][0, -1]
    assert psd[2][2, -1] > psd[2][0, -1]
    assert psd[2][2, -1] > psd[1][1, -1]

    psd2, freqs2 = compute_rest_psd(raw, events=events, event_id=1, tmin=0.,
                                   tmax=1.5, winlen=1., step=1.)
    assert (psd2 == psd[1]).all()
    assert (freqs2 == freqs).all()


def test_psd_class():
    # get data
    data_dir = _get_test_data_dir()
    download_test_data()

    # read data
    raw_fname = op.join(data_dir, 'DiamSar_023_rest_raw.fif')
    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    # compute psd
    event_id = {'S 11': 11}
    events, _ = mne.events_from_annotations(raw, event_id=event_id)
    psd = compute_psd(raw, tmin=0.5, tmax=20.5, winlen=2.,
                      step=0.5, events=events, event_id=[11])

    # make sure plotting does not error
    psd.plot(dB=False, fmax=40, show=False)
    psd.plot(fmax=40, average=True, show=False)

    topo = psd.plot_topomap(fmin=10, fmax=12, show=False)
    assert isinstance(topo.axes, plt.Axes)

    topo = psd.plot_topomap(fmin=[5, 11], fmax=[7, 13], show=False)
    assert len(topo.axes) == 2

    # data shape checks
    assert len(psd.ch_names) == len(raw.ch_names)
    assert psd.data.shape[-1] == len(psd.freqs)
    assert psd.data.shape[-2] == len(raw.ch_names)

    avg_psd_arr = psd.average(fmin=10, fmax=12)
    assert avg_psd_arr.ndim == 1
    assert avg_psd_arr.shape[0] == len(raw.ch_names)

    psd2 = psd.average()
    assert (psd.data == psd2.data).all()

    psd.crop(fmin=10, fmax=15)
    assert (psd.freqs[0] - 10) < 0.5
    assert (psd.freqs[-1] - 15) < 0.5

    # test for (deep)copy
    psd2 = psd.copy()
    psd2.data[0, 0] = 23
    assert not (psd._data[0, 0] == 23)

    # psd with Epochs
    # ---------------
    epochs = mne.Epochs(raw, events, event_id, tmin=0., tmax=23.,
                        baseline=None, preload=True)
    psd_epo = compute_psd(epochs, tmin=0.5, tmax=20.5, winlen=2.,
                          step=0.5, events=events, event_id=[11])

    psd_epo.plot(show=False)
    psd_avg = psd_epo.copy().average()
    assert psd_epo.data.ndim == 3
    assert psd_avg.data.ndim == 2

    avg_psd_arr = psd_epo.copy().average(fmin=10, fmax=12)
    assert avg_psd_arr.ndim == 1
    assert avg_psd_arr.shape[0] == len(raw.ch_names)

    avg_psd_arr = psd_epo.copy().average(fmin=10, fmax=12, epochs=False)
    assert avg_psd_arr.ndim == 2
    assert avg_psd_arr.shape[0] == len(epochs)
    assert avg_psd_arr.shape[1] == len(raw.ch_names)

    with pytest.raises(TypeError, match='works only with Raw or Epochs'):
        psd_epo = compute_psd(avg_psd_arr, tmin=0.5, tmax=20.5, winlen=2.,
                              step=0.5, events=events, event_id=[11])

    # psd construction
    with pytest.raises(ValueError, match='has to be 3d'):
        use_data = psd_epo.data[..., np.newaxis]
        psd = PSD(use_data, psd_epo.freqs, raw.info)

    with pytest.raises(ValueError, match='or 2d'):
        use_data = psd_epo.data[0, :, 0]
        psd = PSD(use_data, psd_epo.freqs, raw.info)

    # test for __repr__
    psd_epo2 = psd_epo.copy().crop(fmin=8, fmax=12)
    rpr = '<borsar.freq.PSD (2 epochs, 64 channels, 9 frequencies), 8 - 12 Hz>'
    assert str(psd_epo2) == rpr

    # test chaining
    psd_epo3 = psd_epo.copy().crop(fmin=8, fmax=12).average()
    rpr = '<borsar.freq.PSD (64 channels, 9 frequencies), 8 - 12 Hz>'
    assert str(psd_epo3) == rpr

    # test channel picking
    psd2 = psd_epo.copy().pick_channels(['Fz', 'Pz'])
    assert psd2.data.shape[1] == 2

    # missing:
    # compute_rest_psd when events is None

    # Epochs with metadata
    # --------------------
    # read data
    epochs_fname = op.join(data_dir, 'GabCon-48_epo.fif')
    epochs = mne.read_epochs(epochs_fname, preload=True)

    assert epochs.metadata is not None

    psd = compute_psd(epochs, tmin=0.5, tmax=1.)
    assert psd.data.ndim == 3
    assert psd.data.shape[0] == epochs._data.shape[0]

    psd_slow = psd['RT > 0.65']
    epochs_slow = epochs['RT > 0.65']
    # TODO: later use len(psd_slow) here
    assert len(epochs_slow) == psd_slow.data.shape[0]
