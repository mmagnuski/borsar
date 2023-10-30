import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pytest
import mne
from packaging import version

from borsar.utils import (create_fake_raw, download_test_data,
                          _get_test_data_dir)
from borsar.freq import compute_rest_psd, compute_psd, PSD


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


<<<<<<< HEAD
=======
# @pytest.mark.skip(reason="borsar.freq.PSD will be deprecated soon")
>>>>>>> f8e746a (check if we still have PSD issues)
def test_psd_class():
    mne_version = version.parse(mne.__version__)
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
    if mne_version > version.parse('0.18'):
        # FIXME mne's _plot_psd changes very frequently, so currently we drop
        #  this test. Later on - we'll switch to mne spectral objects.
        pass
        # psd.plot(dB=False, fmax=40, show=False)
        # psd.plot(fmax=40, average=True, show=False)
    else:
        with pytest.raises(ImportError):
            psd.plot(dB=False, fmax=40, show=False)

    topo = psd.plot_topomap(freqs=[11], show=False)
    assert isinstance(topo.axes, plt.Axes)

    topo = psd.plot_topomap(freqs=[6, 11], show=False)
    assert len(topo.axes) == 2

    topo = psd.plot_topomap(freqs=[8, 10, 12], show=False)
    assert len(topo.axes) == 3
    plt.close('all')

    # data shape checks
    assert len(psd.ch_names) == len(raw.ch_names)
    assert psd.data.shape[-1] == len(psd.freqs)
    assert psd.data.shape[-2] == len(raw.ch_names)

    psd_orig = psd.copy()
    psd2 = psd.average()
    assert (psd.data == psd2.data).all()

    psd.crop(fmin=10, fmax=15)
    assert (psd.freqs[0] - 10) < 0.5
    assert (psd.freqs[-1] - 15) < 0.5

    # test for (deep)copy
    psd2 = psd.copy()
    psd2.data[0, 0] = 23
    assert not (psd._data[0, 0] == 23)

    # test to_evoked
    psd2 = psd_orig.copy().average()
    evoked = psd_orig.to_evoked()
    assert isinstance(evoked, mne.Evoked)
    assert (evoked.data == psd2.data).all()

    # test plot_joint()
    # FIXME mne's _plot_psd changes very frequently, so currently we drop
    # this test. Later on - we'll switch to mne spectral objects.
    # psd_orig.plot_joint()
    plt.close('all')

    # psd with Epochs
    # ---------------
    epochs = mne.Epochs(raw, events, event_id, tmin=0., tmax=23.,
                        baseline=None, preload=True)
    psd_epo = compute_psd(epochs, tmin=0.5, tmax=20.5, winlen=2.,
                          step=0.5, events=events, event_id=[11])

    if mne_version > version.parse('0.18'):
        # FIXME mne's _plot_psd changes very frequently, so currently we drop
        # this test. Later on - we'll switch to mne spectral objects.
        # psd_epo.plot(show=False)
        pass

    psd_avg = psd_epo.copy().average()
    assert psd_epo.data.ndim == 3
    assert psd_avg.data.ndim == 2

    arr = np.random.randn(23, 48)
    with pytest.raises(TypeError, match='works only with Raw or Epochs'):
        psd_epo = compute_psd(arr, tmin=0.5, tmax=20.5, winlen=2.,
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
