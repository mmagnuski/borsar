import os.path as op
import numpy as np
import mne

from borsar.utils import (create_fake_raw, _check_tmin_tmax, detect_overlap,
                          get_info, valid_windows, get_dropped_epochs,
                          find_range)


def almost_equal(val1, val2, error=1e-13):
    assert abs(val1 - val2) < 1e-13


def test_find_range():
    vec = np.array([2, 4, 6, 7, 8, 8.5, 9, 12, 18, 25])
    assert find_range(vec, (3.5, 8)) == slice(1, 5)

    ranges = find_range(vec, [(3.5, 8), (10, 20)])
    should_be = [slice(1, 5), slice(6, 9)]
    assert all([x == y for x, y in zip(ranges, should_be)])


def test_get_info():
    raw = create_fake_raw(n_channels=2, n_samples=5, sfreq=25.)
    assert raw.info == get_info(raw)
    assert raw.info == get_info(raw.info)


def test_detect_overlap():
    seg = [0.2, 0.4]
    ann = np.array([[0.0, 0.3]])
    almost_equal(detect_overlap(seg, ann), 0.5)

    seg = [0.1, 0.4]
    ann = np.array([[0.0, 0.2], [0.3, 0.9]])
    almost_equal(detect_overlap(seg, ann), 2/3)

    seg = [0.3, 0.8]
    ann = np.array([[0.3, 0.4], [0.55, 0.65], [0.7, 0.8]])
    almost_equal(detect_overlap(seg, ann), 0.6)

    seg = [0.1, 1.1]
    ann = np.array([[0.3, 0.4], [0.55, 0.65], [0.7, 0.8]])
    almost_equal(detect_overlap(seg, ann), 0.3)

    # test detect overlap in samples
    # as in slices the last sample is not included
    seg = [5, 12]
    ann = np.array([[3, 6], [7, 9], [11, 12]])
    correct_overlap = (1 + 2 + 1) / np.diff(seg)
    assert(detect_overlap(seg, ann, sfreq=1.) == correct_overlap)

    # another test on samples - this time with mne.Annotation
    sfreq, seg = 2., [10, 18]
    descriptions = ['_BAD'] * 3
    onsets = np.array([6, 9, 14]) / sfreq
    durations = np.array([2, 3, 3]) / sfreq
    annot = mne.Annotations(onsets, durations, descriptions)
    correct_overlap = (2 + 3) / np.diff(seg)[0]
    assert(detect_overlap(seg, annot, sfreq=sfreq) == correct_overlap)

    # last test - when mne.Annotations are not present
    annot = None
    assert(detect_overlap(seg, annot, sfreq=sfreq) == 0.)


def test_check_tmin_tmax():
    raw = create_fake_raw(n_channels=2, n_samples=35, sfreq=10.)
    tmin, tmax = -1., 4.
    tmin, tmax, sfreq = _check_tmin_tmax(raw, tmin, tmax)
    assert tmin == 0.
    assert tmax == 3.5
    assert sfreq == 10.


def test_valid_windows():
    raw = create_fake_raw(n_channels=1, n_samples=30, sfreq=10.)

    #                     0    5    10   15   20   25   30
    #                     |    |    |    |    |    |    |
    # annot coverage (x): ooooxxxxxooooooxxxooooooooooxx
    onset = np.array([0.5, 1.5, 28 / 10])
    duration = np.array([0.5, 3 / 10, 2 / 10])
    description = ['_BAD'] * 3
    raw.annotations = mne.Annotations(onset, duration, description)

    T, F = True, False
    answer = valid_windows(raw, winlen=0.4, step=0.2)
    should_be = np.array([T, F, F, F, F, T, F, F, F, T, T, T, T, F])
    assert (answer == should_be).all()

    answer = valid_windows(raw, tmin=0.4, tmax=1.8, winlen=0.4, step=0.2)
    should_be = should_be[2:-6]
    assert (answer == should_be).all()


def test_dropped_index():
    raw = create_fake_raw(n_channels=1, n_samples=36, sfreq=10.)
    events = np.zeros((4, 3), dtype='int')
    events[:, -1] = 1
    events[:, 0] = [5, 13, 21, 29]
    raw.annotations = mne.Annotations([2.], [1.6], ['BAD_'])
    epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.1, tmax=0.6,
                        preload=True)
    assert (np.array([2, 3]) == get_dropped_epoch_index(epochs)).all()

    epochs.drop([0])
    assert (np.array([0, 2, 3]) == get_dropped_epoch_index(epochs)).all()
