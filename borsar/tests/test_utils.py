import os
import os.path as op
from warnings import warn

import numpy as np
import mne
from mne.io.pick import channel_indices_by_type as get_ch_types

import pytest
from borsar.channels import get_ch_pos
from borsar.utils import (create_fake_raw, _check_tmin_tmax, detect_overlap,
                          get_info, valid_windows, get_dropped_epochs,
                          find_range, find_index, silent_mne, read_info,
                          write_info, _get_test_data_dir, has_numba,
                          group_mask)


def almost_equal(val1, val2, error=1e-13):
    assert abs(val1 - val2) < 1e-13


def test_find_range():
    vec = np.array([2, 4, 6, 7, 8, 8.5, 9, 12, 18, 25])
    assert find_range(vec, (3.5, 8)) == slice(1, 5)

    ranges = find_range(vec, [(3.5, 8), (10, 20)])
    should_be = [slice(1, 5), slice(6, 9)]
    assert all([x == y for x, y in zip(ranges, should_be)])


def test_find_index():
    vec = np.array([0.1, 0.12, 0.15, 0.3, 0.2, 0.05])
    assert find_index(vec, 0.15) == 2
    assert find_index(vec, 0.18) == 4
    np.testing.assert_array_equal(find_index(vec, [0.07, 0.13, 0.14]),
                                  np.array([5, 1, 2]))


def test_get_info():
    raw = create_fake_raw(n_channels=2, n_samples=5, sfreq=25.)
    assert raw.info == get_info(raw)
    assert raw.info == get_info(raw.info)


def compare_info(info1, info2):
    assert info1['ch_names'] == info2['ch_names']
    assert get_ch_types(info2) == get_ch_types(info2)
    assert info1['sfreq'] == info2['sfreq']

    # compare positions
    pos1, pos2 = get_ch_pos(info1), get_ch_pos(info2)
    pos1 = pos1[~np.isnan(pos1).all(axis=1)]
    pos2 = pos2[~np.isnan(pos2).all(axis=1)]

    def is_empty(pos):
        return pos.shape[0] == 0

    assert (is_empty(pos1) and is_empty(pos2)) or (pos1 == pos2).all()


def test_read_write_info():
    data_dir = _get_test_data_dir()
    raw = create_fake_raw(n_channels=4, n_samples=5, sfreq=25.)
    raw.set_channel_types({'a': 'eeg', 'b': 'ecog', 'c': 'eeg', 'd': 'ecog'})

    fname = op.join(data_dir, 'temp_info.hdf5')
    if op.isfile(fname):
        os.remove(op.join(data_dir, fname))

    write_info(fname, raw.info)
    info = read_info(fname)
    compare_info(info, raw.info)

    raw = mne.io.read_raw_fif(op.join(data_dir, 'rest_sample_data-raw.fif'))
    write_info(fname, raw.info, overwrite=True)
    info = read_info(fname)
    compare_info(info, raw.info)


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
    annot = mne.Annotations(onset, duration, description)
    try:
        raw.set_annotations(annot)
    except AttributeError:
        raw.annotations = annot

    T, F = True, False
    answer = valid_windows(raw, winlen=0.4, step=0.2)
    should_be = np.array([T, F, F, F, F, T, F, F, F, T, T, T, T, F])
    assert (answer == should_be).all()

    answer = valid_windows(raw, tmin=0.4, tmax=1.8, winlen=0.4, step=0.2)
    should_be = should_be[2:-6]
    assert (answer == should_be).all()


def test_get_dropped_epochs():
    raw = create_fake_raw(n_channels=1, n_samples=36, sfreq=10.)
    events = np.zeros((4, 3), dtype='int')
    events[:, -1] = 1
    events[:, 0] = [5, 13, 21, 29]
    annot = mne.Annotations([2.], [1.6], ['BAD_'])
    try:
        raw.set_annotations(annot)
    except AttributeError:
        raw.annotations = annot
    epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.1, tmax=0.6,
                        preload=True)
    assert (np.array([2, 3]) == get_dropped_epochs(epochs)).all()

    epochs.drop([0])
    assert (np.array([0, 2, 3]) == get_dropped_epochs(epochs)).all()


def test_has_numba():
    if_numba = has_numba()
    if if_numba:
        try:
            from numba import jit
            numba_imported = True
        except ImportError:
            numba_imported = False
        assert numba_imported


def test_silent_mne():
    raw = create_fake_raw(n_channels=2, n_samples=10, sfreq=10.)
    mntg = mne.channels.Montage(np.random.random((2, 3)), ['A', 'B'],
                                'eeg', 'fake')
    raw.set_montage(mntg)

    # adding new reference channel without position gives a warning:
    with pytest.warns(Warning):
        mne.add_reference_channels(raw.copy(), ['nose'])

    # ... but not when using silent_mne() context manager:
    with pytest.warns(None) as record:
        with silent_mne():
            mne.add_reference_channels(raw.copy(), ['nose'])

    assert len(record) == 0

    # with `full_silence` no warnings are raised
    with pytest.warns(None) as record:
        with silent_mne(full_silence=True):
            mne.add_reference_channels(raw.copy(), ['nose'])
            warn('annoying warning!', DeprecationWarning)

    assert len(record) == 0


def test_group():
    msk = np.array([False, False, False, False, True, True, True, True, True,
                    False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False])
    grp = group_mask(msk)
    assert (grp == np.array([[4, 8]])).all()

    msk[:2] = True
    grp = group_mask(msk)
    assert (grp == np.array([[0, 1], [4, 8]])).all()

    msk[-3:] = True
    grp = group_mask(msk)
    assert (grp == np.array([[0, 1], [4, 8], [24, 26]])).all()


    msk[12:19] = True
    grp = group_mask(msk)
    assert (grp == np.array([[0, 1], [4, 8], [12, 18], [24, 26]])).all()
