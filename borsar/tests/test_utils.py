import os.path as op
import numpy as np
import mne
from borsar.utils import detect_overlap, get_info


fname = op.join('.', 'borsar', 'data', 'rest_sample_data-raw.fif')
raw = mne.io.read_raw_fif(fname)


def almost_equal(val1, val2, error=1e-13):
    assert abs(val1 - val2) < 1e-13


def test_get_info():
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


