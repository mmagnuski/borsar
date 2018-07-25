import os.path as op
import numpy as np
import mne
from borsar.utils import detect_overlap, get_info


fname = op.join('.', 'data', 'rest_sample_data-raw.fif')
raw, events = mne.io.read_raw_fif(fname)


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
