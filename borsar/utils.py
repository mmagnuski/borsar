import numpy as np


def find_range(vec, ranges):
    '''
    Find specified ranges in an ordered vector and retur them as slices.

    Parameters
    ----------
    vec : numpy array
        Vector of sorted values.
    ranges: list of tuples/lists or two-element list/tuple

    Returns
    -------
    slices : slice or list of slices
        Slices representing the ranges. If one range was passed the output
        is a slice. If two or more ranges were passed the output is a list
        of slices.
    '''
    assert isinstance(ranges, (list, tuple))
    assert len(ranges) > 0
    one_in = False
    if not isinstance(ranges[0], (list, tuple)) and len(ranges) == 2:
        one_in = True
        ranges = [ranges]

    slices = list()
    for rng in ranges:
        start, stop = [np.abs(vec - x).argmin() for x in rng]
        slices.append(slice(start, stop + 1)) # including last index
    if one_in:
        slices = slices[0]
    return slices


def get_info(inst):
    '''
    Simple helper function that returns Info whatever mne object it gets
    (including mne.Info itself).
    '''
    from mne import Info
    if isinstance(inst, Info):
        return inst
    else:
        return inst.info


def detect_overlap(segment, annot):
    '''
    Detect what percentage of given segment is overlapping with bad annotations.

    Parameters
    ----------
    segment : list or 1d array
        Two-element list or array of [start, stop] values.
    annot : mne.Annotation of 2d numpy array
        Annotations or 2d array of N x 2 (start, stop) values.

    Returns
    -------
    overlap : float
        Percentage overlap in 0 - 1 range.
    '''
    # for convenience we accept mne.Annotation objects and numpy arrays:
    if not isinstance(annot, np.ndarray):
        # if we didn't get numpy array we assume it's mne.Annotation
        # and convert it to N x 2 (N x start, end) array:
        annot_arr = np.hstack([annot.onset[:, np.newaxis],
                               annot.onset[:, np.newaxis]
                               + annot.duration])
    else:
        annot_arr = annot

    # checks for boundary relationships
    ll_beleq = annot_arr[:, 0] <= segment[0]
    hh_abveq = annot_arr[:, 1] >= segment[1]

    # if any annot's lower edge (le) is below or equal segments lower edge
    # and its higher edge (he) is above segments higher edge
    # then annot includes segment and the coverage is 100%
    if (ll_beleq & hh_abveq).any():
        return 1.

    # otherwise we perform more checks
    hl_abv = annot_arr[:, 1] > segment[0]
    segment_length = np.diff(segment)[0]
    overlap = 0.

    # if any annot's le is below or equal segments le and its he is
    # above segments le - the the overlap is from segments le up to annot he
    check = ll_beleq & hl_abv
    if check.any():
        # there should be only one such annot (we assume non-overlapping annot)
        overlap += (annot_arr[:, 1][check][0] - segment[0]) / segment_length

    ll_abv = ~ll_beleq
    hh_bel = ~hh_abveq
    # if any annot's le is above segments le and its he is
    # below segments he - the the annot is within the segment
    # and the overlap is from annot le up to annot he
    # (there can be multiple such annots)
    check = ll_abv & hh_bel
    if check.any():
        overlap += (np.diff(annot_arr[check]) / segment_length).sum()

    # the only remaining case is when annot he is above segments he
    # and its le is above segments le but below segments he
    lh_bel = annot_arr[:, 0] < segment[1]
    check = hh_abveq & ll_abv & lh_bel
    if check.any():
        overlap += (segment[1] - annot_arr[:, 0][check][0]) / segment_length

    return overlap


# FIXME - add warnings etc.
# FIXME - there should be some mne function for that,
#         if so - use that function
def _check_tmin_tmax(raw, tmin, tmax):
    sfreq = raw.info['sfreq']
    lowest_tmin = raw.first_samp / sfreq
    highest_tmax = raw.last_samp / sfreq
    tmin = lowest_tmin if tmin is None else tmin
    tmax = raw.last_samp if tmin is None else tmin
    return tmin, tmax, sfreq


def valid_windows(raw, tmin=None, tmax=None, winlen=2., step=1.):
    '''
    Do uzupełnienia przez Nastię :)
    '''
    tmin, tmax, sfreq = _check_tmin_tmax(raw, tmin, tmax)

    n_windows = 10
    return np.ones((n_windows), dtype='bool')
