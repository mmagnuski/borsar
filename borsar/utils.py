import os
import os.path as op
import warnings

import numpy as np
from contextlib import contextmanager


def find_range(vec, ranges):
    '''
    Find specified ranges in an ordered vector and return them as slices.

    Parameters
    ----------
    vec : numpy.ndarray
        Vector of sorted values.
    ranges: list of tuples/lists | two-element list/tuple
        Range (or ranges) to be found.

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


# - [ ] if vals is np.ndarray try to format output in the right shape
def find_index(vec, vals):
    '''
    Find indices of values in `vec` that are closest to requested values
    `vals`.

    Parameters
    ----------
    vec : numpy.ndarray
        Vector of values.
    vals: list of values | value
        Values to find closest representatives of in the `vec` vector.

    Returns
    -------
    idx : numpy array of int | int
        Indices of `vec` values closest to `vals`. If one value was passed in
        `vals` then `idx` will also be one value. If two or more values were
        passed in `vals` the output is a numpy array of indices.
    '''
    one_in = False
    if not isinstance(vals, (list, tuple, np.ndarray)):
        one_in = True
        vals = [vals]

    outlist = [np.abs(vec - x).argmin() for x in vals]
    if one_in:
        return outlist[0]
    else:
        return np.array(outlist)


def group_mask(mask):
    '''Groups a 1D boolean mask into array of starts and stops of ``True``
    value ranges.

    Parameters
    ----------
    mask : 1D boolean array
        Boolean array to group.

    Returns
    -------
    groups : 2D int array
        Array of True ranges starts (first column) and stops (last column).
        Each row represents one range of True values.
    '''
    changes = np.where(np.diff(mask))[0]
    frnt = [-1] if mask[0] else np.array([], dtype='int')
    bck = [mask.shape[0] - 1] if mask[-1] else np.array([], dtype='int')
    groups = np.concatenate([frnt, changes, bck])
    groups = groups.reshape((-1, 2))
    groups[:, 0] += 1
    return groups


def get_info(inst):
    '''Simple helper function that returns Info whatever mne object it gets.'''

    from mne import Info
    if isinstance(inst, Info):
        return inst
    else:
        return inst.info


def write_info(fname, info, overwrite=False):
    """Save Info object to ``.hdf5`` file.

    Parameters
    ----------
    fname : str
        Name of the file.
    info : mne.Info
        Info object to save.
    """
    from .channels import get_ch_pos
    from mne.utils import _validate_type
    from mne.io.pick import channel_indices_by_type

    try:
        # mne < 1.0
        from mne.externals import h5io
    except ModuleNotFoundError:
        # mne > 1.0 requires separate installation of h5io
        import h5io

    # make sure the types are correct
    _validate_type(fname, 'str', item_name='fname')
    _validate_type(info, 'info', item_name='info')

    # extract type info
    tps = channel_indices_by_type(info)

    # remove empty dict keys
    for k in list(tps.keys()):
        if len(tps[k]) == 0:
            tps.pop(k)

    has_types = list(tps.keys())
    ch_type = has_types[0] if len(has_types) == 1 else tps

    # save to .hdf5
    data_dict = {'ch_names': info['ch_names'], 'sfreq': info['sfreq'],
                 'ch_type': ch_type, 'pos': get_ch_pos(info)}
    h5io.write_hdf5(fname, data_dict, overwrite=overwrite)


def read_info(fname):
    """Read Info object from ``.hdf5`` file.

    Parameters
    ----------
    fname : str
        Name of the file.

    Returns
    -------
    info : mne.Info
        Info object read from file.
    """
    import mne
    try:
        # mne < 1.0
        from mne.externals import h5io
    except ModuleNotFoundError:
        # mne > 1.0 requires separate installation of h5io
        import h5io

    mne.utils._validate_type(fname, 'str', item_name='fname')

    # read file
    data_dict = h5io.read_hdf5(fname)
    ch_names = data_dict['ch_names']

    # parse ch_type
    if isinstance(data_dict['ch_type'], dict):
        ch_type = [None] * len(ch_names)
        for type, idxs in data_dict['ch_type'].items():
            for idx in idxs:
                ch_type[idx] = type
    else:
        ch_type = data_dict['ch_type']

    # check channel positions
    mntg = None
    pos = data_dict['pos']
    if pos is not None and not np.isnan(pos).all():
        try:
            mntg = mne.channels.Montage(pos, ch_names, 'unknown',
                                        np.arange(pos.shape[0]))
        except AttributeError:
            ch_pos = {chnm: chpos for chnm, chpos in zip(ch_names, pos)}
            mntg = mne.channels.make_dig_montage(ch_pos=ch_pos)

    # create info
    try:
        info = mne.create_info(ch_names, data_dict['sfreq'],
                               ch_types=ch_type, montage=mntg, verbose=False)
    except TypeError:
        info = mne.create_info(ch_names, data_dict['sfreq'],
                               ch_types=ch_type, verbose=False)
        info.set_montage(mntg)
    return info


def detect_overlap(segment, annot, sfreq=None):
    '''
    Detect what percentage of given segment is overlapping with annotations.

    Parameters
    ----------
    segment : list or 1d array
        Two-element list or array of [start, stop] values.
    annot : mne.Annotation of 2d numpy array
        Annotations or 2d array of N x 2 (start, stop) values.
    sfreq : float
        Sampling frequency (default: None). If not None segment is assumed to
        be given in samples. `annot` is transformed to samples using sfreq if
        it is of mne.Annotations type. If `annot` is np.ndarray then it is
        transformed to samples only if its dtype is not 'int64' or 'int32'.

    Returns
    -------
    overlap : float
        Percentage overlap in 0 - 1 range.
    '''
    samples = sfreq is not None

    # FIXME - the branching below seems overly complex
    # for convenience we accept mne.Annotation objects and numpy arrays:
    if not isinstance(annot, np.ndarray):
        if annot is None:
            return 0.

        # if we didn't get numpy array we assume it's mne.Annotation
        # and convert it to N x 2 (N x start, end) array:
        onset = (annot.onset if not samples else
                 np.round(annot.onset * sfreq).astype('int'))
        duration = (annot.duration if not samples else
                    np.round(annot.duration * sfreq).astype('int'))
        annot_arr = np.hstack([onset[:, np.newaxis], onset[:, np.newaxis]
                               + duration[:, np.newaxis]])
    else:
        if not samples:
            annot_arr = annot
        else:
            in_samples = (annot.dtype is np.dtype('int64') or
                          annot.dtype is np.dtype('int32'))
            # FIXME - if not in_samples throw an error or issue a warning
            annot_arr = (annot if in_samples else
                         np.round(annot * sfreq).astype('int'))

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
#         if so - use that function (check later)
def _check_tmin_tmax(raw, tmin, tmax):
    sfreq = raw.info['sfreq']
    lowest_tmin = raw.first_samp / sfreq
    highest_tmax = (raw.last_samp + 1) / sfreq
    tmin = lowest_tmin if tmin is None or tmin < lowest_tmin else tmin
    tmax = highest_tmax if tmax is None or tmax > highest_tmax else tmax
    return tmin, tmax, sfreq


def valid_windows(raw, tmin=None, tmax=None, winlen=2., step=1.):
    '''
    Test which moving windows overlap with annotations.

    Parameters
    ----------
    raw : mne.Raw
        Data to use.
    tmin : flot | None
        Start time for the moving windows. Defaults to None which means start
        of the raw data.
    tmax : flot | None
        End time for the moving windows. Defaults to None which means end of
        the raw data.
    winlen : float
        Window length in seconds. Defaults to 2.
    step : float
        Window step in seconds. Defaults to 1.

    Returns
    -------
    valid : boolean numpy array
        Whether the moving windows overlap with annotations. Consecutive values
        inform whether consecutive windows overlap with any annotation.
    '''
    # get and select bad annotations
    annot = raw.annotations
    sel = [idx for idx, desc in enumerate(annot.description)
           if desc.lower().startswith('bad')]
    annot = annot[sel]

    tmin, tmax, sfreq = _check_tmin_tmax(raw, tmin, tmax)
    step = int(round(step * sfreq))
    winlen = int(round(winlen * sfreq))
    tmin_smp, tmax_smp = int(round(tmin * sfreq)), int(round(tmax * sfreq))
    n_windows = int((tmax_smp - tmin_smp - winlen + step) / step)
    valid = np.zeros(n_windows, dtype='bool')
    for win_idx in range(n_windows):
        start = tmin_smp + win_idx * step
        segment = [start, start + winlen]
        overlap = detect_overlap(segment, annot, sfreq=sfreq)
        valid[win_idx] = overlap == 0.
    return valid


def create_fake_raw(n_channels=4, n_samples=100, sfreq=125.):
    '''
    Create fake raw signal for testing.

    Parameters
    ----------
    n_channels : int, optional
        Number of channels in the fake raw signal. Defaults to 4.
    n_samples : int, optional
         Number of samples in the fake raw singal. Defaults to 100.
    sfreq : float, optional
        Sampling frequency of the fake raw signal. Defaults to 125.

    Returns
    -------
    raw : mne.io.RawArray
        Created raw array.
    '''
    import mne
    from string import ascii_letters
    ch_names = list(ascii_letters[:n_channels])
    data = np.zeros((n_channels, n_samples))
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    return mne.io.RawArray(data, info)


def get_dropped_epochs(epochs):
    '''
    Get indices of dropped epochs from `epochs.drop_log`.

    Parameters
    ----------
    epochs : mne Epochs instance
        Epochs to get dropped indices from.

    Returns
    -------
    dropped_epochs : 1d numpy array
        Array containing indices of dropped epochs.
    '''
    current_epoch = 0
    dropped_epochs = list()

    for info in epochs.drop_log:
        if 'IGNORED' not in info:
            if len(info) > 0:
                dropped_epochs.append(current_epoch)
            current_epoch += 1

    return np.array(dropped_epochs)


# - CONSIDER silent(mne=True) or silent(full=True)
@contextmanager
def silent_mne(full_silence=False):
    '''
    Context manager that silences warnings from mne-python.
    '''
    import mne

    log_level = mne.set_log_level('error', return_old_level=True)

    if full_silence:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    else:
        yield

    mne.set_log_level(log_level)


def has_numba():
    """Check if numba is available."""
    try:
        from numba import jit
        return True
    except ImportError:
        return False


def import_hdf5():
    """Import h5py module if available."""
    try:
        # mne < 1.0
        from mne.externals import h5io
        h5io.read_hdf5
    except (ModuleNotFoundError, AttributeError):
        # mne > 1.0 requires separate installation of h5io
        import h5io

    return h5io


def _get_test_data_dir():
    '''Get test data directory.'''
    from . import __path__ as borsar_dir
    return op.join(borsar_dir[0], 'data')


def download_test_data():
    '''Download additional test data from dropbox.'''
    import zipfile
    try:
        from mne.utils import _fetch_file
        use_pooch = False
    except ImportError:
        import pooch
        use_pooch = True

    # check if test data exist
    data_dir = _get_test_data_dir()
    check_files = ['alpha_range_clusters.hdf5', 'DiamSar-eeg-oct-6-fwd.fif',
                   op.join('fsaverage', 'bem', 'fsaverage-ico-5-src.fif'),
                   'chan_alpha_range.hdf5', 'test_clustering.npy',
                   'DiamSar_023_rest_raw.fif', 'GabCon-48_epo.fif']
    if all([op.isfile(op.join(data_dir, f)) for f in check_files]):
        return

    # set up paths
    fname = 'temp_file.zip'
    destination = op.join(data_dir, fname)
    download_link = ('https://www.dropbox.com/sh/l4scs37524lb3pa/'
                     'AABCak4jORjgridWwHlwjhMHa?dl=1')

    # download the file
    if use_pooch:
        hash = ('d88b76a6af113f9faefe4ce616ad17cf0a7fbc99f3febdcebfb2b1580eaaaa62')  # noqa: E501
        pooch.retrieve(url=download_link, known_hash=hash,
                       path=data_dir, fname=fname)
    else:
        _fetch_file(download_link, destination, print_destination=True,
                    resume=True, timeout=30.)

    # unzip and extract
    # TODO - optionally extract only the missing files
    zip_ref = zipfile.ZipFile(destination, 'r')
    zip_ref.extractall(data_dir)
    zip_ref.close()

    # remove the zipfile
    os.remove(destination)
