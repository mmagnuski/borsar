import numpy as np
from .utils import get_info


def get_ch_names(inst):
    '''Get channel names from mne object instance.

    Works both for Raw or Epochs but also for Info.
    '''
    import mne
    if isinstance(inst, mne.Info):
        return inst['ch_names']
    else:
        return inst.ch_names


def get_ch_pos(inst):
    '''Extract xyz position of channels from mne object instance.'''
    info = get_info(inst)
    chan_pos = [info['chs'][i]['loc'][:3] for i in range(len(info['chs']))]
    return np.array(chan_pos)


def find_channels(inst, names):
    '''Find channel indices by their names in mne object instance.'''
    one_name = False
    ch_names = get_ch_names(inst)
    if isinstance(names, str):
        one_name = True
    finder = (lambda val: ch_names.index(val)
              if val in ch_names else None)
    return finder(names) if one_name else list(map(finder, names))


def select_channels(inst, select='all'):
    '''
    Gives indices of channels selected by a text keyword.

    Parameters
    ----------
    inst : mne Raw | mne Epochs | mne Evoked | mne TFR | mne Info
        Mne object with `ch_names` and `info` attributes or just the mne Info
        object.
    select : str
        Can be 'all', 'frontal' or 'pairs'. If 'asy_' is prepended to the
        select string then selected channels are grouped by mirror positions
        on the x axis (left vs right).

    Returns
    -------
    selection : numpy int array or dict of numpy int arrays
        Indices of the selected channels. If 'asy_' was in the select string
        then selection is a dictionary of indices, where selection['left']
        gives channels on the left side of the scalp and selection['right']
        gives right-side homologues of the channels in selection['left'].
    '''
    if select == 'all':
        return np.arange(len(get_ch_names(inst)))
    elif 'asy' in select and 'all' in select:
        return homologous_pairs(inst)

    if 'frontal' in select:
        ch_pos = get_ch_pos(inst)
        frontal_idx = np.where(ch_pos[:, 1] > 0)[0]
        if 'asy' in select:
            hmlg = homologous_pairs(inst)
            sel = np.in1d(hmlg['left'], frontal_idx)
            return {side: hmlg[side][sel] for side in ['left', 'right']}
        else:
            return frontal_idx

    if select == 'asy_pairs':
        pary = dict(left=['F3', 'F7'], right=['F4', 'F8'])
        return {k: find_channels(inst, pary[k]) for k in pary.keys()}


def homologous_pairs(inst):
    '''
    Construct homologous channel pairs based on channel names or positions.

    Parameters
    ----------
    inst : mne object instance
        Mne object like mne.Raw or mne.Epochs.

    Returns
    -------
    selection: dict of {str -> list of int} mappings
        Dictionary mapping hemisphere ('left' or 'right') to array of channel
        indices.
    '''

    ch_names = get_ch_names(inst)
    ch_pos = get_ch_pos(inst)

    labels = ['right', 'left']
    selection = {l: list() for l in labels}
    has_1020_names = 'Cz' in ch_names and 'F3' in ch_names

    if has_1020_names:
        # find homologues by channel names
        left_chans = ch_pos[:, 0] < 0
        y_ord = np.argsort(ch_pos[left_chans, 1])[::-1]
        check_chans = [ch for ch in list(np.array(ch_names)[left_chans][y_ord])
                       if 'z' not in ch]

        for ch in check_chans:
            chan_base = ''.join([char for char in ch if not char.isdigit()])
            chan_value = int(''.join([char for char in ch if char.isdigit()]))

            if (chan_value % 2) == 1:
                selection['left'].append(ch_names.index(ch))
                homologous_ch = chan_base + str(chan_value + 1)
                selection['right'].append(ch_names.index(homologous_ch))
    else:
        # channel names do not come from 10-20 system
        # constructing homologues from channel position
        # (this will not work for digitized channel positions)
        left_chans = ch_pos[ch_pos[:, 0] < 0]
        y_ord = np.argsort(left_chans[:, 1])[::-1]
        left_chans = left_chans[y_ord]

        for idx, pos in enumerate(left_chans):
            inv_x = ch_pos[:, 0] == -pos[0]
            same_y = ch_pos[:, 1] == pos[1]
            found = np.where(inv_x & same_y)[0]
            if len(found) > 0:
                identical = (ch_pos == pos[np.newaxis, :]).all(axis=1)
                orig_idx = np.where(identical)[0][0]
                selection['left'].append(orig_idx)
                selection['right'].append(found[0])

    selection['left'] = np.array(selection['left'])
    selection['right'] = np.array(selection['right'])
    return selection
