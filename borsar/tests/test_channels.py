import os.path as op
import numpy as np
import mne

from borsar.channels import (homologous_pairs, get_ch_pos, select_channels,
                             find_channels)

fname = op.join('.', 'borsar', 'data', 'rest_sample_data-raw.fif')
raw = mne.io.read_raw_fif(fname)


def test_find_channels():
    channels = ['Oz', 'Fz']
    channels_idx = [raw.ch_names.index(chan) for chan in channels]
    assert channels_idx[0] == find_channels(raw, channels[0])
    assert all([x == y for x, y in zip(channels_idx,
                                       find_channels(raw, channels))])


def selchan_test_helper(raw, sel, ch_pos):
    assert (ch_pos[sel['left'], 1] == ch_pos[sel['right'], 1]).all()
    assert (ch_pos[sel['left'], 0] == -ch_pos[sel['right'], 0]).all()

    f3, f4 = find_channels(raw, ['F3', 'F4'])
    where_f3 = np.where(sel['left'] == f3)[0]
    where_f4 = np.where(sel['right'] == f4)[0]
    assert len(where_f3) > 0 and len(where_f4) > 0
    assert where_f3[0] == where_f4[0]


def test_select_channels():
    assert (select_channels(raw, 'all') == np.arange(len(raw.ch_names))).all()

    ch_pos = get_ch_pos(raw)
    sel = select_channels(raw, 'frontal')
    assert (ch_pos[sel, 1] > 0.).all()

    sel = select_channels(raw, 'asy_frontal')
    for side in ['left', 'right']:
        assert (ch_pos[sel[side], 1] > 0).all()
    selchan_test_helper(raw, sel, ch_pos)

    # select_channels should also work on info
    sel_info = select_channels(raw.info, 'asy_frontal')
    assert all([(sel[k] == sel_info[k]).all() for k in sel.keys()])

    sel = select_channels(raw, 'asy_all')
    selchan_test_helper(raw, sel, ch_pos)


def test_homologous_pairs():
    homo_dict1 = homologous_pairs(raw)

    new_channel_names = {ch: str(idx) for idx, ch in enumerate(raw.ch_names)}
    raw.rename_channels(new_channel_names)
    homo_dict2 = homologous_pairs(raw)

    assert (homo_dict1['left'] == homo_dict2['left']).all()
    assert (homo_dict1['right'] == homo_dict2['right']).all()
