import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
import pytest

from borsar.channels import select_channels
from borsar.freq import compute_rest_psd
from borsar.utils import find_range
from borsar.viz import Topo, _extract_topo_channels

fname = op.join('.', 'borsar', 'data', 'rest_sample_data-raw.fif')
raw = mne.io.read_raw_fif(fname)

def test_topo():
    psd, freq = compute_rest_psd(raw, tmin=1., tmax=3., winlen=1.)
    alpha = find_range(freq, (8, 12))
    alpha_topo = psd[:, alpha].mean(axis=-1)

    # currently a smoke test
    topo = Topo(alpha_topo, raw.info, show=False)
    topo.set_linewidth(2.5)
    topo.solid_lines()
    topo.remove_levels(0.)
    topo.mark_channels([1, 2, 3, 6, 8], markersize=10.)
    ch, pos = _extract_topo_channels(topo.axis)

    fig, ax = plt.subplots()
    with pytest.raises(RuntimeError):
        ch, pos = _extract_topo_channels(ax)

    topo = Topo(alpha_topo, raw.info, axes=ax, show=False)
    assert topo.axis == ax
    assert topo.fig == fig

    # various selections
    for selection in ['frontal', 'asy_frontal', 'asy_all']:
        select = select_channels(raw, selection=selection)
        select = ([select] if isinstance(select, list) else
                  [select['left'], select['right']])
        for sel in select:
            sel_info = mne.pick_info(raw.info, picks=sel)
            topo = Topo(alpha_topo[sel], sel_info, show=False)
