import os.path as op
from scipy.io import loadmat
from numpy.testing import assert_allclose
import mne

import borsar
from borsar.csd import _current_source_density, current_source_density


# load G and H matrices
data_dir = op.join(op.split(borsar.__file__)[0], 'data')
GH = loadmat(op.join(data_dir, 'G_and_H_matrices.mat'))
G, H = GH['G'], GH['H']


def test_csd_base():
    G_orig, H_orig = G.copy(), H.copy()

    # load example ERP
    data = loadmat(op.join(data_dir, 'erp_data.mat'))
    erp = data['erp']
    erp_csd = data['erp_csd']

    # calculate csd
    erp_csd_py = _current_source_density(erp, G, H)

    # make sure erp_csd and erp_csd_py are the same
    # (use lower rtol for a stricter test)
    assert_allclose(erp_csd, erp_csd_py, rtol=5e-9)

    # make sure G and H were not changed during computation
    assert (G == G_orig).all()
    assert (H == H_orig).all()


def test_csd_mne():
    # more of a smoke test at the moment - just testing no errors are risen
    fname = op.join(data_dir, 'rest_sample_data-raw.fif')
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_channels(raw.ch_names[:31])
    raw_csd = current_source_density(raw.copy(), G, H)

    assert (~(raw._data == raw_csd._data)).all()