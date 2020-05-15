import warnings
import numpy as np
from numbers import Integral

# check mne version number
import mne
from packaging import version
has_new_mne = version.parse(mne.__version__) > version.parse('0.19.3')

if has_new_mne:
    from ._mne_020_modified import plot_topomap
else:
    from ._mne_pre_020_modified import plot_topomap
