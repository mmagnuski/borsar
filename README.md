[![CircleCI](https://dl.circleci.com/status-badge/img/gh/mmagnuski/borsar/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/mmagnuski/borsar/tree/master)
[![Coverage Status](https://codecov.io/gh/mmagnuski/borsar/branch/master/graph/badge.svg)](https://codecov.io/gh/mmagnuski/borsar)

Various tools, objects and functions for EEG/MEG data analysis and visualisation. Some functionality that is available here may
be later moved to [mne-python](https://martinos.org/mne/dev/index.html).

`borsar` includes:
* `PSD` object for manipulation of power spectral results
* `Clusters` object for storage, manipulation and plotting of clutser-based results, both in channel and sourcee space
* efficient regression for multichannel data (`compute_regression_t`)
* `cluster_based_regression` to perform regression tests in cluster-based permutation framework
* numpy and numba implementations of cluster-based permutation tests in 3d space (for example in `channels x frequency x time` space) with optional filtering by minimum number of adjacent channels (`min_adj_ch`, equivalent of `minnbchan` in fieldtrip).
* `Topo` object for topomap plots that retains the topomap state, allows to mark channels, efficiently update data, change contour line width and style for one or multiple topomaps.


## Installation
`borsar` is not yet released on `PyPI` so to install you have to download it from GitHub using pip in the following way:
```
pip install git+https://github.com/mmagnuski/borsar
```
or, if you plan to frequently update the dev version and contribute to `borsar`, install by cloning the repo with
git and installing in dev mode:
```
cd where_you_want_to_download_borsar
git clone https://github.com/mmagnuski/borsar
cd borsar
python setup.py develop
```
both methods require you to have [git](https://git-scm.com/) installed.

## Documentation
Go to the [online documentation](https://mmagnuski.github.io/borsar.github.io/index.html) for more information about usage examples or full API docs.
:construction: be warned that documentation is under contstruction :construction:
