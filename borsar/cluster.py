import numpy as np
from scipy import sparse


def construct_adjacency_matrix(neighbours, ch_names=None, as_sparse=False):
    '''
    Construct adjacency matrix out of neighbours structure (fieldtrip format).
    '''
    # checks for ch_names
    if ch_names is not None:
        assert isinstance(ch_names, list), 'ch_names must be a list.'
        assert all(map(lambda x: isinstance(x, str), ch_names)), \
            'ch_names must be a list of strings'
    else:
        ch_names = neighbours['label'].tolist()

    n_channels = len(ch_names)
    conn = np.zeros((n_channels, n_channels), dtype='bool')

    for ii, chan in enumerate(ch_names):
        ngb_ind = np.where(neighbours['label'] == chan)[0]

        # safty checks:
        if len(ngb_ind) == 0:
            raise ValueError(('channel {} was not found in neighbours.'
                              .format(chan)))
        elif len(ngb_ind) == 1:
            ngb_ind = ngb_ind[0]
        else:
            raise ValueError('found more than one neighbours entry for '
                             'channel name {}.'.format(chan))

        # find connections and fill up adjacency matrix
        connections = [ch_names.index(ch) for ch in neighbours['neighblabel']
                       [ngb_ind] if ch in ch_names]
        chan_ind = ch_names.index(chan)
        conn[chan_ind, connections] = True
    if as_sparse:
        return sparse.coo_matrix(conn)
    return conn
