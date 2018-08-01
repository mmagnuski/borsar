import pytest
import numpy as np
from scipy import sparse

from borsar.cluster import construct_adjacency_matrix


def test_contstruct_adjacency():
    T, F = True, False
    ch_names = list('ABCD')
    adj_correct = np.array([[F, T, T, F],
                            [T, F, T, F],
                            [T, T, F, T],
                            [F, F, T, F]])

    # contruct neighbourhood
    dtypes = [('label', 'O'), ('neighblabel', 'O')]
    arr = np.array([(ch_names[idx], np.array(ch_names)[adj_correct[idx]])
                    for idx in range(adj_correct.shape[0])], dtype=dtypes)

    # test 1, general use case
    adj = construct_adjacency_matrix(arr)
    assert (adj_correct == adj).all()

    # test 2, selected channels
    idx = np.ix_([0, 1, 3], [0, 1, 3])
    adj = construct_adjacency_matrix(arr, ch_names=list('ABD'))
    assert (adj_correct[idx] == adj).all()

    # test 3, as_sparse
    adj = construct_adjacency_matrix(arr, ch_names=list('ABC'), as_sparse=True)
    assert (sparse.coo_matrix(adj_correct[:3, :3]) == adj).toarray().all()

    # test 4, ch_names must be a list
    with pytest.raises(AssertionError):
        construct_adjacency_matrix(arr, ch_names='abc')

    # test 5, ch_names must contain string
    with pytest.raises(AssertionError):
        construct_adjacency_matrix(arr, ch_names=['A', 23, 'C'])

    # test 6, channel not found in neighbours
    with pytest.raises(ValueError):
        construct_adjacency_matrix(arr, ch_names=['A', 'Bi', 'C'])
