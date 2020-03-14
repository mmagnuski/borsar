import numpy as np


def _clusters_safety_checks(clusters, pvals, stat, dimnames, dimcoords,
                            description):
    '''Perform basic type and safety checks for Clusters.'''
    # check clusters when it is a list
    if isinstance(clusters, list):
        n_clusters = len(clusters)
        if n_clusters > 0:
            cluster_shapes = [clst.shape for clst in clusters]
            all_shapes_equal = all([clst_shp == cluster_shapes[0]
                                    for clst_shp in cluster_shapes])
            if not all_shapes_equal:
                raise ValueError('All clusters have to be of the same '
                                 'shape.')
            all_arrays_bool = all([clst.dtype == 'bool' for clst in clusters])
            if not all_arrays_bool:
                raise TypeError('All clusters have to be boolean arrays.')
            clusters = np.stack(clusters, axis=0)
        else:
            clusters = None

    # check stat
    if not isinstance(stat, np.ndarray):
        raise TypeError('`stat` must be a numpy array.')

    # check clusters shape along stat shape
    if isinstance(clusters, np.ndarray):
        n_clusters = clusters.shape[0]
        if n_clusters > 0:
            if not stat.shape == clusters.shape[1:]:
                msg = ('Every cluster has to have the same shape as stat. '
                       '`stat` has shape {} while `clusters` have shape {}.')
                raise ValueError(msg.format(stat.shape, clusters.shape[1:]))
        else:
            clusters = None
    elif clusters is not None:
        raise TypeError('`clusters` has to be either a list of arrays or one '
                        'array with the first dimension corresponding to '
                        'clusters or None if no clusters were found.')

    if clusters is None:
        # TODO: maybe warn if no clusters but pvals is not None/empty
        pvals = None
    elif not isinstance(pvals, (list, np.ndarray)):
        raise TypeError('`pvals` has to be a list of floats or numpy array.')
        # check if each element of list is float and array is of dtype float

    if dimnames is not None:
        if not isinstance(dimnames, list):
            raise TypeError('`dimnames` must be a list of dimension names.'
                            'Got {}.'.format(type(dimnames)))
        which_str = np.array([isinstance(el, str) for el in dimnames])
        if not which_str.all():
            other_type = type(dimnames[np.where(~which_str)[0][0]])
            raise TypeError('`dimnames` must be a list of strings, but some '
                            'of the elements in the list you passed are not '
                            'strings, for example: {}.'.format(other_type))
        if not len(dimnames) == stat.ndim:
            raise ValueError('Length of `dimnames` must be the same as number'
                             ' of dimensions in `stat`.')
        if ('chan' in dimnames and not dimnames.index('chan') == 0
            or 'vert' in dimnames and not dimnames.index('vert') == 0):
            msg = ('If using channels ("chan" dimension name) or vertices ('
                   'for source space - "vert" dimension name) - it must be '
                   'the first dimension in the `stat` array and therefore the'
                   ' first dimension name in `dimnames`.')
            raise ValueError(msg)
    if dimcoords is not None:
        if not isinstance(dimcoords, list):
            raise TypeError('`dimcoords` must be a list of dimension '
                            'coordinates. Got {}.'.format(type(dimcoords)))
        if not len(dimcoords) == stat.ndim:
            raise ValueError('Length of `dimcoords` must be the same as number'
                             ' of dimensions in the `stat`.')
        dims = list(range(len(dimcoords)))
        if dimnames[0] in ['chan', 'vert']:
            dims.pop(0)
        equal_len = [stat.shape[idx] == len(dimcoords[idx]) for idx in dims]
        if not all(equal_len):
            msg = ('The length of each dimension coordinate (except for the '
                   'spatial dimension - channels or vertices) has to be the '
                   'same as the length of the corresponding dimension in '
                   '`stat` array.')
            raise ValueError(msg)
    _check_description(description)
    return clusters, pvals


def _check_description(description):
    '''Validate if description is of correct type.'''
    if description is not None:
        if not isinstance(description, (str, dict)):
            raise TypeError('Description has to be either a string or a dict'
                            'ionary, got {}.'.format(type(description)))


# TODO - [ ] move to clusterutils?
def _clusters_chan_vert_checks(dimnames, dimcoords, info, src, subject,
                               subjects_dir):
    '''Safety checks for Clusters spatial dimension.'''
    import mne
    if dimnames is not None and 'chan' in dimnames:
        if info is None or not isinstance(info, mne.Info):
            raise TypeError('You must pass an `mne.Info` in order to use '
                            '"chan" dimension. Use `info` keyword argument.')
    elif dimnames is not None and 'vert' in dimnames:
        if src is None or not isinstance(src, mne.SourceSpaces):
            raise TypeError('You must pass an `mne.SourceSpaces` in order to '
                            'use "vert" dimension. Use `src` keyword'
                            ' argument.')
        if subject is None or not isinstance(subject, str):
            raise TypeError('You must pass a subject string in order to '
                            'use "vert" dimension. Use `subject` keyword'
                            ' argument.')
        if subjects_dir is None:
            subjects_dir = mne.utils.get_subjects_dir()
        if subjects_dir is None or not isinstance(subjects_dir, str):
            raise TypeError('You must pass a `subjects_dir` freesurfer folder'
                            ' name in order to use "vert" dimension. Use '
                            '`subjects_dir` keyword argument.')

        vertices = dimcoords[0]
        if vertices is not None:
            # FIXME - move to separate function
            # check against left and right hemi
            vert_num_lh = src[0]['vertno'].shape[0]
            vert_num_rh = src[1]['vertno'].shape[0]
            vert_num_all = vert_num_lh + vert_num_rh

            if isinstance(vertices, np.ndarray):
                vert_idx_in_src = (vertices < vert_num_all).all()
                if not vert_idx_in_src:
                    msg = ('Some vertex indices exceed the available source '
                           'space size. Number of vertices in the src (lh + '
                           'rh) = {:d}, while maximum index in the ``vertice'
                           's`` = {:d}.')
                    raise ValueError(msg.format(vert_num_all, vertices.max()))

                # turn to lh, rh dictionary
                lh_mask = vertices < vert_num_lh
                vertices = {'lh': vertices[lh_mask],
                            'rh': vertices[~lh_mask] - vert_num_lh}
                dimcoords[0] = vertices
            elif isinstance(vertices, dict):
                assert 'lh' in vertices and 'rh' in vertices
                vert_idx_in_src = ((vertices['lh'] < vert_num_all).all()
                                   and (vertices['rh'] < vert_num_rh).all())
                if not vert_idx_in_src:
                    msg = ('Some vertex indices exceed the available source '
                           'space size. Number of vertices in the src is: lh '
                           '= {:d}; rh = {:d}, while maximum index in the``ve'
                           'rtices``: lh = {:d}; rh = {:d}.')
                    formatted_msg = msg.format(vert_num_lh, vert_num_rh,
                                               vertices['lh'].max(),
                                               vertices['rh'].max())
                    raise ValueError(formatted_msg)

    return dimcoords


def _check_dimname_arg(clst, dimname):
    '''Check dimension name and find its index.'''
    if not isinstance(dimname, (str, int)):
        raise TypeError('Dimension argument has to be string (dimension name) '
                        'or int (dimension index).')
    if isinstance(dimname, str):
        if clst.dimnames is None:
            raise TypeError('Clusters has to have `dimnames` attribute to use '
                            'operations on named dimensions.')
        if dimname not in clst.dimnames:
            raise ValueError('Clusters does not seem to have the dimension you'
                             ' requested. You asked for "{}", while Clusters '
                             'has the following dimensions: {}.'.format(
                                 dimname, ', '.join(clst.dimnames)))
        idx = clst.dimnames.index(dimname)
    else:
        if not (dimname >= 0 and dimname < clst.stat.ndim):
            raise ValueError('Dimension, if integer, must be greater or equal '
                             'to 0 and lower than number of dimensions of the '
                             'statistical map. Got {}'.format(dimname))
        idx = dimname
    return idx


def _check_dimnames_kwargs(clst, check_dimcoords=False, ignore_dims=None,
                           split_range_mass=False, allow_lists=True, **kwargs):
    '''Ensure that **kwargs are correct dimnames and dimcoords.

    ignore_dims : list of int?
        Dimensions to ignore.
    split_range_mass : bool
        Whether to separate range (normal) and mass indices.
    allow_lists : bool
        Whether to allow passing lists or numpy arrays to specified dimensions.
    '''
    if clst.dimnames is None:
        raise TypeError('Clusters has to have dimnames to use operations '
                        'on named dimensions.')
    if check_dimcoords and clst.dimcoords is None:
        raise TypeError('Clusters has to have dimcoords to use operations '
                        'on named dimensions.')

    if split_range_mass:
        normal_indexing = kwargs.copy()
        mass_indexing = dict()

    for dim in kwargs.keys():
        if dim not in clst.dimnames:
            msg = ('Could not find requested dimension {}. Available '
                   'dimensions: {}.'.format(dim, ', '.join(clst.dimnames)))
            raise ValueError(msg)

        if not allow_lists and isinstance(kwargs[dim], (list, np.ndarray)):
            msg = ('Use of lists/numpy arrays of datapoints are not supported'
                   ' in this context. Use range ((min, max) tuple) or mass/'
                   'extent (float).')
            raise TypeError(msg)

        if split_range_mass:
            dval = kwargs[dim]
            # TODO - more elaborate checks
            dim_type = ('range' if isinstance(dval, (list, tuple, np.ndarray))
                        else 'mass' if isinstance(dval, float) else None)
            if dim_type == 'mass':
                mass_indexing[dim] = dval
                normal_indexing.pop(dim)
            elif dim_type is None:
                raise TypeError('The values used in dimension name indexing '
                                'have to be either specific points (list or '
                                'array of values), ranges (tuple of two values'
                                ') or cluster extent to retain (float), got '
                                '{} for dimension {}.'.format(dval, dim))
    if split_range_mass:
        return normal_indexing, mass_indexing
