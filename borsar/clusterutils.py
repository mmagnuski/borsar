import numpy as np
from borsar.stats import format_pvalue


def _get_units(dimname, fullname=False):
    '''Return unit for specified dimension name.'''
    if not fullname:
        return {'freq': 'Hz', 'time': 's', 'vert': 'vert'}[dimname]
    else:
        return {'freq': 'hertz', 'time': 'seconds',
                'vert': 'vertices'}[dimname]


def _check_stc(clst):
    '''Make sure Clusters has a list of mne.SourceEstimate in stc attribute.'''
    import mne
    if clst.stc is None:
        if clst.vertices is None:
            vert = [clst.src[0]['vertno'], clst.src[1]['vertno']]
        else:
            lh, rh = [clst.vertices[hemi] for hemi in ['lh', 'rh']]
            vert = [clst.src[0]['vertno'][lh], clst.src[1]['vertno'][rh]]

        assert clst.dimnames.index('vert') == 0

        tmin, tstep = 1., 1.
        if len(clst.dimnames) > 1:
            data_single = clst.stat[:, [0]]
        else:
            data_single = clst.stat[:, np.newaxis].copy()

        clst.stc = mne.SourceEstimate(data_single, vertices=vert, tmin=tmin,
                                         tstep=tstep, subject=clst.subject)


# prepare figure colorbar limits
def _get_clim(data, vmin=None, vmax=None, pysurfer=False):
    """Get color limits from data - rounding to steps of 0.5."""
    if vmin is None and vmax is None:
        vmax = np.abs([data.min(), data.max()]).max()
        vmax_round = np.round(vmax)
        if np.abs(np.abs(vmax_round - vmax) - 0.5) < 0.25:
            vmax_round += 0.5 * np.sign(vmax - vmax_round)
        vmin, vmax = -vmax_round, vmax_round
    elif vmin is None:
        vmin = -vmax
    elif vmax is None:
        vmax = -vmin

    if pysurfer:
        return dict(kind='value', lims=[vmin, 0, vmax])
    else:
        return vmin, vmax


# TODO:
# - [ ] _aggregate_cluster aggregates by default everything except the spatial
#       dimension. This would be problematic for spaces like [freq, time]
#       consider adding ``along`` argument.
# - [ ] beware of changing dimension order for some complex "facny index"
#       operations
def _aggregate_cluster(clst, cluster_idx, mask_proportion=0.5,
                       retain_mass=0.65, ignore_space=True, **kwargs):
    '''Aggregate cluster mask and cluster stat map.'''
    do_aggregation = clst.stat.ndim > 1
    cluster_idx = None if len(clst) < cluster_idx + 1 else cluster_idx
    if do_aggregation:
        # find indexing
        idx = clst.get_index(cluster_idx=cluster_idx, retain_mass=retain_mass,
                             ignore_space=ignore_space, **kwargs)
        reduce_axes = tuple(ix for ix in range(1, clst.stat.ndim) if not
                            isinstance(idx[ix], (np.ndarray, list)))
        clst_mask = (clst.clusters[cluster_idx][idx].mean(axis=reduce_axes)
                     >= mask_proportion if cluster_idx is not None else None)
        clst_stat = clst.stat[idx].mean(axis=reduce_axes)
    else:
        # no aggregation
        idx = (slice(None),)
        clst_stat = clst.stat.copy()
        clst_mask = (clst.clusters[cluster_idx] if cluster_idx is not None
                     else None)
    return clst_mask, clst_stat, idx


# TODO - save labels? this would require saving all parameters
#        of cluter reduction and cluster index
def _label_from_cluster(clst, clst_mask):
    '''Get pysurfer label from cluster mask.'''
    import mne
    clst.stc.data[:, 0] = clst_mask
    labels_l, labels_r = mne.stc_to_label(
        clst.stc, src=clst.src, subjects_dir=clst.subjects_dir, smooth=True)
    if isinstance(labels_l, list):
        clst_label = labels_l[0] if len(labels_l) > 0 else labels_r[0]
    else:
        clst_label = labels_l if labels_l is not None else labels_r
    return clst_label


def _prepare_cluster_description(clst, cluster_idx, idx, reduce_axes=None):
    """Prepare text description about cluster range and p value."""
    if clst.stat.ndim > 1:
        if reduce_axes is None:
            # TODO - start at one only when showing spatial map
            reduce_axes = tuple(range(1, clst.stat.ndim))
        time_label = list()
        for dim in reduce_axes:
            uni = _get_units(clst.dimnames[dim])
            coords = ('{} - {}'.format(*clst.dimcoords[dim][idx[dim]][[0, -1]])
                      if clst.dimcoords is not None else 'all')
            time_label.append('{} {}'.format(coords, uni))
        time_label = ', '.join(time_label) + '\n'
    else:
        time_label = ''
    time_label += format_pvalue(clst.pvals[cluster_idx])
    return time_label
