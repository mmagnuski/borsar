from .utils import (_check_stc, _label_from_cluster, _get_clim,
                    _prepare_cluster_description, _aggregate_cluster)


def plot_cluster_src(clst, cluster_idx=None, aggregate='mean', set_light=True,
                     vmin=None, vmax=None, title=None, figure_size=None,
                     **kwargs):
    '''
    Plot cluster in source space.

    Parameters
    ----------
    clst : Clusters
        Clusters object to use in plotting.
    cluster_idx : int
        Cluster index to plot.
    aggregate : str
        TODO: mean, max, weighted
    set_light : bool
        Whether to change light to preferred setting.
    vmin : float, optional
        Value mapped to minimum in the colormap. Inferred from data by default.
    vmax : float, optional
        Value mapped to maximum in the colormap. Inferred from data by default.
    title : str, optional
        Optional title for the figure.
    figure_size : None | (width, height)
        Size of the figure in pixels.
    **kwargs : additional keyword arguments
        Additional arguments used in aggregation, defining the range to
        aggregate for given dimension. List of two values defines explicit
        range: for example keyword argument `freq=[6, 8]` aggregates the
        6 - 8 Hz range. Float argument between 0. and 1. defines range that is
        dependent on cluster mass. For example `time=0.75` defines time range
        that retains at least 75% of the cluster (calculated along the
        aggregated dimension - in this case time). If no kwarg is passed for
        given dimension then the default value is 0.65 so that range is
        defined to retain at least 65% of the cluster mass.

    Returns
    -------
    brain : surfer.Brain
        Brain object used in plotting.

    Examples
    --------
    > # to plot the first cluster within 8 - 10 Hz
    > clst.plot(cluster_idx=0, freq=[8, 10])
    > # to plot the second cluster selecting frequencies that make up at least
    > # 70% of the cluster mass:
    > clst.plot(cluster_idx=1, freq=0.7)
    '''
    cluster_idx = 0 if cluster_idx is None else cluster_idx
    _check_stc(clst)

    if figure_size is None:
        figure_size = 800

    # get and aggregate cluster mask and cluster stat
    # TODO - first idx then aggregation
    clst_mask, clst_stat, idx = _aggregate_cluster(
        clst, cluster_idx, mask_proportion=0.5, retain_mass=0.65, **kwargs)
    if clst_mask is not None:
        clst_mask = clst_mask.any(axis=0)

    if clst_mask.ndim > 1 and clst_mask.shape[1] == 1:
        clst_mask = clst_mask[:, 0]
        clst_stat = clst_stat[:, 0]

    # create label from cluster
    if clst_mask is not None:
        clst_label = _label_from_cluster(clst, clst_mask.astype('float'))
        use_hemi = clst_label.hemi
    else:
        use_hemi = 'lh' if len(clst.dimcoords[0]['lh']) > 0 else 'rh'

    # prepare 'time' label
    time_label = _prepare_cluster_description(clst, cluster_idx, idx)

    # create pysurfer brain
    clst.stc.data[:, 0] = clst_stat
    clim = _get_clim(clst_stat, vmin=vmin, vmax=vmax, pysurfer=True)
    brain = clst.stc.plot(
        subjects_dir=clst.subjects_dir, hemi=use_hemi, alpha=0.8,
        colormap='RdBu_r', transparent=False, background='white',
        foreground='black', clim=clim, time_label=time_label,
        size=figure_size)

    # add title and cluster label
    if title is not None:
        brain.add_text(0.1, 0.9, title, 'title', font_size=18)
    if clst_mask is not None:
        brain.add_label(clst_label, borders=True, color='w')

    # set light
    if set_light:
        try:
            # works only for mayavi...
            fig = brain._figures[0][0]
            camera_light0 = fig.scene.light_manager.lights[0]
            camera_light0.azimuth = 0.
            camera_light0.elevation = 42.
            camera_light0.intensity = 1.0
        except AttributeError:
            pass
    return brain
