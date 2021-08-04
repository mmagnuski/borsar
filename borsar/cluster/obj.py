import warnings
import numpy as np

from .viz import plot_cluster_contribution, plot_cluster_chan
from ._viz3d import plot_cluster_src
from .utils import (_get_mass_range, _cluster_selection, _index_from_dim,
                    _ensure_correct_info, _handle_dims, _full_dimname,
                    _prepare_dimindex_plan, _update_plan, _find_mass_index)
from .checks import (_clusters_safety_checks, _clusters_chan_vert_checks,
                     _check_dimnames_kwargs, _check_dimname_arg,
                     _check_description)
from ..utils import import_hdf5


def read_cluster(fname, subjects_dir=None, src=None, info=None):
    '''
    Read standard Clusters .hdf5 file and return Clusters object.
    You need to pass correct subjects_dir and src to `read_cluster` if your
    results are in source space or correct info if your results are in channel
    space.

    Parameters
    ----------
    fname : str
        File path for the file to read.
    subjects_dir : str, optional
        Path to Freesurfer subjects directory.
    src : mne.SourceSpaces, optional
        Source space that the results are represented in.
    info : mne.Info, optional
        Channel space that the results are represented in.

    Returns
    -------
    clst : Clusters
        Cluster results read from file.
    '''
    h5io = import_hdf5()
    # subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)
    data_dict = h5io.read_hdf5(fname)
    clst = Clusters(
        data_dict['stat'], data_dict['clusters'], data_dict['pvals'],
        dimnames=data_dict['dimnames'], dimcoords=data_dict['dimcoords'],
        subject=data_dict['subject'], subjects_dir=subjects_dir, info=info,
        src=src, description=data_dict['description'])
    return clst


# TODO - consider empty lists/arrays instead of None when no clusters...
#      - [ ] sometime: make only stat necessary
#      - [ ] add reading and writing to FieldTrip cluster structs
class Clusters(object):
    '''
    Container for results of cluster-based tests.

    Parameters
    ----------
    stat : ndarray
        Statistical map of the analysis. Usual dimensions are: space,
        space x time, space x frequencies, space x frequencies x
        time where space corresponds to channels or vertices (in the source
        space).
    clusters : list of boolean ndarrays | boolean ndarray
        List of boolean masks - one per cluster. The masks should match the
        dimensions of the `stat` ndarray. Each mask describes which elements
        are members of given cluster. Alternatively - one boolean array where
        first dimension corresponds to consecutive clusters. When no clusters
        were found this can be an empty numpy array, an empty list or None.
    pvals : list or array of float
        List/array of p values corresponding to consecutive clusters in
        `clusters`. If no clusters were found this can be an empty numpy array,
        an empty list or None.
    dimnames : list of str, optional
        List of dimension names. For example ``['chan', 'freq']`` or ``['vert',
        'time']``. The length of `dimnames` has to mach ``stat.ndim``.
        If 'chan' dimension name is given, you also need to provide ``mne.Info``
        corresponding to the channels via ``info`` keyword argument.
        If ``'vert'`` dimension name is given, you also need to provide
        ``mne.SourceSpaces`` via ``src`` keyword argument. You also have to
        specify ``subject`` and ``subjects_dir`` via respective keyword
        arguments.
    dimcoords : list of arrays, optional
        List of arrays, where each array contains coordinates (labels) for
        consecutive elements in corresponding dimension. For example if your
        `stat` represents channels by frequencies then a) `dimcoords[0]` should
        have length of `stat.shape[0]` and its consecutive elements should
        represent channel names while b) `dimcoords[1]` should have length of
        `stat.shape[1]` and its consecutive elements should represent centers
        of frequency bins (in Hz).
        When first dimension corresponds to channels and all channels provided
        in ``info`` object appear in the same order in the data - there is no
        need to specify channel names in ``dimcoords``. In such situation the
        respective ``dimcoords`` for channels can be ``None``.
        When the first dimension corresponds to the vertices in the source
        space then the ``dimcoords`` should specify vertex indices with respect
        to the provided ``src`` (``mne.SourceSpaces``). These indices have to
        be with respect to vertices used in the source space, not all possible
        vertices present in the original brain model. Because of that filling
        ``dimcoords`` with vertex indices should only be used when the analysis
        was not conducted on the whole ``src`` space, but on sub-selection of
        vertices (for example: only frontal regions). Otherwise ``dimcoords``
        corresponding to ``'vert'`` dimension can be left as ``None``.
    info : mne.Info, optional
        When using channel space ('chan' is one of the dimnames) you need to
        provide information about channel position in ``mne.Info`` object (for
        example ``epochs.info``).
    src : mne.SourceSpaces, optional
        When using source space ('vert' is one of the dimnames) you need to
        pass ``mne.SourceSpaces`` corresponding to the data.
    subject : str, optional
        When using source space ('vert' is one of the dimnames) you need to
        pass a subject name (name of the freesurfer directory with file for
        given subject).
    subjects_dir : str, optional
        When using source space ('vert' is one of the dimnames) you need to
        pass a Freesurfer subjects directory (path to the folder constraining
        subjects as sub-folders).
    description : str | dict, optional
        Optional description of the Clusters - for example analysis parameters
        and some other additional details.
    '''
    def __init__(self, stat, clusters=None, pvals=None, dimnames=None,
                 dimcoords=None, info=None, src=None, subject=None,
                 subjects_dir=None, description=None,
                 safety_checks=True):

        if safety_checks:
            # basic safety checks
            clusters, pvals = _clusters_safety_checks(
                clusters, pvals, stat, dimnames, dimcoords, description)

            # check channel or source space
            dimcoords = _clusters_chan_vert_checks(dimnames, dimcoords, info,
                                                   src, subject, subjects_dir)

            # check polarity of clusters
            polarity = ['neg', 'pos']
            self.polarity = ([polarity[int(stat[cl].mean() > 0)]
                              for cl in clusters]
                             if pvals is not None else None)

            if pvals is not None:
                pvals = np.asarray(pvals)

                # sort by p values if necessary
                pval_sort = np.argsort(pvals)
                if not (pval_sort == np.arange(pvals.shape[0])).all():
                    clusters = clusters[pval_sort]
                    pvals = pvals[pval_sort]
                    self.polarity = [self.polarity[idx] for idx in pval_sort]

        # create attributes
        self.subjects_dir = subjects_dir
        self.description = description
        self.dimcoords = dimcoords
        self.clusters = clusters
        self.dimnames = dimnames
        self.subject = subject
        self.pvals = pvals
        self.stat = stat
        self.info = info
        self.stc = None
        self.src = src

        # FIXME: find better way for this (maybe during safety checks earlier)
        #        maybe just a private constructor?
        if self.info is not None and safety_checks:
            _ensure_correct_info(self)

# - [ ] more tests for select (n_points was not working)
# - [ ] add warning if all clusters removed
# - [ ] consider select to _not_ work inplace or make sure all methods
#       work this way (we'll see about that...)
    def select(self, p_threshold=None, percentage_in=None, n_points_in=None,
               n_points=None, selection=None, **kwargs):
        '''
        Select clusters by p value threshold or its location in the data space.

        .. note:: ``select`` method works in-place.

        Parameters
        ----------
        p_threshold : None | float
            Threshold for cluster-level p value. Only clusters associated with
            a p value lower than this threshold are selected. Defaults to None
            which does not select clusters by p value.
        percentage_in : None | float
            Select clusters by percentage participation in range of the data
            space specified in ``**kwargs``. For example
            ``clst.select(percentage_in=0.15, freq=(3, 7))`` selects only those
            clusters that have at least 15% of their mass in 3 - 7 Hz frequency
            range. Defaults to None which does not select clusters by their
            participation in data space.
        n_points_in : None | int
            Select clusters by number of their minimum number of data points
            that lie in the range of the data specified in ``**kwargs``. For
            example `clst.select(n_points_in=25, time=[0.2, 0.35])` selects
            only those clusters that contain at least 25 points within
            0.2 - 0.35 s time range. Defaults to None which does not select
            clusters by number of points participating in data space.
        n_points : None | int
            Select clusters by their minimum number of data points. For example
            `clst.select(n_points=5)` selects only those clusters that have at
            least 5 data points. Default to None which does not perform the
            selection.
        selection : list-like of int | list-like of boolean | None
            Which clusters to select. Used when the user knows which clusters
            should be selected instead of using criteria like p value or
            cluster surface.
        **kwargs : additional arguments
            Additional arguments when selection is meant to be performed only
            based on some subspace of the effects. Defines the points to use in
            the selection (if argument value is a list of float) or the range
            to use for the dimension specified by the argument name. Tuple of
            two values defines explicit range: for example keyword argument
            ``freq=(6, 8)`` performs selection on the 6 - 8 Hz range. Float
            argument between 0. and 1. defines range that is dependent on
            cluster mass or extent. For example ``time=0.75`` defines time
            range that retains at least 75% of the cluster extent (calculated
            along the specified dimension - in this case time).
            If no kwarg is passed for given dimension then the whole extent of
            that dimension is used in the selection.
            Using ``**kwargs`` makes sense only when performing selection via
            ``percentage_in`` or ``n_points_in``. Otherwise ``*kwargs`` are
            ignored.

        Returns
        -------
        clst : borsar.cluster.Clusters
            Selected clusters.
        '''
        if self.clusters is None:
            return self

        if selection is not None:
            # FIXME additional checks for ``selection``
            self = _cluster_selection(self, selection)

        # select clusters by p value threshold
        if p_threshold is not None:
            sel = self.pvals < p_threshold
            self = _cluster_selection(self, sel)

        if n_points is not None:
            dims = np.arange(self.stat.ndim) + 1
            cluster_size = self.clusters.sum(axis=tuple(dims))
            sel = cluster_size > n_points
            self = _cluster_selection(self, sel)

        if (len(kwargs) > 0 or n_points_in is not None) and len(self) > 0:
            # kwargs check should be in a separate function
            if len(kwargs) > 0:
                _check_dimnames_kwargs(self, allow_lists=False, **kwargs)
            plan, kwargs = _prepare_dimindex_plan(self.dimnames, **kwargs)
            dim_idx = _index_from_dim(self.dimnames, self.dimcoords, plan,
                                      **kwargs)

            dims = np.arange(self.stat.ndim) + 1
            clst_idx = (slice(None),) + dim_idx
            cluster_sel_size = self.clusters[clst_idx].sum(axis=tuple(dims))

            sel = np.ones(len(self), dtype='bool')
            if n_points_in is not None:
                sel = cluster_sel_size >= n_points_in
            if percentage_in is not None:
                cluster_size = self.clusters.sum(axis=tuple(dims))
                sel = ((cluster_sel_size / cluster_size >= percentage_in)
                       & sel)
            self = _cluster_selection(self, sel)
        return self

    def copy(self, deep=True):
        '''
        Copy the Clusters object.

        The lists/arrays are not copied however. The SourceSpaces are always
        copied because they often change when plotting.

        Returns
        -------
        clst : Clusters
            Copied Clusters object.
        deep : bool
            Whether to deep-copy Clusters object. Defaults to ``True``.
        '''
        if deep:
            from copy import deepcopy
            clst = deepcopy(self)
        else:
            clst = Clusters(
                self.stat, self.clusters, self.pvals, self.dimnames,
                self.dimcoords, info=self.info, src=self.src,
                subject=self.subject, subjects_dir=self.subjects_dir,
                description=self.description, safety_checks=False)
            clst.stc = self.stc if self.stc is None else self.stc.copy()
            clst.polarity = self.polarity
        return clst

    def __len__(self):
        '''Return number of clusters in Clusters.'''
        return len(self.clusters) if self.clusters is not None else 0

    def __iter__(self):
        '''Initialize iteration.'''
        self._current = 0
        return self

    def __next__(self):
        '''
        Get next cluster in iteration. Allows to do things like:
        >>> for clst in clusters:
        >>>     clst.plot()
        '''
        if self._current >= len(self):
            raise StopIteration
        clst = Clusters(self.stat, self.clusters[self._current],
                        self.pvals[[self._current]], self.dimnames,
                        self.dimcoords, info=self.info, src=self.src,
                        subject=self.subject, subjects_dir=self.subjects_dir,
                        description=self.description, safety_checks=False)
        clst.stc = self.stc  # or .copy()?
        clst.polarity = [self.polarity[self._current]]
        self._current += 1
        return clst

    def __repr__(self):
        '''Clusters text representation.'''
        base_txt = '<borsar.Clusters  |  {} clusters in {} space>'
        n_clusters = len(self)
        dimnames = [_full_dimname(dimname) for dimname in self.dimnames]
        dimnames = ' x '.join(dimnames)
        return base_txt.format(n_clusters, dimnames)

    def save(self, fname, description=None, overwrite=False):
        '''
        Save Clusters to hdf5 file.

        Parameters
        ----------
        fname : str
            Path to save the file in.
        description : str, dict
            Additional description added when saving. When passed overrides
            the description parameter of Clusters.
        '''
        h5io = import_hdf5()

        if description is None:
            description = self.description
        else:
            _check_description(description)

        data_dict = {'clusters': self.clusters, 'pvals': self.pvals,
                     'stat': self.stat, 'dimnames': self.dimnames,
                     'dimcoords': self.dimcoords, 'subject': self.subject,
                     'description': description}
        h5io.write_hdf5(fname, data_dict, overwrite=overwrite)

    # TODO - consider weighting contribution by stat value
    #      - consider contributions along two dimensions
    # - [ ] change to use some parts of _find_mass_index (moved out)
    def get_contribution(self, cluster_idx=None, along=None, norm=True,
                         idx=None):
        '''
        Get mass percentage contribution to given clusters along specified
        dimension.

        Parameters
        ----------
        cluster_idx : int | array of int, optional
            Indices of clusters to get contribution of. Default is to calculate
            contribution for all clusters.
        along : int | str, optional
            Dimension along which the clusters contribution should be
            calculated. Default is to calculate along the first dimension.
        norm : bool, optional
            Whether to normalize contributions. Defaults to `True`.
        idx : tuple
            Tuple for numpy array indexing that selects some subspace of
            interest.

        Returns
        -------
        contrib : array
            One-dimensional array containing float values (percentage
            contributions) if `norm=True` or integer values (number of
            elements) if `norm=False`.
        '''
        return_many = True
        if cluster_idx is None:
            cluster_idx = np.arange(len(self))
        if (not isinstance(cluster_idx, (list, np.ndarray))
            and (isinstance(cluster_idx, int)
                 or np.issubdtype(cluster_idx, np.integer))):
            cluster_idx = [cluster_idx]
            return_many = False

        along = 0 if along is None else along
        dim_idx = _check_dimname_arg(self, along)

        # one line for each cluster
        all_dims = list(range(self.stat.ndim + 1))
        all_dims.remove(0)
        all_dims.remove(dim_idx + 1)

        clst = self.clusters[cluster_idx]
        if idx is not None:
            clst = clst[(slice(None),) + idx]

        contrib = clst.sum(axis=tuple(all_dims))
        if norm:
            contrib = contrib / contrib.sum(axis=-1, keepdims=True)

        if return_many:
            return contrib
        else:
            return contrib[0]

    # TODO: consider continuous vs discontinuous limits
    # TODO: consider merging more with get_index?
    # TODO: rename to `get_limits()`
    def get_cluster_limits(self, cluster_idx, retain_mass=0.65,
                           dims=None, **kwargs):
        '''
        Find cluster limits based on percentage of cluster mass contribution
        to given dimensions.

        Parameters
        ----------
        cluster_idx : int
            Cluster index to find limits of.
        retain_mass : float
            Percentage of cluster mass to retain in cluster limits for
            dimensions not specified with keyword arguments (see `kwargs`).
            Defaults to 0.65.
        dims : list-like of int | list-like of str | None, optional
            Which dimensions to check. Defaults to None which checks all
            dimensions except spatial.
        **kwargs : additional keyword arguments
            Additional arguments defining the cluster extent to retain along
            specified dimensions. Float argument between 0. and 1. - defines
            range that is dependent on cluster mass. For example ``time=0.75``
            defines time range limits that retain at least 75% of the cluster
            (calculated along given dimension - in this case time). If no kwarg
            is passed for given dimension then the default value of 0.65 is
            used - so that cluster limits are defined to retain at least 65%
            of the relevant cluster mass.

        Returns
        -------
        limits : tuple of slices
            Found cluster limits expressed as a slice for each dimension,
            grouped together in a tuple. Can be used in indexing stat
            (`clst.stat[limits]`) or original data for example.
            The spatial dimension, if not ignored, is returned as a numpy array
            of indices.
        '''
        # TODO: add safety checks
        has_space = (self.dimnames is not None
                     and self.dimnames[0] in ['vert', 'chan'])

        if dims is None:
            check_dims = list(range(self.stat.ndim))
            if has_space:
                check_dims.remove(0)
        else:
            if isinstance(dims[0], str):
                check_dims = _handle_dims(self, dims)
            else:
                check_dims = dims

        limits = list()
        for dim_idx in range(self.stat.ndim):
            if dim_idx in check_dims:
                # use _find_mass_index_for_dim() !
                dimname = self.dimnames[dim_idx]
                mass = kwargs[dimname] if dimname in kwargs else retain_mass
                contrib = self.get_contribution(cluster_idx, along=dimname)

                # current method - start at max and extend
                adj = not (dim_idx == 0 and has_space)
                lims = _get_mass_range(contrib, mass, adjacent=adj)
                limits.append(lims)
            else:
                limits.append(slice(None))
        return tuple(limits)

    # TODO - do not use get_index() for limit calculations - division of labor!
    # TODO - make sure that when one dim is specified with coords and other
    #        with mass to retain, the mass is taken only from the part
    #        specified? (this is done in get_limits with `idx` variable)
    def get_index(self, cluster_idx=None, ignore_dims=None, retain_mass=0.65,
                  **kwargs):
        '''
        Get indices (tuple of slices) selecting a specified range of data.

        Parameters
        ----------
        cluster_idx : int | None, optional
            Cluster index to use when calculating index. Dimensions that are
            not addressed using range keyword arguments will be sliced by
            maximizing cluster mass along that dimensions with mass to retain
            given either in relevant keyword argument or if not such keyword
            argument `retain_mass` value is used. See `kwargs`.
        ignore_dims : str | list of str | None
            Dimensions to ignore when finding cluster extent. Returned indices
            corresponding to these dimensions will be empty slices (thus
            including the whole extent for given dimension). ``None`` defaults
            to the spatial dimension.
        retain_mass : float, optional
            If cluster_idx is passed then dimensions not addressed using keyword
            arguments will be sliced to maximize given cluster's retained mass.
            The default value is 0.65. See `kwargs`.
        **kwargs : additional arguments
            Additional arguments used in aggregation, defining the points to
            select (if argument value is a list of float) or the range to
            aggregate for the dimension specified by the argument name. Tuple
            of two values defines explicit range: for example keyword argument
            ``freq=(6, 8)`` aggregates the 6 - 8 Hz range. List of floats
            defines specific points to pick: for example ``time=[0.1, 0.2]``
            selects time points corresponding to 0.1 and 0.2 seconds.
            Float argument between 0. and 1. defines range that is dependent on
            cluster mass or extent. For example ``time=0.75`` defines time
            range that retains at least 75% of the cluster extent (calculated
            along the aggregated dimension - in this case time). If no kwarg is
            passed for given dimension then the default value is ``0.65``.
            This means that the range for such dimension is defined to retain
            at least 65% of the cluster extent.

        Returns
        -------
        idx : tuple of slices
            Tuple of slices selecting the requested range of the data. Can be
            used in indexing stat (`clst.stat[idx]`) or clusters (
            `clst.clusters[:, *idx]`) for example.
        '''
        _check_dimnames_kwargs(self, check_dimcoords=True, **kwargs)
        plan, kwargs = _prepare_dimindex_plan(self.dimnames, **kwargs)
        idx = _index_from_dim(self.dimnames, self.dimcoords, plan, **kwargs)
        plan, kwargs = _update_plan(self.dimnames, plan, kwargs,
                                    select=retain_mass, ignore=ignore_dims)

        # when retain_mass is specified it is used to get ranges for
        # dimensions not addressed with kwargs
        # FIXME - error if mass_indexing specified but no cluster_idx
        if cluster_idx is not None:
            # check cluster limits only for non-indexed dimensions
            idx = _find_mass_index(self, cluster_idx, plan, kwargs, idx)
        return idx

    # maybe rename to `plot mass`?
    def plot_contribution(self, dims=None, picks=None, axis=None, **kwargs):
        '''
        Plot contribution of clusters along specified dimension.

        Parameters
        ----------
        dims : str | int
            Dimension along which to calculate contribution.
        picks : list-like | None, optional
            Cluster indices whose contributions should be shown.
        axis : matplotlib Axes | None, optional
            Matplotlib axis to plot in.

        Returns
        -------
        axis : matplotlib Axes
            Axes with the plot.
        '''

        return plot_cluster_contribution(self, dims, picks=picks, axis=axis,
                                         **kwargs)

    def plot(self, cluster_idx=None, dims=None, set_light=True, vmin=None,
             vmax=None, mark_kwargs=None, figure_size=None, **kwargs):
        '''
        Plot cluster.

        Parameters
        ----------
        cluster_idx : int
            Cluster index to plot.
        dims : str | list of str | None
            Dimensions to plot. Defaults to ``None`` which plots only the
            spatial dimension.
        vmin : float, optional
            Value mapped to minimum in the colormap. Inferred from data by
            default.
        vmax : float, optional
            Value mapped to maximum in the colormap. Inferred from data by
            default.
        title : str, optional
            Optional title for the figure.
        mark_kwargs : dict | None, optional
            Keyword arguments for ``Topo.mark_channels``. For example:
            ``mark_kwargs={'markersize'=3}`` to change the size of the markers.
            ``None`` defaults to ``{'markersize=5'}``.
        **kwargs : additional arguments
            Additional arguments used in aggregation, defining the points to
            select (if argument value is a list of float) or the range to
            aggregate for the dimension specified by the argument name. Tuple
            of two values defines explicit range: for example keyword argument
            ``freq=(6, 8)`` aggregates the 6 - 8 Hz range. List of floats
            defines specific points to pick: for example ``time=[0.1, 0.2]``
            selects time points corresponding to 0.1 and 0.2 seconds.
            Float argument between 0. and 1. defines range that is dependent on
            cluster mass or extent. For example ``time=0.75`` defines time
            range that retains at least 75% of the cluster extent (calculated
            along the aggregated dimension - in this case time). If no kwarg is
            passed for given dimension then the default value is ``0.65``.
            This means that the range for such dimension is defined to retain
            at least 65% of the cluster extent.

        Returns
        -------
        topo : borsar.viz.Topo | pysurfer.Brain
            Figure object used in plotting - borsar.viz.Topo for channel-level
            plotting and pysurfer.Brain for plots on brain surface.

        Examples
        --------
        > # to plot the first cluster within 8 - 10 Hz
        > clst.plot(cluster_idx=0, freq=(8, 10))
        > # to plot the second cluster selecting frequencies that make up at
        > # least 70% of the cluster mass:
        > clst.plot(cluster_idx=1, freq='70%')
        '''
        if self.dimnames is None:
            raise TypeError('To plot the data you need to construct the '
                            'cluster using the dimnames keyword argument.')
        if self.dimnames[0] == 'vert':
            return plot_cluster_src(self, cluster_idx, vmin=vmin, vmax=vmax,
                                    figure_size=figure_size,
                                    set_light=set_light, **kwargs)
        else:
            return plot_cluster_chan(self, cluster_idx, dims=dims, vmin=vmin,
                                     vmax=vmax, mark_kwargs=mark_kwargs,
                                     **kwargs)

    @property
    def cluster_polarity(self):
        """Just for the deprecation period - returns polarity of clusters."""
        warnings.warn('``Clusters.cluster_polarity`` is now ``Clusters.pola'
                      'rity``. ``.cluster_polarity`` will be deprecated.',
                      DeprecationWarning)
        return self.polarity
