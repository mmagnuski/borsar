from copy import deepcopy
import numpy as np
import mne
from mne.viz.epochs import plot_epochs_psd

try:
    has_epochs_mixin = True
    from mne.utils import GetEpochsMixin
except ImportError:
    has_epochs_mixin = False

from mne.channels.channels import UpdateChannelsMixin

try:
    from mne.channels.channels import ContainsMixin
except ImportError:
    from mne.io.meas_info import ContainsMixin

if has_epochs_mixin:
    mixins = (ContainsMixin, GetEpochsMixin, UpdateChannelsMixin)
else:
    mixins = (ContainsMixin, UpdateChannelsMixin)

from .utils import valid_windows, find_range, find_index
from .viz import Topo


# [ ] event_id should support dict!
# [ ] change name to compute_psd_raw
# [ ] use annotations when event_id was passed as str or list of str
def compute_rest_psd(raw, events=None, event_id=None, tmin=None, tmax=None,
                     winlen=2., step=None, padto=None, picks=None):
    '''
    Compute power spectral density (psd) for given time segments for all
    channels of given raw file. The segments (if more than one) are averaged
    taking into account the artifact-free range of each segment. Signal during
    _BAD annotations (parts of signal marked as artifacts) is excluded by
    default in `mne.time_frequency.psd_welch` which can lead to some segments
    'donating' more data than others. This has to be taken into account during
    segments averaging - so the segments are weighted with the percentage of
    welch windows that had artifact free data (and thus were not rejected in
    `mne.time_frequency.psd_welch`).

    Parameters
    ----------
    raw: mne.Raw
        Raw file to use.
    events: numpy array | None
        Mne events array of shape (n_events, 3). If None (default) `tmin` and
        `tmax` are not calculated with respect to events but the whole time
        range of the `raw` file.
    event_id: list | numpy array
        Event types (IDs) to use in defining segments for which psd is
        computed. If None (default) and events were passed all event types are
        used.
    tmin: float
        Lower edge of each segment in seconds. If events are given the lower
        edge is with respect to each event. If events are not given only one
        segment is used and `tmin` denotes the lower edge of the whole `raw`
        file.
    tmax: float
        Higher edge of each segment in seconds. If events are given the higher
        edge is with respect to each event. If events are not given only one
        segment is used and `tmax` denotes the higher edge of the whole `raw`
        file.
    winlen: float
        Length of the welch window in seconds.
    step: float
        Step of the welch window in seconds.

    Returns
    -------
    psd : numpy array
        Power spectral density in <FIX: check shape> matrix.
    freqs : numpy array
        Frequencies for which psd was calculated.
    '''
    from mne.time_frequency import psd_welch

    sfreq = raw.info['sfreq']
    step = winlen / 4 if step is None else step
    n_per_seg, n_overlap, n_fft = _psd_welch_input_seconds_to_samples(
        raw, winlen, step, padto)

    # FIXME - add warning when event_id is something and events are not passed
    #         or warn and use annotations
    if events is not None:
        # select events
        got_event_id = event_id is not None
        if got_event_id:
            if isinstance(event_id, int):
                event_id = [event_id]
        else:
            event_id = np.unique(events[:, -1])
        events_of_interest = np.in1d(events[:, -1], event_id)
        events = events[events_of_interest]

        psd_dict = {ev: list() for ev in event_id}
        psd_weights = {ev: list() for ev in event_id}
        for event_idx in range(events.shape[0]):
            # find event type, define tmin and tmax based on event
            event_type = events[event_idx, -1]
            event_onset = events[event_idx, 0] / sfreq
            this_tmin = event_onset + tmin
            this_tmax = event_onset + tmax

            # compute psd for given segment, then add to psd_dict
            try:
                this_psd, freqs = psd_welch(
                    raw, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg,
                    tmin=this_tmin, tmax=this_tmax, picks=picks, average=None,
                    verbose=False)
                this_psd = np.nanmean(this_psd, axis=-1)
            except TypeError:
                # old psd function, no average kwarg...
                this_psd, freqs = psd_welch(
                    raw, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg,
                    tmin=this_tmin, tmax=this_tmax, picks=picks, verbose=False)

            # compute percent of windows that do not overlap with artifacts
            # these constitute weights used in averaging
            weight = (valid_windows(raw, tmin=this_tmin, tmax=this_tmax,
                                    winlen=winlen, step=step)).mean()
            if not weight == 0:
                psd_dict[event_type].append(this_psd)
                psd_weights[event_type].append(weight)

        # use np.average() with weights to compute wieghted average
        psd = {k: np.average(np.stack(psd_dict[k], axis=0),
                             weights=psd_weights[k], axis=0)
               for k in psd_dict.keys()}
        if len(event_id) == 1 and got_event_id:
            psd = psd[event_id[0]]

        return psd, freqs
    else:
        # use only tmin, tmax, a lot easier
        return psd_welch(raw, n_fft=n_fft, n_overlap=n_overlap,
                         tmin=tmin, tmax=tmax)


# - [x] make the default to be simple fft
# - [ ] welch args: proj=False, n_jobs=1, reject_by_annotation=True,
#                   verbose=None
def compute_psd(inst, tmin=None, tmax=None, winlen=None, step=None, padto=None,
                events=None, event_id=None, picks=None):
    """Compute power spectral density on Raw or Epochs.

    Parameters
    ----------
    inst : mne.io._BaseRaw | mne.Epochs
        mne Epochs or Raw object to compute psd on.
    tmin : float | None
        Time (in seconds) marking the start of the time segment used in PSD
        calculation.
    tmax : float | None
        Time (in seconds) marking the end of the time segment used in PSD
        calculation.
    winlen : float
        Welch window length (in seconds). The default is 2 seconds.
    step : float | None
        Welch window step interval (in seconds).
    padto : float | None
        Length in seconds to which each Welch window should be zero-padded.
    events : numpy.ndarray | None
        mne events array (n_events by 3). Used only when event-related PSD
        calculation is used.
    event_id : int | list of int | array of int
        The id of the event to consider. If a list, all events with the IDs
        specified in the list are used. If None, all events will be used.
    picks : list/array of int | list of str | None
        Channels to use in PSD calculation. The default (``None``) uses all
        data channels.

    Returns
    -------
    psd : borsar.freq.PSD
        PowerSpectralDensity (PSD) object.
    """
    from mne.time_frequency import psd_welch

    if tmin is None:
        tmin = inst.times[0]
    if tmax is None:
        tmax = inst.times[-1]
    if winlen is None:
        winlen = tmax - tmin

    # FIXME - maybe check: if one long winlen and at least some bad annotations
    #         there should be some warning in compute_psd_raw if all data is
    #         nan due to annotations
    step = winlen / 4 if step is None else step
    if isinstance(inst, mne.BaseEpochs):
        n_per_seg, n_overlap, n_fft = _psd_welch_input_seconds_to_samples(
            inst, winlen, step, padto)
        psd, freq = psd_welch(inst, tmin=tmin, tmax=tmax, n_fft=n_fft,
                              picks=picks, n_per_seg=n_per_seg,
                              n_overlap=n_overlap)
        # FIXME: this will need fixing:
        #  * inst has events and event_id
        #  * can the user pass events? should it be ignored?
        if event_id is not None:
            # check which epochs were selected
            chosen_events = (list(event_id.values())
                             if isinstance(event_id, dict) else event_id)
            msk = np.in1d(inst.events[:, -1], chosen_events)
            this_inst = inst[msk]

            events = this_inst.events
            event_id = this_inst.event_id
            metadata = this_inst.metadata
        else:
            events = inst.events
            event_id = inst.event_id
            metadata = inst.metadata

    elif isinstance(inst, mne.io.BaseRaw):
        psd, freq = compute_rest_psd(inst, events=events, event_id=event_id,
                                     tmin=tmin, tmax=tmax, winlen=winlen,
                                     step=step)
        metadata = None
    else:
        raise TypeError('`compute_psd` works only with Raw or Epochs data '
                        'formats, got {}'.format(type(inst)))

    # construct PSD object
    try:
        from mne.selection import pick_types
    except ModuleNotFoundError:
        from mne import pick_types

    picks_int = pick_types(inst.info, eeg=True, selection=picks)
    info = mne.pick_info(inst.info, sel=picks_int)

    psd = PSD(psd, freq, info, events=events, event_id=event_id,
              metadata=metadata)

    return psd


def _psd_welch_input_seconds_to_samples(inst, winlen, step, padto):
    sfreq = inst.info['sfreq']
    padto = winlen if padto is None else padto

    def _convert_s_to_smp(s, sfreq):
        return int(round(s * sfreq))

    n_per_seg, step_smp, n_fft = [_convert_s_to_smp(x, sfreq) for x in
                                  [winlen, step, padto]]
    n_overlap = n_per_seg - step_smp if step_smp > 0 else 0
    return n_per_seg, n_overlap, n_fft


# - [ ] LATER: add .get_peak()
class PSD(*mixins):
    def __init__(self, psd, freqs, info, events=None, event_id=None,
                 metadata=None):
        '''Construct PowerSpectralDensity (PSD) object.

        Parameters
        ----------
        psd : numpy.ndarray
            Channels by frequencies (or epochs by channels by frequencies)
            matrix of spectrum values.
        freqs : numpy.ndarray
            Vector of frequencies.
        info : mne.Info
            Info object with channel names and positions.

        Attributes
        ----------
        data : numpy.ndarray
            The data array of either (channels, frequencies) shape or
            (epochs, channels, frequencies).
        freqs : numpy.ndarray
            Frequencies.
        info : mne.Info
            Info object with channel names and positions.
        ch_names : list of str
            Channel names.
        '''
        # add check for psd dimensions
        if (psd.ndim < 2) or (psd.ndim > 3):
            raise ValueError('`psd` array has to be 3d (epochs, channels, '
                             'frequencies) or 2d (channels, frequencies), got '
                             'array with {} dimensions.'.format(psd.ndim))
        self._has_epochs = False if psd.ndim == 2 else True
        self._data = psd
        self.freqs = freqs
        self.info = info

        # make sure that event, event_id and metadata are used only when
        # _has_epochs
        if self._has_epochs:
            self.preload = True
            self.events = events
            self.event_id = event_id
            self._metadata = metadata

        # otherwise - disable indexing
        # FIXME

    def __repr__(self):
        '''String representation of the PSD object.'''
        base_str = '<borsar.freq.PSD {}'
        dim_str = '{} channels, {} frequencies)'
        pre_str = '({} epochs, ' if self._has_epochs else '('
        dim_str = (pre_str + dim_str).format(*self.data.shape)
        base_str = base_str.format(dim_str)

        freq_str = ', {:g} - {:g} Hz'.format(*self.freqs[[0, -1]])
        base_str = base_str + freq_str + '>'
        return base_str

    # - [ ] check if the way we copy docs from mne makes this so slow when
    #       reloading (autoreload in ipython/spyder/notebook)...
    def plot(self, fmin=0, fmax=None, proj=False, picks=None, ax=None,
             color='black', xscale='linear', area_mode='std', area_alpha=0.33,
             dB=True, estimate='auto', show=True, n_jobs=1, average=False,
             line_alpha=None, spatial_colors=True, verbose=None, sphere=None):
        from mne.viz.utils import _plot_psd, plt_show

        # set up default vars
        from packaging import version
        mne_version = version.parse(mne.__version__)
        has_new_mne = mne_version >= version.parse('0.22.0')
        has_20_mne = (mne_version >= version.parse('0.20.0')
                      and mne_version < version.parse('0.22.0'))
        if has_new_mne:
            from mne.defaults import _handle_default
            from mne.io.pick import _picks_to_idx
            from mne.viz._figure import _split_picks_by_type

            if ax is None:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            ax_list = [ax]

            units = _handle_default('units', None)
            picks = _picks_to_idx(self.info, picks)
            titles = _handle_default('titles', None)
            scalings = _handle_default('scalings', None)

            make_label = len(ax_list) == len(fig.axes)
            xlabels_list = [False] * (len(ax_list) - 1) + [True]
            (picks_list, units_list, scalings_list, titles_list
             ) = _split_picks_by_type(self, picks, units, scalings, titles)
        elif has_20_mne:
            from mne.viz.utils import _set_psd_plot_params
            fig, picks_list, titles_list, units_list, scalings_list, \
                ax_list, make_label, xlabels_list = _set_psd_plot_params(
                    self.info, proj, picks, ax, area_mode)
        else:
            from mne.viz.utils import _set_psd_plot_params
            fig, picks_list, titles_list, units_list, scalings_list, ax_list, \
                make_label = _set_psd_plot_params(self.info, proj, picks, ax,
                                                  area_mode)
        del ax

        crop_inst = not (fmin == 0 and fmax is None)
        fmax = self.freqs[-1] if fmax is None else fmax

        inst = self.copy()
        if crop_inst:
            inst.crop(fmin=fmin, fmax=fmax)
        inst.average()

        # create list of psd's (one element for each channel type)
        psd_list = list()
        for picks in picks_list:
            psd_list.append(inst.data[picks])

        args = [inst, fig, inst.freqs, psd_list, picks_list, titles_list,
                units_list, scalings_list, ax_list, make_label, color,
                area_mode, area_alpha, dB, estimate, average, spatial_colors,
                xscale, line_alpha]
        if has_20_mne or has_new_mne:
            args += [sphere, xlabels_list]

        fig = _plot_psd(*args)
        plt_show(show)
        return fig

    def to_evoked(self, dB=False):
        '''Turn the PSD object to Evoked to use standard mne functions like
        mne.viz.plot_compare_evokeds.'''
        freq_diff = np.diff(self.freqs)[0]
        sfreq = 1 / freq_diff
        info = self.info.copy()
        info['sfreq'] = sfreq
        data = (self.copy().average().data if self._has_epochs
                else self.data.copy())
        if dB:
            data = 10 * np.log(data)
        psd_evkd = mne.EvokedArray(data, info, tmin=self.freqs[0])
        if dB:
            psd_evkd.comment = 'units=dB'
        return psd_evkd

    # - [ ] plot_joint uses show=False by default and **args
    #       if show is in **args we will get an error
    def plot_joint(self, freqs=None, fmin=None, fmax=None, dB=False, **args):
        '''The same as plot_joint for Evokeds but for PSDS.

        Parameters
        ----------
        freqs : float | list of float
            Frequencies to plot as topomaps.
        fmin : float
            Frequency to start the line plot from.
        fmax : float
            Frequency to end the line plot with.
        dB : bool
            Whether to present data in decibels.

        Returns
        -------
        fig : matplotlib Figure
            Figure containing the visualisation.
        '''
        psd_evkd = self.to_evoked(dB=dB)
        if fmin is not None or fmax is not None:
            psd_evkd = psd_evkd.crop(tmin=fmin, tmax=fmax)

        if dB:
            vmin = np.percentile(psd_evkd.data, 1)
            vmax = np.percentile(psd_evkd.data, 99)

        freqs = 'peaks' if freqs is None else freqs
        fig = psd_evkd.plot_joint(times=freqs, show=False, **args)

        # set up labels
        axs = fig.axes
        ylabel = 'Power'
        ylabel += ' (dB)' if dB else ''
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel(ylabel)

        for ax in axs[1:-3]:
            ttl = ax.get_title()
            val = float(ttl.split(' ')[0])
            txtval = '{:.1f} Hz'.format(val)
            ax.set_title(txtval)

            if dB:
                # update color limits
                ax.images[0].set_clim(vmin=vmin * 1e6, vmax=vmax * 1e6)

        return fig

    # - [ ] LATER: add support for labeled grid (grid=True?)
    # - [ ] LATER: add support for passing axes
    def plot_topomap(self, freqs=None, **args):
        '''Plot topomap of given frequency range (or ranges).

        Properties
        ----------
        freqs : value | list of values
            Frequencies to plot as topographies.

        additional arguments are passed to the topomap plotting function.

        Returns
        -------
        tp : borsar.viz.Topo
            Instance of ``borsar.viz.Topo``.
        '''

        idxs = find_index(self.freqs, freqs)
        psd_array = (self.copy().average().data
                     if self._has_epochs else self.data)
        psd_array = psd_array[:, idxs]
        topos = Topo(psd_array, self.info, **args)

        # add frequency titles
        template = '{:.1f} Hz'
        for idx, tp in zip(idxs, topos):
            frq = self.freqs[idx]
            tp.axes.set_title(template.format(frq))

        return topos

    def average(self):
        '''Average epochs.'''
        if self._has_epochs:
            use_data = np.nanmean(self.data, axis=0)
            self._data = use_data
            self._has_epochs = False
        return self

    def crop(self, fmin=None, fmax=None):
        """Crop frequency range to ``fmin:fmax`` (inclusive).

        Parameters
        ----------
        fmin : value | None
            Lower edge of frequency range. The default is ``None`` which takes
            the lowest frequency.
        fmax : value | None
            Higher edge of frequency range. This frequency is included in the
            retained range. The default is ``None`` which takes the highest
            frequency.
        """
        fmin = self.freqs[0] if fmin is None else fmin
        fmax = self.freqs[-1] if fmax is None else fmax

        rng = find_range(self.freqs, [fmin, fmax])
        self.freqs = self.freqs[rng]
        self._data = self._data[..., rng]
        return self

    def copy(self):
        """Copy the instance of PSD."""
        psd = deepcopy(self)
        return psd

    @property
    def data(self):
        """The data matrix."""
        return self._data

    @property
    def ch_names(self):
        """List of channel names."""
        return self.info['ch_names']


PSD.plot.__doc__ = plot_epochs_psd.__doc__
