import numpy as np
from .utils import valid_windows


def compute_rest_psd(raw, events=None, event_id=None, tmin=None, tmax=None,
                     winlen=2., step=0.5):
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

    raw: mne.Raw
        Raw file to use.
    events: numpy array N x 3 or None
        Mne events array. If None (default) `tmin` and `tmax` are not
        calculated with respect to events but the whole time range of the
        `raw`.
    event_id: list or numpy array
        Event types to use in defining segments for which psd is computed.
        If None (default) and events were passed all event types are used.
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
    '''
    from mne.time_frequency import psd_welch

    sfreq = raw.info['sfreq']
    n_fft = int(round(winlen * sfreq))
    n_overlap = n_fft - int(round(step * sfreq))

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
            this_psd, freq = psd_welch(raw, n_fft=n_fft, n_overlap=n_overlap,
                                       tmin=this_tmin, tmax=this_tmax)
            psd_dict[event_type].append(this_psd)

            # compute percent of windows that do not overlap with artifacts
            # these constitute weights used in averaging
            weight = (valid_windows(raw, tmin=this_tmin, tmax=this_tmax,
                                    winlen=winlen, step=step)).mean()
            psd_weights[event_type].append(weight)

        # use np.average() with weights to compute wieghted average
        psd = {k: np.average(np.stack(psd_dict[k], axis=0),
                             weights=psd_weights[k], axis=0)
                             for k in psd_dict.keys()}
        if len(event_id) == 1 and got_event_id:
            psd = psd[event_id[0]]

        return psd, freq
    else:
        # use only tmin, tmax, a lot easier
        return psd_welch(raw, n_fft=n_fft, n_overlap=n_overlap,
                         tmin=tmin, tmax=tmax)
