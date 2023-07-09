
import sys
import argparse
import pickle
import math
import numpy as np

import svGPFA.plot.plotUtilsPlotly
import socialMiceUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_name", help="subject name",
                        type=str, default="BLA00")
    parser.add_argument("--region", help="brain region",
                        type=str, default="BLA")
    parser.add_argument("--unit_id", type=int, help="unit_id to analyze",
                        default=106)
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="DoorOpen")
    parser.add_argument("--sorting_event_name", type=str,
                        help="behavioral event name to use to sort trials",
                        default=None)
    parser.add_argument("--colors_event_name", type=str,
                        help="events names used to color spikes",
                        default="outcome")
    parser.add_argument("--align_event_name",
                        help="name of event used for alignment",
                        type=str,
                        default="DoorOpen")
    parser.add_argument("--events_names",
                        help="names of marked events",
                        type=str,
                        default="[TrialOn,FoodNosePoke,DoorClosed,TrialOff]")
    parser.add_argument("--events_colors",
                        help="colors for marked events",
                        type=str, default="[black,magenta,green,black]")
    parser.add_argument("--events_markers",
                        help="markers for marked events",
                        type=str, default="[circle,circle,circle,circle]")
    parser.add_argument("--xmin", type=float, help="mininum x-axis value",
                        default=-math.inf)
    parser.add_argument("--xmax", type=float, help="maximum x-axis value",
                        default=math.inf)
    parser.add_argument("--epoched_spikes_times_filename_pattern",
                        help="epoched spikes times filename pattern",
                        type=str,
                        default=("../../results/epochedSpikes_subject_{:s}_"
                                 "region_{:s}_epochedBy_{:s}.{:s}"))
    parser.add_argument("--fig_filename_pattern",
                        help="spikes times figure filename pattern",
                        type=str,
                        default=("../../figures/spikes_times_subject_{:s}_"
                                 "region_{:s}_epochedBy_{:s}_unit_{:03d}.{:s}"))
    args = parser.parse_args()

    subject_name = args.subject_name
    region = args.region
    unit_id = args.unit_id
    epoch_event_name = args.epoch_event_name
    sorting_event_name = args.sorting_event_name
    colors_event_name = args.colors_event_name
    align_event_name = args.align_event_name
    events_names = [str for str in args.events_names[1:-1].split(",")]
    events_colors = [str for str in args.events_colors[1:-1].split(",")]
    events_markers = [str for str in args.events_markers[1:-1].split(",")]
    xmin = args.xmin
    xmax = args.xmax
    epoched_spikes_times_filename_pattern = \
        args.epoched_spikes_times_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    epoched_spikes_times_filename = \
        epoched_spikes_times_filename_pattern.format(
            subject_name, region, epoch_event_name, "pickle")

    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]
    units_ids = load_res["units_ids"]
    trials_ids = load_res["trials_ids"]
    trials_start_times = load_res["trials_start_times"]
    trials_end_times = load_res["trials_end_times"]
    trials_info = load_res["trials_info"]

    epoch_times = trials_info[epoch_event_name]
    n_trials = len(epoch_times)

    # begin remove trials
    if sorting_event_name is None:
        remove_trial = np.isnan(epoch_times)
    else:
        sorting_times = trials_info[sorting_event_name]
        remove_trial = np.logical_or(np.isnan(epoch_times),
                                     np.isnan(sorting_times))
    keep_trial = np.logical_not(remove_trial)

    spikes_times = [spikes_times[r] for r in range(n_trials) if keep_trial[r]]
#     for key in trials_info.keys():
#         trials_info[key] = [trials_info[key][r] for r in range(n_trials)
#                             if keep_trial[r]]
    epoch_times = epoch_times[keep_trial]
    n_trials = len(epoch_times)
    trials_ids = trials_ids[keep_trial]
    if sorting_event_name is not None:
        sorting_times = sorting_times[keep_trial]
        sorting_times -= epoch_times
    else:
        sorting_times = None
    # end remove trials

    colors_event = trials_info[colors_event_name]

    neuron_index = np.nonzero(np.array(units_ids) == unit_id)[0].item()

    colors_event = trials_info[colors_event_name]
    trials_colors = [None] * n_trials
    for i, an_event in enumerate(colors_event):
        if an_event == 2:  # hit
            trials_colors[i] = "red"
        elif an_event == 5:  # correct rejection
            trials_colors[i] = "darkred"
        elif an_event == 3:  # Miss
            trials_colors[i] = "lightblue"
        elif an_event == 4:  # False alarm
            trials_colors[i] = "blue"
        else:
            raise ValueError(f"Invalid {colors_event_name}={an_event}")

    events_times = []
    for event_name in events_names:
        events_times.append([trials_info[event_name][trial_id]
                             for trial_id in trials_ids])

    marked_events_times, marked_events_colors, marked_events_markers = \
        socialMiceUtils.buildMarkedEventsInfo(
            events_times=events_times,
            events_colors=events_colors,
            events_markers=events_markers,
        )

    align_event_times = [trials_info[align_event_name][trial_id]
                         for trial_id in trials_ids]

    sorting_label = sorting_event_name if sorting_event_name is not None \
        else "None"
    title = (f"Neuron: {unit_id}, Region: {region}, "
             f"Epoched by: {epoch_event_name}, Sorted by: {sorting_label}, "
             f"Spike colors by: {colors_event_name}")
    fig = svGPFA.plot.plotUtilsPlotly.getSpikesTimesPlotOneNeuron(
        spikes_times=spikes_times,
        sorting_times=sorting_times,
        neuron_index=neuron_index,
        title=title,
        trials_ids=trials_ids,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        align_event_times=align_event_times,
        trials_colors=trials_colors,
    )
    if xmin is None:
        xmin = np.min(trials_start_times)
    if xmax is None:
        xmax = np.max(trials_end_times)
    fig.update_xaxes(range=[xmin, xmax])

    fig.write_image(fig_filename_pattern.format(subject_name, region,
                                                epoch_event_name, unit_id, "png"))
    fig.write_html(fig_filename_pattern.format(subject_name, region,
                                               epoch_event_name, unit_id, "html"))
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
