
import sys
import argparse
import pickle

import gcnu_common.utils
import svGPFA.plot.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_name", help="subject name",
                        type=str, default="BLA00")
    parser.add_argument("--region", help="brain region",
                        type=str, default="BLA")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="DoorOpen")
    parser.add_argument("--epoched_spikes_times_filename_pattern",
                        help="epoched spikes times filename pattern",
                        type=str,
                        default=("../../results/epochedSpikes_subject_{:s}_"
                                 "region_{:s}_epochedBy_{:s}.{:s}"))
    parser.add_argument("--spikes_rates_fig_filename_pattern",
                        help=("spikes rates figure filename pattern"),
                        type=str,
                        default=("../../figures/spikes_rates_subject_{:s}_"
                                 "region_{:s}_epochedBy_{:s}.{:s}"))
    args = parser.parse_args()

    subject_name = args.subject_name
    region = args.region
    epoch_event_name = args.epoch_event_name
    epoched_spikes_times_filename_pattern = \
        args.epoched_spikes_times_filename_pattern
    spikes_rates_fig_filename_pattern = \
        args.spikes_rates_fig_filename_pattern

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
    n_trials = len(spikes_times)

    trials_durations = [trials_end_times[r] - trials_start_times[r]
                        for r in range(n_trials)]
    spikes_rates_allTrials_allNeurons = \
        gcnu_common.utils.neural_data_analysis.\
        getSpikesRatesAllTrialsAllNeurons(
            spikes_times=spikes_times, trials_durations=trials_durations)

    fig = svGPFA.plot.plotUtilsPlotly.\
        getPlotSpikesRatesAllTrialsAllNeurons(
            spikes_rates=spikes_rates_allTrials_allNeurons,
            trials_ids=trials_ids, clusters_ids=units_ids)

    fig.write_image(spikes_rates_fig_filename_pattern.format(subject_name,
                                                             region,
                                                             epoch_event_name,
                                                             "png"))
    fig.write_html(spikes_rates_fig_filename_pattern.format(subject_name,
                                                            region,
                                                            epoch_event_name,
                                                            "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
