import sys
import os
import argparse
import configparser
import pickle
import numpy as np
import pandas as pd

import socialMiceUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_name", help="subject name",
                        type=str, default="BLA00")
    parser.add_argument("--region", help="brain region",
                        type=str, default="BLA")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="DoorOpen")
    parser.add_argument("--epoch_start_event_name",
                        help="trial start event name", type=str,
                        default="TrialOn")
    parser.add_argument("--epoch_end_event_name", help="trial end event name",
                        type=str, default="TrialOff")
    parser.add_argument("--data_dirname", help="data dirname", type=str,
                        default="../../data")
    parser.add_argument("--trials_info_filename",
                        help="trails info filename", type=str,
                        default="behavior_data.csv")
    parser.add_argument("--results_filename_pattern",
                        help="results filename pattern",
                        type=str,
                        default=("../../results/epochedSpikes_subject_{:s}_"
                                 "region_{:s}_epochedBy_{:s}.{:s}"))
    args = parser.parse_args()

    subject_name = args.subject_name
    region = args.region
    epoch_event_name = args.epoch_event_name
    epoch_start_event_name = args.epoch_start_event_name
    epoch_end_event_name = args.epoch_end_event_name
    data_dirname = args.data_dirname
    trials_info_filename = args.trials_info_filename
    results_filename_pattern = args.results_filename_pattern

    trials_info = pd.read_csv((f"{data_dirname}/{subject_name}/"
                               f"{trials_info_filename}"))
    n_trials = trials_info.shape[0]
    spikes_dirname = f"{data_dirname}/{subject_name}/{region}"
    spikes_filenames = [f for f in os.listdir(spikes_dirname)
                        if os.path.isfile(os.path.join(spikes_dirname, f))]
    n_units = len(spikes_filenames)
    epoch_times = trials_info[epoch_event_name]
    epoch_start_times = trials_info[epoch_start_event_name]
    epoch_end_times = trials_info[epoch_end_event_name]
    n_trials = len(epoch_times)
    trials_ids = np.arange(n_trials)

    spikes_times_by_neuron = []
    units_ids = []
    for spikes_filename in spikes_filenames:
        unit_id = int(spikes_filename[(spikes_filename.find("_")+1):
                                      spikes_filename.find(".")])
        units_ids.append(unit_id)
        print(f"Processing unit {unit_id}")
        neuron_spikes_times = pd.read_csv(
            f"{spikes_dirname}/{spikes_filename}").to_numpy().squeeze()
        n_epoched_spikes_times = socialMiceUtils.epoch_neuron_spikes_times(
            neuron_spikes_times=neuron_spikes_times,
            epoch_times=epoch_times,
            epoch_start_times=epoch_start_times,
            epoch_end_times=epoch_end_times)
        spikes_times_by_neuron.append(n_epoched_spikes_times)
    spikes_times = [[spikes_times_by_neuron[n][r] for n in range(n_units)]
                    for r in range(n_trials)]

    trials_start_times = [epoch_start_times[r]-epoch_times[r]
                          for r in range(n_trials)]
    trials_end_times = [epoch_end_times[r]-epoch_times[r]
                        for r in range(n_trials)]

    epoch_config = configparser.ConfigParser()
    epoch_config["params"] = {
        "subject_name": subject_name,
        "region": region,
        "epoch_event_name": epoch_event_name,
        "epoch_start_event_name": epoch_start_event_name,
        "epoch_end_event_name": epoch_end_event_name,
        "data_dirname": data_dirname,
        "trials_info_filename": trials_info_filename,
        "results_filename_pattern": results_filename_pattern,
        "units_ids": units_ids,
        "trials_ids": trials_ids,
    }
    metadata_filename = results_filename_pattern.format(subject_name, region,
                                                        epoch_event_name,
                                                        "metadata")
    with open(metadata_filename, "w") as f:
        epoch_config.write(f)
    print(f"Saved {metadata_filename}")

    results = {"spikes_times": spikes_times,
               "trials_start_times": trials_start_times,
               "trials_end_times": trials_end_times,
               "units_ids": units_ids,
               "trials_ids": trials_ids,
               "trials_info": trials_info}
    results_filename = results_filename_pattern.format(subject_name, region,
                                                       epoch_event_name,
                                                       "pickle")
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
