import pandas as pd
import numpy as np
import mne
from scipy.signal import detrend
from denoising_h import DataPreprocess, Plot, Filter, Events
import matplotlib.pyplot as plt
import os


# Define the file path
file_path = "wu_venv/ssEEG/10_29_24 experiment/csv_files/SDS00005.csv"
# eeg_data = DataPreprocess.remove_missing(file_path)
# result = DataPreprocess.convert_to_fif(eeg_data)
# print(result)
def main():
    try:
        # Step 1: Load the EEG data from the CSV file
        eeg_data = DataPreprocess.remove_missing(file_path)
        print("Successfully loaded EEG data.")

        # Step 2: Convert the EEG data to a Raw MNE object and save as .fif
        raw, message = DataPreprocess.convert_to_fif(eeg_data)
        print(message)

        if raw:
            # Step 3: Apply filters and get the filtered raw data
            filtered_raw = Filter.apply_MNE_filters(raw)

            # Step 4: Plot individual segments based on stimulation events
            events_dict = Events.get_events_dict()
            for event_name, (start_time, end_time) in events_dict.items():
                print(f"Plotting event: {event_name} (from {start_time}s to {end_time}s)")
                Plot.plot_event_segment(filtered_raw, event_name, start_time, end_time)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()