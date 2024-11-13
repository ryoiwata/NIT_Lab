import pandas as pd
import numpy as np
import mne
from scipy.signal import detrend
from denoising_h import DataPreprocess, Plot, Filter
import matplotlib.pyplot as plt


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

        # Step 2: Plot the original subset of the EEG data
        print("Plotting the original EEG data...")
        #Plot.plot_original(eeg_data)

        # Step 3: Convert the EEG data to a Raw MNE object and save as .fif
        raw, message = DataPreprocess.convert_to_fif(eeg_data)
        print(message)

        # If the raw object was created successfully, proceed
        if raw:
            # Step 4: Plot the raw EEG signal
            print("Plotting the raw EEG data...")
            Plot.plot_raw(raw)

            # Step 5: Apply filters and plot the filtered EEG signal
            print("Applying filters and plotting the filtered EEG data...")
            filtered_raw = Filter.apply_MNE_filters(raw)
            Plot.plot_filtered_raw(filtered_raw)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()