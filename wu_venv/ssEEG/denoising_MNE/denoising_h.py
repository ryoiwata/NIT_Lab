import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, detrend
import numpy as np
import pywt
import mne
from mne.preprocessing import ICA
import os
# Updating the code to fix ICA components and adjust filter length.

class DataPreprocess:
    
    # in order to use the .csv file with mne, we need to first 
    # convert it to a fif file format 
    def remove_missing(csv_path):
        df = pd.read_csv(csv_path, skiprows=11, engine='python')
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['Volt'])
        eeg_data = df['Volt'].values
        expected_samples = 1400000
        
        if len(eeg_data) != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples, but got {len(eeg_data)} samples.")
        
        return eeg_data
        
    def convert_to_fif(eeg_data):
        try:
            sfreq = 5000
            info = mne.create_info(ch_names=['EEG'], sfreq=sfreq, ch_types=['eeg'])
            #eeg_data = eeg_data / 1000.0
            eeg_data = eeg_data.reshape(1,-1)
            raw = mne.io.RawArray(eeg_data, info)

            fif_path = "eeg_data_raw.fif"
            raw.save(fif_path, overwrite=True)
            message =  f"CSV converted to FIF and saved at {fif_path}"
            return raw, message
        
        except Exception as e:
            return f"Error occurred: {e}"
        
        
class Filter:
    
    def inspect_signal(raw):
        data = raw.get_data()
        min_value = np.min(data)
        max_value = np.max(data)
        mean_value = np.mean(data)
        std_dev = np.std(data)
        print(f"Signal Statistics - Min: {min_value:.2f}, Max: {max_value:.2f}, Mean: {mean_value:.2f}, Std Dev: {std_dev:.2f}")

    @staticmethod
    def apply_downsampling(raw, new_sfreq=250):
        # Downsample the data to the specified frequency (e.g., 250 Hz)
        raw.resample(new_sfreq, npad="auto")
        print(f"Data downsampled to {new_sfreq} Hz.")
        return raw

    @staticmethod
    def apply_detrend(raw):
        # Get the data as a numpy array
        data = raw.get_data()
        # Apply detrending to the data (order=1 for linear detrending)
        data = mne.filter.detrend(data, order=1)
        # Update the raw object with the detrended data
        raw._data = data
        return raw
    
    def apply_bandpass_filter(raw):
        # Apply a band-pass filter to focus on the 0.5-40 Hz range
        raw.filter(l_freq=0.5, h_freq=40, picks='eeg')
        return raw
    
    
    @staticmethod
    def apply_notch_filter(raw):
        # Apply a notch filter to remove power line noise (50 Hz and 60 Hz)
        raw.notch_filter(freqs=[50, 60], picks='eeg')
        return raw


    
    @staticmethod
    def apply_all_filters(raw):
        # Apply all preprocessing steps
        raw = Filter.apply_detrend(raw)
        raw = Filter.apply_notch_filter(raw)
        raw = Filter.apply_bandpass_filter(raw)
        return raw

import matplotlib.pyplot as plt

class Plot:
    @staticmethod
    def plot_original(eeg_data):
        plt.plot(eeg_data[:10000])  # Plot the first 10,000 samples (~2 seconds)
        plt.xlabel("Sample Index")
        plt.ylabel("Voltage (V)")
        plt.title("Subset of Raw EEG Signal from .csv")
        plt.savefig("wu_venv/ssEEG/denoising_MNE/output/original_eeg_plot.png", dpi=300)
        plt.close()

    @staticmethod
    def plot_raw(raw):
        fig = raw.plot(scalings='auto', n_channels=1, duration=40, title="Raw EEG Signal", show=False)
        fig.savefig("wu_venv/ssEEG/denoising_MNE/output/raw_eeg_plot.png", dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_filtered_raw(filtered_raw):
        fig = filtered_raw.plot(scalings='auto', n_channels=1, duration=40, title="Filtered EEG Signal", show=False)
        fig.savefig("wu_venv/ssEEG/denoising_MNE/output/filtered_eeg_plot.png", dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_event_segment(raw, event_name, start_time, end_time):
        # Get the signal statistics
        data = raw.get_data()
        std_dev = np.std(data)
        
        # Set the scaling based on the standard deviation of the signal
        dynamic_scaling = std_dev * 2  # Use 2 times the standard deviation
        custom_scalings = {'eeg': dynamic_scaling}
        print(f"Using dynamic scaling: {dynamic_scaling:.2e}")

        # Plot the segment for the specified event
        duration = end_time - start_time
        fig = raw.plot(
            start=start_time,
            duration=duration,
            scalings=custom_scalings,
            n_channels=1,
            title=f"EEG Signal: {event_name}",
            show=False
        )
        
        # Save the figure
        save_path = f"wu_venv/ssEEG/denoising_MNE/output/{event_name.lower().replace(' ', '_')}_segment.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot for {event_name} at {save_path}")


class Events:
    @staticmethod
    def get_events_dict():
        # Define the time points for each event (in EEG time from CSV)
        return {
            "Sound On": (0, 40),          # From 0s to 40s in EEG time
            "Sound Off": (40, 113),       # From 40s to 113s
            "Touch Whiskers": (113, 173), # From 113s to 173s
            "Stop Touching": (173, 280),  # From 173s to 280s
        }
    