import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, detrend
import numpy as np
import pywt
import mne
from mne.preprocessing import ICA

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
            return f"CSV converted to FIF and saved at {fif_path}"
        
        except Exception as e:
            return f"Error occurred: {e}"
        
        
class Filter:
    def apply_MNE_filters(raw):
        raw.notch_filter(freqs=[50, 60])
        raw.filter(l_freq=None, h_freq=50)
        raw.plot(scalings={'eeg': 1e-4}, n_channels=1, duration=10, title="Filtered EEG Signal")            
        filtered_raw = raw
        return filtered_raw

class Plot:
    def plot_original(eeg_data):
        plt.plot(eeg_data[:10000])  # Plot the first 10,000 samples (~2 seconds)
        plt.xlabel("Sample Index")
        plt.ylabel("Voltage (V)")
        plt.title("Subset of Raw EEG Signal from .csv")
        plt.show()
    
    def plot_raw(raw):
        raw.plot(scalings={'eeg': 1e-4}, n_channels=1, duration=10, title="Raw EEG Signal")            
        plt.show()
    
    def plot_filtered_raw(filtered_raw):
        filtered_raw.plot(scalings={'eeg': 1e-4}, n_channels=1, duration=10, title="Filtered EEG Signal") 
        plt.show()
        
