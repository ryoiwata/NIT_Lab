import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, detrend
import numpy as np
import pywt
import mne
from mne.preprocessing import ICA
from denoising_h import Denoising_EEG
from denoising_h import Plots

# data-specific information: 
fs = 5000  # Example sampling frequency, adjust according to your data
notch_freq = 60
lowcut = 0.5
highcut = 40

# path to data that we will be transforming
file_path = "wu_venv/ssEEG/10_29_24 experiment/csv_files/WAVE4_Flick.CSV"
df = pd.read_csv(file_path, skiprows=11)
print(df.head())
print(df.tail())

# data organization: 
df.columns = ['index', 'CH1_Voltage(mV)']  # read in the 3 columns and name them for clarity. we will focus on Ch1 here.
time = df['index'].values  # assign "second" col to be time data 
signal = df['CH1_Voltage(mV)'].values  # assign "Ch1" col to be signal data 

signal_array = np.array(signal)
#detrend the signal to remove the offset from DC values
signal = detrend(signal)

# to be safe, we check for NaN or infinite values in the signal which can mess with later calculations
if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
    raise ValueError("signal has NaN or infinite vals")


# ================== USING MNE-PYTHON =====================
# =========================================================

time = df['index'].values
eeg_data = df['CH1_Voltage(mV)'].values

# convert from mV to V: 
eeg_data = eeg_data * 1e-3

ch_names = ['EEG']  # Single-channel name
ch_types = ['eeg']  # Channel type

# set the info as well: 
info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

# create the raw array object so that we can call mne functions on it: 
eeg_data = eeg_data.reshape(1, -1)  # Reshape to (n_channels, n_samples)
raw = mne.io.RawArray(eeg_data, info)

print("EEG Data Min:", np.min(eeg_data))
print("EEG Data Max:", np.max(eeg_data))
# plot the raw data : 
raw.plot(block=True)

# ========================================================
# add the notch filter to remove 60hz noise 
#signal_notch = Denoising_EEG.notch_filter(signal, notch_freq, fs)

# add bandpass filter
#signal_bandpassed = Denoising_EEG.bandpass_filter(signal_notch, lowcut, highcut, fs)

# add wavelet denoising
#signal_denoised = Denoising_EEG.wavelet_denoise(signal_bandpassed)

#Plots.plot_original(file_path)

#Plots.plot_4panel(time, signal, fs)