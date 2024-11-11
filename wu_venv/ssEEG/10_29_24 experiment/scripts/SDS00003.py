import pywt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import iirnotch, butter, filtfilt, detrend

df = pd.read_csv("wu_venv/ssEEG/10_29_24 experiment/csv_files/SDS00003.csv")
print(df.head())

def remove_60hz(df, fs, column='Volt'):
    time = df['Second'].values
    signal = df[column].values

    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("Original signal contains NaN or infinite values. Please check your data.")
    
    signal = signal.detrend(signal)

    # we need to downsample bc data is gigantic
    downsample_factor = 20
    time_downsampled = time[::downsample_factor]
    signal_downsampled = signal[::downsample_factor]
    fs_downsampled = fs / downsample_factor  # Adjusted sampling rate

    #  definition of band pass filteration 
    low_cutoff = 20
    high_cutoff = 1200
    nyquist = fs_downsampled / 2
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b_bandpass, a_bandpass = butter(2, [low, high], btype='bandpass')

    # here's where we apply the band pass filter 
    bandpass_filtered_signal = filtfilt(b_bandpass, a_bandpass, signal_downsampled)

    # here's where we define and use the notch filter 
    notch_freq = 60
    quality_factor = 20.0
    b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs_downsampled)

    # Apply notch filter
    filtered_signal = filtfilt(b_notch, a_notch, bandpass_filtered_signal)

    # Check for NaN or infinite values in the filtered signal
    if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
        raise ValueError("Filtered signal contains NaN or infinite values. Filter configuration may need adjustment.")

    # Plot the filtered signal
    plt.figure(figsize=(10, 5))
    plt.plot(time_downsampled, filtered_signal, color='black')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Filtered Voltage (Volt)')
    plt.title('Filtered EEG Signal with 60 Hz Noise Removed')
    plt.show()

    # add filtered signal to DataFrame
    df[column + '_filtered'] = np.interp(time, time_downsampled, filtered_signal)

    return df


