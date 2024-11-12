import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, detrend
import numpy as np
import pywt
import mne
from mne.preprocessing import ICA

class Data_Preprocess:
    def import_AndCheck(file_path):
        df = pd.read_csv(file_path, skiprows=11)

        # Set the column names according to the .csv 
        df.columns = ['index', 'CH1_Voltage(mV)']

        # Step 2: Handle Missing Data
        if df['CH1_Voltage(mV)'].isna().any():
            print("Warning: Missing values detected in the data. Replacing with zeros.")
            df['CH1_Voltage(mV)'].fillna(0, inplace=True)
    
    def convert_csv_to_fif(df):
                # Step 3: Scale the Voltage and Convert to Volts
        eeg_data = df['CH1_Voltage(mV)'].values * 4.0 * 1e-3  # Scale and convert from mV to V

        # Step 4: Reshape the Data for MNE (n_channels, n_samples)
        eeg_data = eeg_data.reshape(1, -1)

        # Step 5: Create MNE Info Object
        fs = 100  # Corrected sampling frequency based on time interval
        ch_names = ['EEG']
        ch_types = ['eeg']
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

        # Step 6: Create RawArray Object
        raw = mne.io.RawArray(eeg_data, info)

        # Step 7: Save as FIF File
        fif_file_path = "ssEEG_data.fif"
        raw.save(fif_file_path, overwrite=True)

        print(f"Data successfully saved to {fif_file_path}")

        # Load the FIF file
        raw = mne.io.read_raw_fif(fif_file_path, preload=True)
        
    def filter_raw(raw):
        # Apply a bandpass filter (0.5-40 Hz) to focus on the typical EEG frequency range
        raw.filter(l_freq=0.5, h_freq=40, filter_length='auto')            
            
        
class Denoising_EEG:

    @staticmethod
    def notch_filter(signal, freq, fs, quality_factor=30):
        w0 = freq / (fs / 2)  # Normalize the frequency
        b, a = iirnotch(w0, quality_factor, fs)
        y = filtfilt(b, a, signal)
        return y

    @staticmethod
    def bandpass_filter(signal, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, signal)
        return y

    @staticmethod
    def wavelet_denoise(signal, wavelet='db4', level=1):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
        y = pywt.waverec(coeffs, wavelet)
        return y[:len(signal)]

    @staticmethod
    def apply_ica(df, n_components=15):
        ica = ICA(n_components=n_components, random_state=97)
        ica.fit(df)
        ica.exclude = [0]  # indices of components to exclude
        raw_corrected = df.copy()
        ica.apply(raw_corrected)
        return raw_corrected


class Plots:

    def plot_original(file_path):
        df = pd.read_csv(file_path, skiprows=11)

        # Rename columns for clarity
        df.columns = ['index', 'CH1_Voltage(mV)']

        # Extract time and signal data
        time = df['index'].values
        signal = df['CH1_Voltage(mV)'].values

        # Detrend the raw signal to remove any linear trend
        signal = detrend(signal)

        # Plot the raw signal
        plt.figure(figsize=(10, 5))
        plt.plot(time, signal, color='blue', label='Raw EEG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        plt.title('Raw EEG Signal Before Processing')
        plt.legend()
        plt.grid()
        plt.show()
    
    @staticmethod
    def plot_4panel(time, signal, fs, notch_freq=60, lowcut=0.5, highcut=40):
        # Step 1: Raw Signal
        raw_signal = signal

        # Step 2: Notch Filter
        signal_notch = Denoising_EEG.notch_filter(raw_signal, notch_freq, fs)

        # Step 3: Notch + Bandpass Filter
        signal_bandpassed = Denoising_EEG.bandpass_filter(signal_notch, lowcut, highcut, fs)

        # Step 4: Notch + Bandpass + Wavelet Denoising
        signal_wavelet = Denoising_EEG.wavelet_denoise(signal_bandpassed)

        signal_detredned = detrend(signal)
        # Create a figure with four subplots (2 rows, 2 columns)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Raw Signal
        axs[0, 0].plot(time, raw_signal, color='blue')
        axs[0, 0].set_title('Raw EEG Signal')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Voltage (mV)')
        axs[0, 0].grid()

        # Plot 2: Notch Filtered Signal
        axs[0, 1].plot(time, signal_notch, color='orange')
        axs[0, 1].set_title('After Notch Filter (60 Hz)')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Voltage (mV)')
        axs[0, 1].grid()

        # Plot 3: Notch + Bandpass Filtered Signal
        axs[1, 0].plot(time, signal_bandpassed, color='green')
        axs[1, 0].set_title('After Notch + Bandpass Filter')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Voltage (mV)')
        axs[1, 0].grid()

        # Plot 4: Notch + Bandpass + Wavelet Denoised Signal
        axs[1, 1].plot(time, signal_wavelet, color='purple')
        axs[1, 1].set_title('After Notch + Bandpass + Wavelet Denoising')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Voltage (mV)')
        axs[1, 1].grid()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
        fs = 1000  # Sampling frequency
        time = np.linspace(0, 10, fs * 10)  # Time vector
        signal = np.sin(2 * np.pi * 10 * time) + 0.5 * np.random.randn(len(time))  # Synthetic raw EEG signal with noise

        # Apply detrending to simulate preprocessing
        signal_notchfiltered = notch_filter(signal)

        # Create a figure with four subplots (2 rows, 2 columns)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Raw Signal
        axs[0, 0].plot(time, signal, color='blue')
        axs[0, 0].set_title('Raw EEG Signal')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Voltage (mV)')
        axs[0, 0].grid()

        # Plot 2: 60 hz removed signal 
        axs[0, 1].plot(time, signal_detrended, color='orange')
        axs[0, 1].set_title('Detrended EEG Signal')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Voltage (mV)')
        axs[0, 1].grid()

        # Plot 3: Histogram of Raw Signal
        axs[1, 0].hist(signal, bins=50, color='green')
        axs[1, 0].set_title('Histogram of Raw Signal')
        axs[1, 0].set_xlabel('Voltage (mV)')
        axs[1, 0].set_ylabel('Frequency')

        # Plot 4: Power Spectral Density (PSD) of Detrended Signal
        axs[1, 1].psd(signal_detrended, Fs=fs, color='purple')
        axs[1, 1].set_title('Power Spectral Density of Detrended Signal')

        # Adjust layout
        plt.tight_layout()
        plt.show()
        
    def plot_PSD(raw):
        raw.plot_psd(fmax=50)
    
    
    def plot_clean_eeg(raw):
        # Plot the cleaned data
        raw.plot(block=True)
        time = df['index'].values
        signal = df['CH1_Voltage(mV)']
        Plots.plot_4panel(time, signal, fs=100, notch_freq=60, lowcut=0.5, highcut=40)