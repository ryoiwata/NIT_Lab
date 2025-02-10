import mne
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_multitaper
import pandas as pd
import numpy as np
from obspy import Stream, Trace
from obspy.imaging.spectrogram import spectrogram
from obspy.signal.tf_misfit import plot_tfr
from scipy.interpolate import interp1d
import os
from scipy.signal import welch


class CSV: 
        
    # [FUNCTION] Write .set to .csv
    def set_to_csv(set_filename, output_csv):
        raw = mne.io.read_raw_eeglab(set_filename, preload=True)

        data = raw.get_data()  # Shape: (channels, samples)

        data = data.T

        channel_names = raw.ch_names

        df = pd.DataFrame(data, columns=channel_names)
        print(df.head())
        df.to_csv(output_csv, index=False)

        print(f"EEG data saved as CSV: {output_csv}")


    def plot_all_channels(data, srate, title="All Channels", labels=None):
        num_samples = data.shape[0]  
        times = np.arange(0, num_samples / srate, 1 / srate)

        plt.figure(figsize=(12, 6))
        colors = ['blue', 'red', 'pink', 'green', 'black', 'orange', 'purple', 'cyan', 'magenta', 'brown']

        if labels is None:
                labels = [f"Channel {i+1}" for i in range(data.shape[1])]

        data_mV = data * 1e3

        for i in range(data_mV.shape[1]):  
            plt.plot(times, data_mV[:, i], label=labels[i], color=colors[i % len(colors)], alpha=0.7)

        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)") 
        # plt.yticks(-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3)# Label y-axis in millivolts
        plt.ticklabel_format(style='plain', axis='y')  # Remove scientific notation on y-axis
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_separate_channels(data, srate, title="EEG Recording", labels=None):
        num_samples, num_channels = data.shape
        times = np.arange(0, num_samples / srate, 1 / srate)

        data_microvolts = data * 1e6  # Convert to µV

        y_min = np.min(data_microvolts)
        y_max = np.max(data_microvolts)

        if labels is None or len(labels) != num_channels:
            labels = [f"Channel {i+1}" for i in range(num_channels)]

        fig, axes = plt.subplots(num_channels, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

        colors = ['blue', 'red', 'pink', 'green', 'black', 'orange', 'purple']

        for i in range(num_channels):
            axes[i].plot(times, data_microvolts[:, i], color=colors[i % len(colors)], alpha=0.7)
            axes[i].set_ylabel(f"{labels[i]} (µV)", fontsize=10)  
            axes[i].set_ylim(y_min, y_max)  
            axes[i].grid(True)

        plt.suptitle(title, fontsize=14)
        axes[-1].set_xlabel("Time (s)")

        plt.show()


        
    # [FUNCTION] Plot the PSD of a specified dataset
    def plot_psd(raw, label, color):

        psd = raw.compute_psd(method="welch", fmax=50)  
        freqs, power = psd.freqs, psd.get_data()  
        
        plt.plot(freqs, power[0], label=label, color=color, alpha=0.7)  
        
        return freqs, power
    
    def plot_channel_spectrogram(csv_path, channel_index, sample_rate, title):
        df = pd.read_csv(csv_path)

        if channel_index < 0 or channel_index >= df.shape[1]:
            raise ValueError(f"Invalid channel index: {channel_index}. Must be between 0 and {df.shape[1]-1}")

        channel_name = df.columns[channel_index] 
        eeg_data = df.iloc[:, channel_index].values.astype(np.float32)  

        num_samples = len(eeg_data)  
        recording_duration = num_samples / sample_rate  

        trace = Trace(data=eeg_data)
        trace.stats.sampling_rate = sample_rate  
        trace.stats.starttime = 0 

        fig = plot_spectrogram(
            trace,
            samp_rate=sample_rate, 
            log=True,    
            per_lap=0.9, 
            wlen=2.0,    
            dbscale=True,
            cmap="viridis"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Spectrogram of {channel_name} ({title})")

        plt.show()
                
                
    def plot_channel_TFR_2(csv_path,column_index,sample_rate,title):
        df = pd.read_csv(csv_path)

        if column_index < 0 or column_index >= df.shape[1]:
            raise ValueError(f"Invalid column index: {column_index}. Must be between 0 and {df.shape[1]-1}")

        channel_name = df.columns[column_index]
        voltage = df.iloc[:, column_index].values.astype(np.float32)

        num_samples = len(voltage) 
        dt = 1.0 / sample_rate 
        recording_duration = num_samples * dt 
        times = np.linspace(0, recording_duration, num_samples) 

        trace = Trace(data=voltage)
        trace.stats.sampling_rate = sample_rate
        trace.stats.starttime = 0

        fig = plot_tfr(
            trace.data, 
            dt=dt, 
            t0=0, 
            fmin=0.5, 
            fmax=40,
            nf=300, 
            w0=8, 
            cmap="viridis", 
            mode="absolute", 
            show=False, 
            clim=0.00001
        )
        

        ax = fig.axes[2] 
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 3, 5, 10, 15, 20, 25, 30])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xscale("linear")
        ax.set_xticks([0,0.0001,.0002,.0003,.0004,.0005])
        
        ax1 = fig.axes[0] 
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (Volts)")
        ax1.set_yscale("linear")
        ax1.set_ylim(-0.000010,0.000010)
        ax1.set_yticks([-.00010,-0.00005,0,.00005,0.00010])
        ax1.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax1.set_xscale("linear")
        ax1.set_xticks(np.arange(0, recording_duration, step=5))
        
        plt.title = title
        plt.show()
            
    def scale_csv_to_microvolts(csv_path, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        df = pd.read_csv(csv_path)

        df.iloc[:, :] = df.iloc[:, :] * 1e6  

        filename = os.path.basename(csv_path).replace(".csv", "_uV.csv")
        output_csv_path = os.path.join(output_folder, filename)

        df.to_csv(output_csv_path, index=False)

        # print(f"CSV file scaled to microvolts and saved as: {output_csv_path}")


    def plot_scaled_channel_TFR(csv_path,column_index,sample_rate,):
        df = pd.read_csv(csv_path)

        if column_index < 0 or column_index >= df.shape[1]:
            raise ValueError(f"Invalid column index: {column_index}. Must be between 0 and {df.shape[1]-1}")

        channel_name = df.columns[column_index]  
        voltage = df.iloc[:, column_index].values.astype(np.float32) 

        num_samples = len(voltage)  
        dt = 1.0 / sample_rate 
        recording_duration = num_samples * dt  
        times = np.linspace(0, recording_duration, num_samples) 

        trace = Trace(data=voltage)
        trace.stats.sampling_rate = sample_rate
        trace.stats.starttime = 0 

        fig = plot_tfr(
            trace.data, 
            dt=dt, 
            t0=0, 
            fmin=0.5, 
            fmax=40,
            nf=300, 
            w0=8, 
            cmap="viridis", 
            mode="absolute", 
            show=False, 
            clim=1
        )

        ax = fig.axes[2] 
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 3, 5, 10, 15, 20, 25, 30])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xscale("linear")
        ax.set_xticks([0,1,2,3,4,5])
        
        ax1 = fig.axes[0]  
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (Volts)")
        ax1.set_yscale("linear")
        ax1.set_ylim(-100,100)
        ax1.set_yticks([-10,-8,-6,-4,-2,0,2,4,6,8,10])
        ax1.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax1.set_xscale("linear")
        ax1.set_xticks(np.arange(0, recording_duration, step=5))
        plt.show()
        
        
    def plot_psd(csv_path, sample_rate, channel_labels=None):
        df = pd.read_csv(csv_path)

        if channel_labels is None:
            channel_labels = df.columns.tolist()  
        elif len(channel_labels) != df.shape[1]:
            raise ValueError(f"Provided channel labels do not match the number of columns in {csv_path}")

        colors = ['blue', 'red', 'pink', 'green', 'black', 'orange', 'purple', 'cyan', 'magenta', 'brown']

        plt.figure(figsize=(10, 6))

        for i, (channel, label) in enumerate(zip(df.columns, channel_labels)):
            data = df[channel].values 
            freqs, psd = welch(data, fs=sample_rate, nperseg=1024)  

            plt.plot(freqs, psd, label=label, color=colors[i % len(colors)], alpha=0.8)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (µV²/Hz)")
        plt.title(f"Power Spectral Density (PSD) - {csv_path.split('/')[-1]}")
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 50]) 
        plt.yscale("log") 

        plt.show()

    def average_channels(csv_path, output_csv_path):

        df = pd.read_csv(csv_path)

        df['Average_Channel'] = df.mean(axis=1)  

        df[['Average_Channel']].to_csv(output_csv_path, index=False)

        print(f"Averaged EEG channel saved as: {output_csv_path}")
