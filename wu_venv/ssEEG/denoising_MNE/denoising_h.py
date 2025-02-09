import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, detrend
import numpy as np
import pywt
import mne
from mne.preprocessing import ICA
import os
from mne.time_frequency import tfr_multitaper
import sklearn
from sklearn.decomposition import FastICA

class DataPreprocess:
    # in order to use the .csv file with mne, we need to first 
    # convert it to a fif file format 
    def remove_missing(csv_path): # first, we scan through and drop rows that are empty
        df = pd.read_csv(csv_path, skiprows=11, engine='python')
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['Volt']) # specifically from the volt channel
        eeg_data = df['Volt'].values # then reset the df to be the cleaned data 
        expected_samples = 1400000 
        
        if len(eeg_data) != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples, but got {len(eeg_data)} samples.")
        
        return eeg_data
        
    def convert_to_fif(eeg_data):
        try:
            sfreq = 5000
            info = mne.create_info(ch_names=['EEG'], sfreq=sfreq, ch_types=['eeg'])
            #eeg_data = eeg_data / 1000.0
            eeg_data = eeg_data.reshape(1,-1) # then we reshape the data to be 2d
            raw = mne.io.RawArray(eeg_data, info) # then we define raw 

            fif_path = "eeg_data_raw.fif" # set the path of the .fif file 
            raw.save(fif_path, overwrite=True)
            message =  f"CSV converted to FIF and saved at {fif_path}"
            return raw, message # .fif returns as RAW NUMPY ARRAY 
        
        except Exception as e:  # in case it doesn;t work 
            return f"Error occurred: {e}"
        
    def convert_to_csv(fif_path):
        
        try:
            raw = mne.io.read_raw_fif(fif_path, preload=True)

            data = raw.get_data()[0]
            timestamps = raw.times
            
            df = pd.DataFrame({
                'Time (s)': timestamps,
                'Voltage (V)': data
            })
            
            csv_output_path = "wu_venv/ssEEG/denoising_MNE/converted_filtered_csv/filtered_csv.csv"
            
            df.to_csv(csv_output_path, index=False)
            
            print(f"Data successfully converted from {fif_path} to {csv_output_path}")
        
        except Exception as e:
            print(f"Error during conversion: {e}")
        

    def return_filtered_fif_path(raw):
        try:
        # Define the output .fif file path
            output_fif_path = "wu_venv/ssEEG/denoising_MNE/filtered_data.fif"

            # Apply the filters
            print("Applying filters to the raw data...")
            raw = Filter.apply_detrend(raw)
            raw = Filter.apply_notch_filter(raw)
            raw = Filter.apply_bandpass_filter(raw)
            raw = Filter.apply_fastICA(raw)

            # Save the filtered raw data to a .fif file
            raw.save(output_fif_path, overwrite=True)
            print(f"Filtered data successfully saved to {output_fif_path}")

            # Return the path of the saved .fif file
            return output_fif_path

        except Exception as e:
            print(f"Error during filtering and saving: {e}")
            return None
        
        
        
class Filter:
    
    # this function just returns basic information about the data 
    def inspect_signal(raw): 
        data = raw.get_data()
        # information about the bounds of the data 
        min_value = np.min(data) 
        max_value = np.max(data)
        mean_value = np.mean(data) # return the mean of the dataset 
        std_dev = np.std(data) #return the std.dev of the dataset 
        print(f"Signal Statistics - Min: {min_value:.2f}, Max: {max_value:.2f}, Mean: {mean_value:.2f}, Std Dev: {std_dev:.2f}")

    @staticmethod
    def apply_downsampling(raw, new_sfreq=5000): # default downsample rate is 5000 hZ
        downsampled_raw = raw.resample(new_sfreq, npad="auto") # automatically pad data 
        print(f"Data downsampled to {new_sfreq} Hz.")
        raw = downsampled_raw
        return raw # finally return the same thing again

    @staticmethod
    def apply_detrend(raw): # next detrend the data using the iir filter 
        data = raw.get_data()
        data = mne.filter.detrend(data, order=1)
        raw._data = data
        return raw
    
    def apply_bandpass_filter(raw):
        bandpassed_raw = raw.filter(l_freq=0.5, h_freq=20, picks='eeg')
        raw = bandpassed_raw
        return raw
    
    @staticmethod
    def apply_notch_filter(raw):
        # edit to also filter out noise from the harmonics of the power line noise
        raw.notch_filter(freqs=[50, 100], picks= 'eeg') 
        raw.notch_filter(freqs=[60, 120], picks='eeg')
        return raw
    
    def average_signal(raw):
        data = raw.get_data()

        avg_data = np.mean(data, axis=0)

        print(f"Averaged signal computed. Length: {len(avg_data)} samples.")
        return avg_data
    
    def apply_fastICA(raw):
        n_channels = raw.info['nchan']
        n_times = raw.n_times
        print(f"Number of Channels: {n_channels}, Number of Time Points: {n_times}")
        data = raw.get_data().T  # shape is now (1400000, 1)
        ica = FastICA(n_components=1, random_state=42)
        transformed_data = ica.fit_transform(data)
        print("Transformed Data Shape:", transformed_data.shape)
        return raw
    
    
    @staticmethod
    def apply_all_filters(raw):
        raw = Filter.apply_detrend(raw)
        raw = Filter.apply_notch_filter(raw)
        raw = Filter.apply_bandpass_filter(raw)
        raw = Filter.apply_fastICA(raw)
        return raw
        
        
        
    def remove_artifacts(stimulus_recording, baseline_recording, fs, freq_band=(0.5, 100), threshold=2.5):
        # Ensure same length
        min_length = min(len(stimulus_recording), len(baseline_recording))
        stim = stimulus_recording[:min_length]
        baseline = baseline_recording[:min_length]
        
        # Apply bandpass filter to both recordings
        nyq = fs / 2
        b, a = signal.butter(4, [freq_band[0]/nyq, freq_band[1]/nyq], btype='band')
        stim_filtered = signal.filtfilt(b, a, stim)
        baseline_filtered = signal.filtfilt(b, a, baseline)
        
        # Calculate baseline statistics
        baseline_std = np.std(baseline_filtered)
        baseline_mean = np.mean(baseline_filtered)
        
        # Detect artifacts using z-score
        z_scores = zscore(stim_filtered)
        artifact_mask = np.abs(z_scores) > threshold
        
        # Additional artifact detection using baseline comparison
        amplitude_ratio = np.abs(stim_filtered) / (np.abs(baseline_filtered) + 1e-10)
        amplitude_artifacts = amplitude_ratio > 3  # Threshold for unusual amplitude ratios
        
        # Combine artifact masks
        combined_artifact_mask = artifact_mask | amplitude_artifacts
        
        # Remove artifacts by interpolation
        cleaned_recording = stim.copy()
        artifact_indices = np.where(combined_artifact_mask)[0]
        
        for idx in artifact_indices:
            # Use local window for interpolation
            window = 50  # points
            start = max(0, idx - window)
            end = min(len(stim), idx + window)
            
            # Find clean points within window
            clean_indices = np.where(~combined_artifact_mask[start:end])[0] + start
            
            if len(clean_indices) > 0:
                # Interpolate using clean points
                cleaned_recording[idx] = np.interp(idx, clean_indices,stim[clean_indices])
        
        return cleaned_recording, combined_artifact_mask

class Plot:
    @staticmethod
    def plot_original(eeg_data):
        plt.plot(eeg_data[:10000])  # teh first (~2 seconds)
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
    def plot_cropped_event_segment(raw, event_name, start_time, end_time):
        start_time = max(0, start_time)
        end_time = min(raw.times[-1], end_time)
        
        duration = end_time - start_time
        
        data = raw.get_data()
        std_dev = np.std(data)
        dynamic_scaling = std_dev * 2
        custom_scalings = {'eeg': dynamic_scaling}
        
        print(f"Using dynamic scaling: {dynamic_scaling:.2e}")
        print(f"Cropping and plotting {event_name} from {start_time}s to {end_time}s")

        # crop
        cropped_raw = raw.copy().crop(tmin=start_time, tmax=end_time)

        fig = cropped_raw.plot(
            scalings=custom_scalings,
            n_channels=1,
            duration=duration,
            title=f"EEG Signal: {event_name} (Cropped)",
            show=False
        )

        save_path = f"wu_venv/ssEEG/denoising_MNE/output/{event_name.lower().replace(' ', '_')}_cropped_segment.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved cropped plot for {event_name} at {save_path}")

    @staticmethod
    def plot_event_segment(raw, event_name, start_time, end_time):
        data = raw.get_data()
        std_dev = np.std(data)
        
        dynamic_scaling = std_dev * 2  
        custom_scalings = {'eeg': dynamic_scaling}
        print(f"Using dynamic scaling: {dynamic_scaling:.2e}")

        duration = end_time - start_time
        fig = raw.plot(
            start=start_time,
            duration=duration,
            scalings=custom_scalings,
            n_channels=1,
            title=f"EEG Signal: {event_name}",
            show=False
        )
        
        save_path = f"wu_venv/ssEEG/denoising_MNE/output/{event_name.lower().replace(' ', '_')}_segment.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot for {event_name} at {save_path}")
        
        
    @staticmethod
    def plot_average_filtered(filtered_raw):
        averaged_signal = Filter.average_signal(filtered_raw)

        plt.figure(figsize=(10, 4))
        plt.plot(averaged_signal, label='Averaged EEG Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Voltage (V)')
        plt.title('Averaged EEG Signal')
        plt.grid()
        plt.legend()
        plt.show()

    @staticmethod
    def plot_comparison(stimulus_recording, cleaned_recording, artifact_mask, fs):
        
        time = np.arange(len(stimulus_recording)) / fs
        
        plt.figure(figsize=(15, 8))
        
        plt.subplot(211)
        plt.plot(time, stimulus_recording, 'b', label='Original')
        plt.plot(time[artifact_mask], stimulus_recording[artifact_mask], 
                'r.', label='Artifacts')
        plt.legend()
        plt.title('Original Recording with Detected Artifacts')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        plt.subplot(212)
        plt.plot(time, cleaned_recording, 'g', label='Cleaned')
        plt.legend()
        plt.title('Cleaned Recording')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.show()
        
        
    def plot_sound_overlap(raw):
        data = raw.get_data()
        std_dev = np.std(data)
        
        dynamic_scaling = std_dev * 2  
        custom_scalings = {'eeg': dynamic_scaling}
        print(f"Using dynamic scaling: {dynamic_scaling:.2e}")

        sound_on_segment = raw.copy().crop(tmin=0, tmax=40).get_data()[0]  # "Sound On" EEG data
        sound_off_segment = raw.copy().crop(tmin=40, tmax=113).get_data()[0]  # "Sound Off" EEG data

        # Time axes
        time_sound_on = np.linspace(0, 40, len(sound_on_segment))
        time_sound_off = np.linspace(40, 113, len(sound_off_segment))

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_sound_on, sound_on_segment, label='Sound On', color='blue', linestyle='-')
        plt.plot(time_sound_off, sound_off_segment, label='Sound Off', color='orange', linestyle='--')

        # Add labels, title, legend, and grid
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('EEG Signal: Sound On vs Sound Off')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.savefig("wu_venv/ssEEG/denoising_MNE/output/sound_on_vs_off.png", dpi=300)
        
        #plt.show()
    
    
    def plot_touch_overlap(raw):
        data = raw.get_data()
        std_dev = np.std(data)
        
        dynamic_scaling = std_dev * 2  
        custom_scalings = {'eeg': dynamic_scaling}
        print(f"Using dynamic scaling: {dynamic_scaling:.2e}")

        touch_whiskers_segment = raw.copy().crop(tmin=113, tmax=173).get_data()[0]  # "Sound On" EEG data
        stop_touching_segment = raw.copy().crop(tmin=173, tmax=279).get_data()[0]  # "Sound Off" EEG data

        # Time axes
        time_start_touching = np.linspace(113, 173, len(touch_whiskers_segment))
        time_stop_touching = np.linspace(173, 279, len(stop_touching_segment))

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_start_touching, touch_whiskers_segment, label='Touching', color='blue', linestyle='-')
        plt.plot(time_stop_touching, stop_touching_segment, label='Not Touching', color='orange', linestyle='--')

        # Add labels, title, legend, and grid
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('EEG Signal: Touching Whiskers vs Not Touching')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.savefig("wu_venv/ssEEG/denoising_MNE/output/touching_whiskers_vs_not.png", dpi=300)


    def plot_filtered_fft(filtered_raw,):
        title = "FFT Graph of Filtered Signal"
        output_path = "wu_venv/ssEEG/denoising_MNE/output/filtered_fft.png"
        data = filtered_raw.get_data(picks='eeg')[0]  # Assuming single-channel EEG
        sfreq = filtered_raw.info['sfreq']  # Sampling frequency in Hz

        # Compute FFT
        fft_values = np.fft.rfft(data)  # Real FFT
        fft_freqs = np.fft.rfftfreq(len(data), d=1/sfreq)  # Frequency axis

        # Compute amplitude spectrum
        amplitude = np.abs(fft_values)

        # Plot the FFT
        plt.figure(figsize=(10, 6))
        plt.plot(fft_freqs, amplitude, color='blue', label='Amplitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid(True)
        plt.legend()

        plt.savefig(output_path, dpi=300)
        print(f"FFT plot saved to {output_path}")
        plt.show()
        
class FFT:
    def compute_psd_plot(raw):
        fig = psd = raw.compute_psd(method='welch', fmin=0.5, fmax=50, n_fft=2048, n_overlap=512)
        psd.plot()
        #plt.show()
        plt.figure()
        psd.plot(dB=True)
        plt.savefig("wu_venv/ssEEG/denoising_MNE/output/mne_psd_line_plot.png", dpi=300)
        #plt.close()


    @staticmethod
    def compute_tfr_multitaper(raw, output_path="output/tfr_multitaper_entire_duration.png"):
        # Compute and save Time-Frequency Representation (TFR) for the entire duration.
        
        # Parameters:
        # - raw: mne.io.Raw, the raw EEG data
        # - output_path: str, path to save the TFR plot
        
        try:
            print("Performing Time-Frequency Analysis for the entire duration...")
            
            # Extract the data and create an Epochs-like object for the entire signal
            data = raw.get_data(picks='eeg')  # Shape: (n_channels, n_times)
            times = raw.times  # Time array corresponding to the data
            
            # Define frequency range for TFR analysis
            freqs = np.linspace(0.5, 40, 100)  # 0.5 to 40 Hz, 100 frequency steps
            n_cycles = freqs / 2.0  # Number of cycles per frequency
            
            # Compute the time-frequency representation (TFR)
            power = tfr_multitaper(
                raw,
                freqs=np.linspace(0.5, 40, 100),
                n_cycles=freqs / 2.0,
                time_bandwidth=2.0,
                picks="eeg",
                average=False,
                return_itc=False
            )

            # Directly plot the TFR
            power.plot(
                baseline=(None, 0),
                mode="logratio",
                title="TFR for Entire Duration"
            )
            plt.savefig("wu_venv/ssEEG/denoising_MNE/output/tfr_multitaper_entire_duration.png", dpi=300)
            plt.show()
            
        except Exception as e:
            print(f"An error occurred during TFR analysis: {e}")
        
class Events:
    
    # FOR CSV 00005
    # @staticmethod
    # def get_events_dict():
    #     return {
    #         "Sound On": (0, 40),          # From 0s to 40s in EEG time
    #         "Sound Off": (40, 113),       # From 40s to 113s
    #         "Touch Whiskers": (113, 173), # From 113s to 173s
    #         "Stop Touching": (173, 280),  # From 173s to 280s
    #     }


    # FOR CSV 00006:
    @staticmethod
    def get_events_dict():
        return {
            "Sound On": (0, 22),          # From 0s to 40s in EEG time
            "Sound Off": (22, 92),       # From 40s to 113s
            "Touch Whiskers": (92, 152), # From 113s to 173s
            "Stop Touching": (152, 279),  # From 173s to 280s
        }