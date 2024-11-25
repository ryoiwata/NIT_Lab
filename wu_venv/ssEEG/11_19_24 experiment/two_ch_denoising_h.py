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


class Menu:
    def menu(option, csv_path):
        while True:
            print("\nSelect an option:")
            print("1. Plot Raw Signal (CSV)")
            print("2. Plot Unfiltered Fourier")
            print("3. Plot Filtered Fourier")
            print("4. Plot Filtered PSD Diagram")
            print("5. Downsample the Data")
            print("6. Inspect the Signal")
            print("7. Detrend the Signal")
            print("8. Apply Bandpass")
            print("9. Apply 60,50 Hz Notch")
            print("10. Average Signal")
            print("11. Apply FastICA")
            print("12. Apply all Filters & plot")
            print("13. Start New Analysis")
            print("14. Close Analysis")
            
            try:
                if option == 14:
                    print("Closing analysis. Goodbye!")
                    break  # Exit the loop if "Close Analysis" is selected
                elif 1:
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    Plot.plot_original(eeg_data)
                elif 2:
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    raw = DataPreprocess.convert_to_fif(eeg_data)
                    Plot.plot_unfiltered_fft(raw)

                elif 3:
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    raw = DataPreprocess.convert_to_fif(eeg_data)
                    filtered_raw = Filter.apply_all_filters(raw)
                    Plot.plot_filtered_fft(filtered_raw)

                elif 4: 
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    raw = DataPreprocess.convert_to_fif(eeg_data)
                    filtered_raw = Filter.apply_all_filters(raw)
                    FFT.compute_tfr_multitaper(raw, output_path="wu_venv/ssEEG/11_19_24 experiment/new_output/tfr_multitaper_entire_duration.png")

                elif 5: 
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    raw = DataPreprocess.convert_to_fif(eeg_data)
                    Filter.apply_downsampling(raw, new_sfreq=None)                  
                    
                elif 6: 
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    raw = DataPreprocess.convert_to_fif(eeg_data)
                    Filter.inspect_signal(raw)

                elif 7:
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    raw = DataPreprocess.convert_to_fif(eeg_data)
                    Filter.apply_detrend(raw)
                    
                elif 8:
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    raw = DataPreprocess.convert_to_fif(eeg_data)
                    Filter.apply_bandpass_filter
                    
                elif 9: 
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    raw = DataPreprocess.convert_to_fif(eeg_data)
                    Filter.apply_notch_filter(raw)
                    
                elif 10:
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    raw = DataPreprocess.convert_to_fif(eeg_data)
                    Filter.average_signal(raw)
                    
                elif 11:
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    raw = DataPreprocess.convert_to_fif(eeg_data)
                    Filter.apply_fastICA(raw)
                    
                elif 12:
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    DataPreprocess.convert_to_fif(eeg_data)
                    filtered_raw = Filter.apply_all_filters(raw)
                    Plot.plot_filtered_raw(filtered_raw)
                    
                elif 13:
                    eeg_data = DataPreprocess.remove_missing(csv_path)
                    DataPreprocess.convert_to_fif(eeg_data)
                    new_csv = input("Which file would you like to look at next?: ")
                else:
                    print("Choose a valid option")
            except ValueError:
                print("Pick between 1-14. That's all we have so far.")


class DataPreprocess:
    # in order to use the .csv file with mne, we need to first 
    # convert it to a fif file format 
    def remove_missing(csv_path): 
        df = pd.read_csv(csv_path, skiprows=11, engine='python')  # Skip metadata rows
        df.columns = df.columns.str.strip()  # Strip whitespace from column names

        # Debugging: Print column names
        print("Column names in the file:", df.columns)

        # Verify required columns
        required_columns = ['Volt', 'Volt2']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Missing expected column: {col}")

        # Drop rows with NaN values
        df = df.dropna(subset=required_columns)
        
        # Extract and transpose data for MNE (shape: 2 x samples)
        eeg_data = df[required_columns].values.T
        
        # Verify sample count
        expected_samples = 70000  # Updated based on your file
        if eeg_data.shape[1] != expected_samples:
            print(f"Warning: Expected {expected_samples} samples, but got {eeg_data.shape[1]} samples.")
        
        return eeg_data


        
    def convert_to_fif(eeg_data):
        try:
            sfreq = 1000  # Sampling frequency (adjust as needed)
            ch_names = ['EEG1', 'EEG2']  # Two channels
            ch_types = ['eeg', 'eeg']  # Both are EEG signals

            # Create MNE Info object
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            
            # EEG data must have shape (channels, samples), which is already ensured
            raw = mne.io.RawArray(eeg_data, info)

            # Save to FIF format
            fif_path = "eeg_data_raw.fif"
            raw.save(fif_path, overwrite=True)
            message = f"CSV converted to FIF and saved at {fif_path}"
            return raw, message  # Return both the raw object and a message

        except Exception as e:
            print(f"Error occurred: {e}")
            return None, str(e)
 
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
    def apply_downsampling(raw, new_sfreq=None): # default downsample rate is 5000 hZ
        if new_sfreq is None:
            try:
                new_sfreq = float(input("Enter the new sampling frequency (Hz): "))
            except ValueError:
                print("Use numbers.")
                return raw
    
        downsampled_raw = raw.resample(new_sfreq, npad="auto")  # Automatically pad data
        print(f"Data downsampled to {new_sfreq} Hz.")
        raw = downsampled_raw
        return raw

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
        data = raw.get_data()
        ica = FastICA(n_components=data.shape[0], random_state=42)  # Match components to number of channels
        transformed_data = ica.fit_transform(data.T).T  # Transpose for (samples, channels)
        raw._data = transformed_data
        return raw
    
    
    @staticmethod
    def apply_all_filters(raw):
        raw = Filter.apply_detrend(raw)
        raw = Filter.apply_notch_filter(raw)
        raw = Filter.apply_bandpass_filter(raw)
        raw = Filter.apply_fastICA(raw)
        return raw


class Plot:
    @staticmethod
    def plot_original(eeg_data):
        plt.figure(figsize=(10, 6))
        plt.plot(eeg_data[0, :10000], label='Channel 1')  # First channel
        plt.plot(eeg_data[1, :10000], label='Channel 2')  # Second channel
        plt.xlabel("Sample Index")
        plt.ylabel("Voltage (V)")
        plt.title("Subset of Raw EEG Signal from .csv")
        plt.legend()
        plt.savefig("wu_venv/ssEEG/denoising_MNE/output/original_eeg_plot.png", dpi=300)
        plt.show()

    @staticmethod
    def plot_raw(raw):
        fig = raw.plot(scalings='auto', n_channels=1, duration=40, title="Raw EEG Signal", show=False)
        fig.savefig("wu_venv/ssEEG/11_19_24 experiment/new_output/raw_eeg_plot.png", dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_filtered_raw(filtered_raw):
        fig = filtered_raw.plot(scalings='auto', n_channels=1, duration=40, title="Filtered EEG Signal", show=False)
        fig.savefig("wu_venv/ssEEG/11_19_24 experiment/new_output/filtered_eeg_plot.png", dpi=300)
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

        save_path = f"wu_venv/ssEEG/11_19_24 experiment/new_output/{event_name.lower().replace(' ', '_')}_cropped_segment.png"
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
        
        save_path = f"wu_venv/ssEEG/11_19_24 experiment/new_output/{event_name.lower().replace(' ', '_')}_segment.png"
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

        plt.savefig("wu_venv/ssEEG/11_19_24 experiment/new_output/sound_on_vs_off.png", dpi=300)
        
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

    def plot_unfiltered_fft(raw):
        title = "FFT Graph of Uniltered Signal"
        output_path = "wu_venv/ssEEG/11_19_24 experiment/new_output/unfiltered_fft.png"
        data = raw.get_data(picks='eeg')[0]  # Assuming single-channel EEG
        sfreq = raw.info['sfreq']  # Sampling frequency in Hz

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

    def plot_filtered_fft(filtered_raw,):
        title = "FFT Graph of Filtered Signal"
        output_path = "wu_venv/ssEEG/11_19_24 experiment/new_output/filtered_fft.png"
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
        plt.savefig("wu_venv/ssEEG/11_19_24 experiment/new_output/mne_psd_line_plot.png", dpi=300)
        #plt.close()


    @staticmethod
    def compute_tfr_multitaper(raw, output_path="wu_venv/ssEEG/11_19_24 experiment/new_output/tfr_multitaper_entire_duration.png"):
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
            plt.savefig("wu_venv/ssEEG/11_19_24 experiment/new_output/tfr_multitaper_entire_duration.png", dpi=300)
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