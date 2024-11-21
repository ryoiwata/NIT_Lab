import pandas as pd
import numpy as np
import mne
from scipy.signal import detrend
from two_ch_denoising_h import DataPreprocess, Plot, Filter, Events, FFT
import matplotlib.pyplot as plt
import os
from mne.time_frequency import tfr_multitaper



# Define the file path
file_path = "wu_venv/ssEEG/11_19_24 experiment/csv_files/SDS00008.csv"
eeg_data = DataPreprocess.remove_missing(file_path)
result = DataPreprocess.convert_to_fif(eeg_data)
print(result)
def main():
    
    
    try:
        # Step 1: Load the EEG data from the CSV file
        eeg_data = DataPreprocess.remove_missing(file_path)
        print("Successfully loaded EEG data.")

        
        # Step 2: Convert the EEG data to a Raw MNE object and save as .fif
        raw, message = DataPreprocess.convert_to_fif(eeg_data)
        print(message)

        

        if raw:
            #Filter.inspect_signal(raw)
            
            Plot.plot_raw(raw)
            
            print("Existing SSP Projectors:", raw.info["projs"])

            if len(raw.info["projs"]) > 0:
                print("Projectors are present. Removing them before further processing.")
                raw.del_proj()  # Remove existing projectors
                
            Filter.apply_fastICA(raw)
            # # Step 3: Apply filters and get the filtered raw data
            filtered_raw = Filter.apply_all_filters(raw)

            # Downsample the filtered data
            filtered_raw = Filter.apply_downsampling(filtered_raw, new_sfreq=200)
            
            Plot.plot_filtered_fft(filtered_raw)
            Plot.plot_filtered_raw(raw) 

            Filter.inspect_signal(filtered_raw)

            # Step 4: Plot segments based on events
            events_dict = Events.get_events_dict()
            for event_name, (start_time, end_time) in events_dict.items():
                print(f"Plotting event: {event_name} (from {start_time}s to {end_time}s)")
                Plot.plot_event_segment(filtered_raw, event_name, start_time, end_time)
                Plot.plot_cropped_event_segment(filtered_raw, event_name, start_time, end_time)
            
            FFT.compute_psd_plot(raw)
            
            FFT.compute_tfr_multitaper(raw, output_path="output/tfr_multitaper_entire_duration.png")
            FFT.plot_fft_psd(raw)
            
            Plot.plot_sound_overlap(filtered_raw)
            Plot.plot_touch_overlap(filtered_raw)
            #Filter.average_signal(filtered_raw)
            
            filtered_fif_path = DataPreprocess.return_filtered_fif_path(raw)
            
            # if raw:
            #     DataPreprocess.convert_to_csv("path_to_filtered_data.fif")
            # else:
            #     print("Raw data not available for conversion to CSV.")
            # cleaned_csv_path = DataPreprocess.convert_to_csv(filtered_fif_path)
            
            # filtered_df = pd.read_csv(cleaned_csv_path)
            # time = filtered_df['Time (s)'].values
            # voltage = filtered_df['Voltage (V)'].values
            
            # print("Applying Fourier Transform on the filtered data...")
            # freqs = np.fft.rfftfreq(len(voltage), d=(time[1] - time[0]))
            # fft_values = np.fft.rfft(voltage)
            
            # plt.figure(figsize=(10, 5))
            # plt.plot(freqs, np.abs(fft_values))
            # plt.title("Fourier Transform of Filtered EEG Signal")
            # plt.xlabel("Frequency (Hz)")
            # plt.ylabel("Amplitude")
            # plt.grid()
            # plt.savefig("wu_venv/ssEEG/denoising_MNE/output/clean_fft_plot.png", dpi=300)
            # plt.show()
            # print("Fourier Transform plot saved to output directory.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()