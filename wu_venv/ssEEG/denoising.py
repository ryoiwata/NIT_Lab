import pandas as pd
import numpy as np
import mne

# Step 1: Load the CSV Data
file_path = "wu_venv/ssEEG/10_29_24 experiment/csv_files/WAVE4_Flick.CSV"
df = pd.read_csv(file_path, skiprows=11)

# Rename columns for clarity
df.columns = ['index', 'CH1_Voltage(mV)']

# Step 2: Handle Missing Data
if df['CH1_Voltage(mV)'].isna().any():
    print("Warning: Missing values detected in the data. Replacing with zeros.")
    df['CH1_Voltage(mV)'].fillna(0, inplace=True)

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

# Plot the power spectral density (PSD) to inspect noise levels
raw.plot_psd(fmax=50)

# Apply a bandpass filter (0.5-40 Hz) to focus on the typical EEG frequency range
raw.filter(l_freq=0.5, h_freq=40, filter_length='auto')

# Plot the cleaned data
raw.plot(block=True)