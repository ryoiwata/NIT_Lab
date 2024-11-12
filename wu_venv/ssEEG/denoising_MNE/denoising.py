import pandas as pd
import numpy as np
import mne
from scipy.signal import detrend
from denoising_h import DataPreprocess

# Define the file path
file_path = "wu_venv/ssEEG/10_29_24 experiment/csv_files/WAVE4_Flick.CSV"
# eeg_data = DataPreprocess.remove_missing(file_path)
# result = DataPreprocess.convert_to_fif(eeg_data)
# print(result)

df = pd.read_csv(file_path, skiprows=10)
print("Columns in CSV:", df.columns)


eeg_data = DataPreprocess.remove_missing(file_path)
DataPreprocess.convert_to_fif(eeg_data)

duration = len(eeg_data) / 5000
print(f"Expected recording duration: {duration:.2f} seconds")

