from denoising_h import Denoising_EEG



file_path = "wu_venv/ssEEG/10_29_24 experiment/csv_files/WAVE4_Flick.CSV"
df = pd.read_csv(file_path, skiprows=11)
print(df.head())

df.columns = ['index', 'CH1_Voltage(mV)']  # read in the 3 columns and name them for clarity. we will focus on Ch1 here.

time = df['index'].values  # assign "second" col to be time data 
signal = df['CH1_Voltage(mV)'].values  # assign "Ch1" col to be signal data 

signal = detrend(signal)

# to be safe, we check for NaN or infinite values in the signal which can mess with later calculations
if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
    raise ValueError("signal has NaN or infinite vals")


# Apply notch filter
signal_notch = Denoising_EEG.notch_filter(signal, notch_freq, fs)

# Apply bandpass filter
signal_bandpassed = Denoising_EEG.bandpass_filter(signal_notch, lowcut, highcut, fs)

# Apply wavelet denoising
signal_denoised = Denoising_EEG.wavelet_denoise(signal_bandpassed)

