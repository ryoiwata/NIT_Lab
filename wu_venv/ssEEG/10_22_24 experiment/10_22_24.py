import pywt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter 


def remove_60hZ(signal):
    
    
def denoise_signal(signal, wavelet, mode='soft', level=1):
    coeff = pywt.wavedec(signal, wavelet, level=level)
    sigma = (1/0.6745) * mad(coeff[-level])
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeff[1:] = (pywt.threshold(i, value=threshold, mode=mode) for i in coeff[1:])
    return pywt.waverec(coeff, wavelet)

def mad(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)

# load EEG data from experiment
file_path = "wu_venv/ssEEG/10_22_24 experiment/WAVE3.csv"
eeg_data = pd.read_csv(file_path, skiprows=11)
eeg_data.columns = ['Index', 'CH1_Voltage(mV)']

# Extract the signal
signal = eeg_data['CH1_Voltage(mV)'].values

# Denoise the signal
denoised_signal = denoise_signal(signal, 'db4', mode='soft', level=1)

# Plotting the original and denoised signals
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(signal, label='Original Signal')
plt.title('Original EEG Signal')
plt.xlabel('Sample Index')
plt.ylabel('Voltage (mV)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(denoised_signal, label='Denoised Signal after DWT', color='red')
plt.title('EEG Signal After Denoising using DWT')
plt.xlabel('Sample Index')
plt.ylabel('Voltage (mV)')
plt.legend()

plt.tight_layout()
plt.show()
