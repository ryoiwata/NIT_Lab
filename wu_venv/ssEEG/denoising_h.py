import matplotlib as plt
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, detrend
import numpy as np
import pywt
from mne.preprocessing import ICA

class Denoising_EEG:

    @staticmethod
    def notch_filter(signal, freq, fs, quality_factor=30):
        b, a = iirnotch(freq, quality_factor, fs)
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
