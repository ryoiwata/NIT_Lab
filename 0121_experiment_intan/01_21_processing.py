# imports: 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, detrend, welch
import numpy as np
import pywt
import mne
from mne.preprocessing import ICA
import os
from mne.time_frequency import tfr_multitaper
import sklearn
from sklearn.decomposition import FastICA
import csv
from obspy import Stream, Trace
from obspy.imaging.spectrogram import spectrogram
from obspy.signal.tf_misfit import plot_tfr
from scipy.interpolate import interp1d
from functions import CSV
from obspy import Stream, Trace
from obspy.imaging.spectrogram import spectrogram
from obspy.signal.tf_misfit import plot_tfr
from scipy.interpolate import interp1d
from scipy.signal import welch


filtered_paths= ['void',
r'ssEEG\0121_experiment_intan\filtered_sets\rec1_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec2_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec3_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec4_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec5_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec6_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec7_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec8_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec9_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec10_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec11_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec12_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec13_filtered.set',
r'ssEEG\0121_experiment_intan\filtered_sets\rec14_filtered.set']

unfiltered_paths = ['void',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec1.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec2.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec3.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec4.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec5.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec6.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec7.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec8.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec9.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec10.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec11.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec12.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec13.set',
r'ssEEG\0121_experiment_intan\unfiltered_sets\rec14.set']

filtered_csvs = ['void',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec1_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec2_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec3_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec4_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec5_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec6_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec7_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec8_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec9_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec10_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec11_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec12_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec13_filtered.csv',
r'ssEEG\0121_experiment_intan\filtered_csv_files\rec14_filtered.csv']

unfiltered_csvs = ['void',
r'ssEEG\0121_experiment_intan\csv_files\rec1.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec2.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec3.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec4.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec5.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec6.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec7.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec8.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec9.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec10.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec11.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec12.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec13.csv',
r'ssEEG\0121_experiment_intan\csv_files\rec14.csv']

scaled_filtered_csvs=['void',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec2_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec3_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec4_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec5_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec6_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec7_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec8_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec9_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec10_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec11_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec12_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec13_filtered_uV.csv',
r'ssEEG\0121_experiment_intan\scaled_filtered_csv\rec14_filtered_uV.csv']

averaged_csvs=['void',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec1_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec2_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec3_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec4_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec5_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec6_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec7_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec8_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec9_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec10_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec11_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec12_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec13_filtered_avg.csv',
r'ssEEG\0121_experiment_intan\averaged_csvs\rec14_filtered_avg.csv']

if __name__ == "__main__": 
    try:             
        for i, filtered_csv in enumerate(filtered_csvs[1:], start=10): 
            csv_path = filtered_csv  
            column_index = 5
            sample_rate = 250
            title = f"Recording {i}" 

            CSV.plot_psd(csv_path, sample_rate, channel_labels=None)

    except Exception as e:
        print(f"Error: {e}")