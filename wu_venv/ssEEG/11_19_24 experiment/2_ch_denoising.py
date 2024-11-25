import pandas as pd
import numpy as np
import mne
from scipy.signal import detrend
from two_ch_denoising_h import DataPreprocess, Plot, Filter, Events, FFT, Menu
import matplotlib.pyplot as plt
import os
from mne.time_frequency import tfr_multitaper


def main():
    csv_path = "wu_venv/ssEEG/11_19_24 experiment/csv_files/SDS00008.csv"  # Define your CSV file path
    option = int(input("\nEnter your choice (1-14): "))
    Menu.menu(csv_path, option)

if __name__ == "__main__":
    main()