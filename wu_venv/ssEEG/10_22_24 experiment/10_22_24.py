import pywt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter 




file_path = "wu_venv/ssEEG/10_22_24 experiment/WAVE3.csv"
df = pd.read_csv(file_path, skiprows=11)
print(df.head())

def remove_60hz():
    None

def perform_bandpass():
    None
def perform_ICA():
    None
    
    
    