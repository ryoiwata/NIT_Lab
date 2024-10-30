# denoising an EEG signal 
import pywt
import pandas as pd

# link to docs: 
# https://pywavelets.readthedocs.io/en/latest/regression/dwt-idwt.html

file_path = "wu_venv/WAVE3.csv"

data = pd.read_csv(file_path, skiprows=11)
data.columns= ['Index', 'CH1_Voltage']

cA, cD = pywt.dwt(data, 'db2')

print(cA)
print(cD)