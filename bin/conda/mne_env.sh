conda create --channel=conda-forge --strict-channel-priority -p ./mne mne-base
pip install "mne[hdf5]" 
conda install conda-forge::jupyterlab --yes
conda install conda-forge::pandas --yes
conda install conda-forge::openpyxl --yes
conda install conda-forge::scikit-learn --yes
conda install -c edeno spectral_connectivity --yes
conda install seaborn -c conda-forge --yes
