conda create --channel=conda-forge --strict-channel-priority -p ./mne mne-base
pip install "mne[hdf5]" 
conda install conda-forge::jupyterlab --yes
conda install conda-forge::pandas --yes
conda install conda-forge::openpyxl --yes
conda install conda-forge::scikit-learn --yes

