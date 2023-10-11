# Download and pre-process the PTB-XL dataset

Locate inside the data directory
```
cd data_folder_ptb_xl/
```

Download the data
```
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.1/
```

Open the jupyter notebook 'ecg_data_preprocessing.ipynb'

Run the notebook, and the PTB-XL dataloaders are at the end.

Note: these dataloaders are highly functional and allows you to have each patient metadata for further research, however, if you only want the signals and labels, you can also downdload from [here](https://mega.nz/folder/UfUDFYjS#YYUJ3CCUGb6ZNmJdCZLseg)
