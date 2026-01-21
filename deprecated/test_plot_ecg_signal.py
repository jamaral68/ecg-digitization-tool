import numpy as np
import pandas as pd

# plotting
from matplotlib import pyplot as plt

# sample data
from scipy.misc import electrocardiogram
from edt_utils import plot_ecg_signal, plot_ecg


# ecg = electrocardiogram()
# # Sampling frequency in Hz
# fs = 360

# n = int(fs * 10)
# signal = ecg[:n]
# # time in sec
# ts = np.arange(signal.size) / fs



filename = 'bucket/img20250221_12050781'
image_name = filename + '.png'
fs = 500

csv_name =filename + '.csv'

df = pd.read_csv(csv_name,index_col=0)
print(df.columns)
plot_ecg(df,df.columns,filename, n_rows = 3, n_columns = 4, fs = 500, figure_size = (20, 12))