import pandas as pd
from matplotlib import pyplot as plt
from edt_utils import plot_ecg

df = pd.read_csv("img20250221_052.csv",index_col=[0])
plot_ecg(df,df.columns,"img20250221.csv", n_rows = 3, n_columns = 4,
         x_spacing=100,y_spacing=0.2, figure_size = (20, 30))
plt.show()