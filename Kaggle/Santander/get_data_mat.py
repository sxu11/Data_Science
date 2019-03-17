


import pandas as pd
import numpy as np


df = pd.read_csv("all/train_ver2.csv")
colnames = df.columns

np.savetxt('data_mat.txt', df.values)
