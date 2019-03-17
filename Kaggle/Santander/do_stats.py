

import pandas as pd
import numpy as np

#a_keys_pd = pd.DataFrame.from_csv("a_keys_pd.csv")
a_vals_pd = pd.DataFrame.from_csv("a_vals_pd.csv")

row, col = a_vals_pd.shape

for i in range(row):
    ha = a_vals_pd.ix[i,:].values
    #print ha
    for j in range(len(ha)-1):
        if np.isnan(ha[j+1]):
            do_stats(ha)