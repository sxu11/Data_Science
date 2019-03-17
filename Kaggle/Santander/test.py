
import pandas as pd
import numpy as np

if True:
    a = {}

    a[1] = [1,'ha',2]
    a_keys_pd = pd.DataFrame(a.keys())
    a_vals_pd = pd.DataFrame(a.values())

    a_keys_pd.to_csv('a_keys_pd.csv')
    a_vals_pd.to_csv('a_vals_pd.csv')
    #('a_vals.txt', a.values())
else:
    a = pd.DataFrame.from_csv('a_vals_pd.csv')
    print a.values

quit()

df = pd.read_csv("train_ver2_last1000.csv")
colnames = df.columns
ha = df['fecha_dato'] #df[colnames[25:]]

inds_sorted = np.argsort(ha)

print ha[inds_sorted]
quit()




my_mat = np.zeros((24,24))

row, col = ha.shape
for i in range(row):
    for j in range(col):
        if ha.iloc[i,j] == 1:
            ''#my_mat[]
        elif ha.iloc[i,j] > 1:
            print '>1'
        else:
            continue

quit()




print df[colnames[25:]].shape