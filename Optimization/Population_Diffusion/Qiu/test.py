
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('vl_embedding.csv', sep='\t', header=None)
print df.describe()
quit()

df_sample = df.sample(n=1000, random_state=0)
plt.scatter(df_sample[0], df_sample[1])
plt.show()