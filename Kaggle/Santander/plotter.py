


import pandas as pd
import matplotlib.pyplot as plt

a = pd.read_csv('df_counter.csv')
#print a

colnames = a.columns
a.plot(kind='bar')

dates = a[colnames[0]]
print sorted(dates)
plt.xticks(range(len(dates)), dates)

plt.show()
