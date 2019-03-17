

import numpy as np
import pandas as pd

##
data_folder = 'all'
data_file = data_folder + '/' + 'train_ver2.csv'
test_file = 'train_ver2_last1000.csv'

data = pd.read_csv(data_file)

all_customers = data['ncodpers']


##
from collections import Counter
a = Counter(all_customers)

df_counter = pd.DataFrame.from_dict(a, orient='index')
df_counter.to_csv('df_counter__ncodpers.csv')
df_counter.plot(kind='bar')

##
import matplotlib.pyplot as plt
plt.show()