
import pandas as pd
import numpy as np
import os


# df = pd.read_csv('sales_train.csv')
data_folder = os.curdir + "/data/"

df = pd.read_csv(data_folder + 'sales_train_part.csv')

#print df.tail(10)

#print df['item_cnt_day'].value_counts()

#cols: date  date_block_num  shop_id  item_id  item_price  item_cnt_day

#print df['item_id'].value_counts()
#print df.loc[df['item_id'] == 20969]


# ``time of appearance" for each shop, for each item?


quit()

print df['date_block_num'].value_counts()
#print df.head(20)

# a = df['date_block_num'].value_counts()
# print a
quit()

import matplotlib.pyplot as plt
cnts, bins = np.histogram(df['item_cnt_day'], bins=150)

plt.plot((bins[1:]+bins[:-1])/2., cnts)
plt.show()

# sales_train_part = df.sample(n=100000)
# sales_train_part.to_csv("sales_train_part.csv", index=False)