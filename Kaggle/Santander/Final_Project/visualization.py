
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd
from collections import Counter
import numpy as np

sales_train_small = pd.read_csv('raw_data/sales_train_small.csv')
print sales_train_small.shape
print sales_train_small.head()

avg_prices_by_itemid = sales_train_small[['item_id','item_price']].groupby(['item_id'], as_index=False).mean()
avg_cnts_by_itemid = sales_train_small[['item_id','item_cnt_day']].groupby(['item_id'], as_index=False).mean()

prices_cnts_by_itemid = pd.merge(avg_prices_by_itemid,avg_cnts_by_itemid, how='inner', on='item_id')
plt.scatter(prices_cnts_by_itemid['item_price'], prices_cnts_by_itemid['item_cnt_day'])
plt.show()
quit()

# ha = np.log(1+sales_train_small['item_cnt_day'].values)
# ha, he = zip(*Counter(sales_train_small['item_price']).items())
# plt.scatter(ha, he)
# print Counter(sales_train_small['item_cnt_day'])
item_prices = sales_train_small['item_price']
item_cnts = sales_train_small['item_cnt_day'].values
plt.scatter(item_prices, item_cnts)
plt.show()