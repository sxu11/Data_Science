

#http://mlwhiz.com/blog/2017/12/26/How_to_win_a_data_science_competition/
import os
import pandas as pd
from itertools import product
import numpy as np

data_folder = os.curdir + "/data/"
sales = pd.read_csv(data_folder + 'sales_train.csv')


# Create "grid" with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    #print [cur_shops, cur_items, [block_num]]

    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)


sales = sales[sales.item_price<100000]
sales = sales[sales.item_cnt_day<=1000]


sales_m = sales.groupby(['date_block_num','shop_id','item_id'],as_index=False).agg({'item_cnt_day': 'sum','item_price': np.mean})
# print sales_m.head(10)
# sales_m = sales_m.reset_index()
# sales_m = sales_m.reset_index()
# print sales_m.head(10)


items = pd.read_csv(data_folder + 'test.csv')

sales_m = pd.merge(grid,sales_m,on=['date_block_num','shop_id','item_id'],how='left').fillna(0)
# adding the category id too
sales_m = pd.merge(sales_m,items,on=['item_id'],how='left')

for type_id in ['item_id','shop_id','item_category_id']:
    for column_id,aggregator,aggtype in [('item_price',np.mean,'avg'),('item_cnt_day',np.sum,'sum'),('item_cnt_day',np.mean,'avg')]:

        mean_df = sales.groupby([type_id,'date_block_num']).aggregate(aggregator).reset_index()[[column_id,type_id,'date_block_num']]
        mean_df.columns = [type_id+'_'+aggtype+'_'+column_id,type_id,'date_block_num']

        sales_m = pd.merge(sales_m,mean_df,on=['date_block_num',type_id],how='left')


