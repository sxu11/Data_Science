
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


data_folder = os.curdir + "/data/"
df_orig = pd.read_csv(data_folder + 'sales_train.csv')
df_orig = df_orig[["date_block_num", 'item_id', 'shop_id', 'item_cnt_day']]

# data cleaning?

# Sum daily sales into monthly sales:
df = df_orig.groupby(['date_block_num', 'item_id', 'shop_id'], as_index=False).sum()
df.rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=True)

df["prev_month_ind"] = df["date_block_num"] - 1

row_train, col_train = df.shape
print 'df.columns:', df.columns
# Index([u'date_block_num', u'item_id', u'shop_id', u'item_cnt_month', u'prev_month_ind'], dtype='object')
print 'df.shape:', df.shape
# (1609124, 5)

test_df = pd.read_csv(data_folder + 'test.csv')
row_test, col_test = test_df.shape

test_df['date_block_num'] = pd.Series(np.ones((row_test,))*34, index=test_df.index)
test_df['prev_month_ind'] = pd.Series(np.ones((row_test,))*33, index=test_df.index)
test_df['item_cnt_month'] = pd.Series(np.zeros((row_test,)), index=test_df.index)
print 'test_df.shape', test_df.shape

df = df.append(test_df[['date_block_num',
                   'item_id',
                   'shop_id',
                   'item_cnt_month',
                   'prev_month_ind']],ignore_index=True)
print 'df.shape', df.shape

my_dict = {}
for _, data_row in df.iterrows():
    curr_key = (data_row['date_block_num'], data_row['item_id'], data_row['shop_id'])
    my_dict[curr_key] = data_row['item_cnt_month']

prev_month_cnts = []
has_prev_month, not_prev_month = 0, 0
for _, data_row in df.iterrows():
    curr_key = (data_row['prev_month_ind'], data_row['item_id'], data_row['shop_id'])
    if my_dict.has_key(curr_key):
        res = np.clip(my_dict[curr_key], 0, 20)
        prev_month_cnts.append(res)
        has_prev_month += 1
    else:
        prev_month_cnts.append(0)
        not_prev_month += 1


df['prev_month_cnt'] = pd.Series(prev_month_cnts, index=df.index)


print df.shape, row_train

test_df['item_cnt_month'] = df.loc[row_train:, 'prev_month_cnt'].values
print df.loc[row_train:, 'prev_month_cnt']

preds_df = test_df[['ID', 'item_cnt_month']]

preds_df.to_csv("fifth_submission.csv", index=False)
