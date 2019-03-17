import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#def prev_month_predictor():

df = pd.read_csv('sales_train.csv')
print df.shape

df['by_month'] = df["date_block_num"]%12
df = df[["date_block_num", 'item_id', 'shop_id', 'item_cnt_day']]

# ignore shop id for now
item_shop_months = df.groupby(['date_block_num', 'item_id', 'shop_id'], as_index=False).sum()

item_shop_nov = item_shop_months.loc[item_shop_months['date_block_num']==33]
#print item_shop_months.columns
#print item_shop_months['by_month']==10
item_shop_nov = item_shop_nov[['item_id', 'shop_id', 'item_cnt_day']]
#print item_shop_nov
#print item_shop_nov

df1 = df[["date_block_num", 'item_id', 'item_cnt_day']]
item_shop_months1 = df1.groupby(['date_block_num', 'item_id'], as_index=False).sum()
item_shop_nov1 = item_shop_months1.loc[item_shop_months1['date_block_num']==33]
item_shop_nov1 = item_shop_nov1[['item_id', 'item_cnt_day']]


## creat a dictionary
my_dict = {}
for index, train_row in item_shop_nov.iterrows():
    my_dict[(train_row['item_id'], train_row['shop_id'])] = train_row['item_cnt_day']



my_dict1 = {}
for index1, train_row1 in item_shop_nov1.iterrows():
    my_dict1[(train_row1['item_id'])] = train_row1['item_cnt_day']


test_df = pd.read_csv('test.csv')
print 'test_df.shape:', test_df.shape
no_key_cnt = 0
item_key_cnt = 0
preds = []
for index, test_row in test_df.iterrows():
    if my_dict.has_key((test_row['item_id'], test_row['shop_id'])):
        res = my_dict[(test_row['item_id'], test_row['shop_id'])]
        res = np.clip(res, 0, 20)
        preds.append(res)
    # elif my_dict1.has_key(test_row['item_id']):
    #     item_key_cnt += 1
    #     #preds.append(-1)
    #     res = my_dict1[test_row['item_id']]
    #     res = np.clip(res, 0, 20)
    #     preds.append(res)
    else:
        preds.append(0)
        no_key_cnt += 1
        # if there is no entry, use item avg to predict

print 'item_key_cnt:', item_key_cnt
print 'no_key_cnt:', no_key_cnt

test_df['item_cnt_month'] = pd.Series(preds, index=test_df.index)

preds_df = test_df[['ID', 'item_cnt_month']]
preds_df.to_csv("fourth_submission.csv", index=False)
