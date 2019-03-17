
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn import linear_model, tree
import os
from sklearn.ensemble import GradientBoostingRegressor

data_folder = os.curdir + "/data/"
pickle_name = 'tmp.pkl'
use_saved_pickle = os.path.exists(pickle_name)

if use_saved_pickle:
    df = pd.read_pickle(pickle_name)

else:

    df_orig = pd.read_csv(data_folder + 'sales_train.csv')
    df_orig = df_orig[["date_block_num", 'item_id', 'shop_id', 'item_cnt_day']]

    # data cleaning?

    # Sum daily sales into monthly sales:
    df = df_orig.groupby(['date_block_num', 'item_id', 'shop_id'], as_index=False).sum()
    row_train, col_train = df.shape
    df.rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=True)
    df['ID'] = pd.Series(np.ones((row_train,))*-1, index=df.index)

###
    test_df = pd.read_csv(data_folder + 'test.csv')
    row_test, col_test = test_df.shape

    test_df['date_block_num'] = pd.Series(np.ones((row_test,))*34, index=test_df.index)
    #test_df['prev_month_ind'] = pd.Series(np.ones((row_test,))*33, index=test_df.index)
    test_df['item_cnt_month'] = pd.Series(np.zeros((row_test,)), index=test_df.index)

    df = df.append(test_df[['date_block_num',
                       'item_id',
                       'shop_id',
                       'item_cnt_month',
                       'ID']],ignore_index=True)

    df["prev_month_ind"] = df["date_block_num"] - 1
    df["prev_year_ind"] = df["date_block_num"] - 12

    #print 'df.columns:', df.columns
    # Index([u'date_block_num', u'item_id', u'shop_id', u'item_cnt_month', u'prev_month_ind'], dtype='object')
    # (1609124, 5)

    my_dict = {}
    for _, data_row in df.iterrows():
        curr_key = (data_row['date_block_num'], data_row['item_id'], data_row['shop_id'])
        my_dict[curr_key] = data_row['item_cnt_month']

    prev_month_cnts, prev_year_cnts = [], []
    for _, data_row in df.iterrows():
        curr_key = (data_row['prev_month_ind'], data_row['item_id'], data_row['shop_id'])
        if my_dict.has_key(curr_key):
            res = np.clip(my_dict[curr_key], 0, 20)
            prev_month_cnts.append(res)
        else:
            prev_month_cnts.append(0)

        curr_key = (data_row['prev_year_ind'], data_row['item_id'], data_row['shop_id'])
        if my_dict.has_key(curr_key):
            res = np.clip(my_dict[curr_key], 0, 20)
            prev_year_cnts.append(res)
        else:
            prev_year_cnts.append(0)

    df['prev_month_cnt'] = pd.Series(prev_month_cnts, index=df.index)
    df['prev_year_cnt'] = pd.Series(prev_year_cnts, index=df.index)

    df = df[['date_block_num', 'prev_month_cnt', 'prev_year_cnt', 'item_cnt_month', 'ID']]
    df.to_pickle("tmp.pkl")

print 'df.shape:', df.shape

#
#train set size: (1609124, 4)
#test set size: (214200, 4)
#print 'train set size:', df.loc[df['ID']<0].shape
#print 'test set size:', df.loc[df['ID']>=0].shape

to_use_model = False

if to_use_model:
# Split
    X_train, X_valid, y_train, y_valid = train_test_split(
            df.loc[df['ID']<0, ['prev_month_cnt', 'prev_year_cnt']],
            df.loc[df['ID']<0, 'item_cnt_month'],
            test_size=0.2,
            random_state=42)

    # Models
    pred_prevMonth = X_valid['prev_month_cnt']
    pred_prevYear = X_valid['prev_year_cnt']

    regr = GradientBoostingRegressor()
    #regr = tree.DecisionTreeRegressor(random_state=0)
    regr.fit(X_train, y_train)
    pred_prevMonthYear = regr.predict(X_valid)

    #MSE(y_valid, X_valid): 72.4624819448
    print 'MSE(y_valid, pred_prevMonth):', MSE(y_valid, pred_prevMonth)
    print 'MSE(y_valid, pred_prevYear):', MSE(y_valid, pred_prevYear)
    print 'MSE(y_valid, pred_prevMonthYear):', MSE(y_valid, pred_prevMonthYear)

    pred_test = regr.predict(df.loc[df['ID']>=0, ['prev_month_cnt', 'prev_year_cnt']])

    offset = pred_test[1]
    print offset

    #pd.Series(pred_test, index=test_df.index) - offset
    #pd.Series(df.loc[df['ID']>=0, 'prev_month_cnt'].values, index=test_df.index)

else:
    print df.tail(10)
    pred_test = df.loc[(df['date_block_num']==33.), 'item_cnt_month'].values * 3
    pred_test += df.loc[(df['date_block_num']==22.), 'item_cnt_month'].values
    pred_test += df.loc[(df['date_block_num']==10.), 'item_cnt_month'].values
    pred_test /= 5.

test_df = pd.read_csv(data_folder + 'test.csv')

#test_df['item_cnt_month'] = pd.Series((df.loc[df['ID']>=0, 'prev_month_cnt'].values+df.loc[df['ID']>=0, 'prev_year_cnt'].values)/2., index=test_df.index)
test_df['item_cnt_month'] = pd.Series(pred_test, index=test_df.index) - offset
test_df[['ID','item_cnt_month']].to_csv("avg_benchmark.csv", index=False)