


import pandas as pd
from itertools import product
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE

data_folder = 'data/'

get_traintestdata_from_pickle = True
ind_cols = ['bloc_num','shop_id','item_id']

if not get_traintestdata_from_pickle:
    kk = pd.read_csv(data_folder + 'data/sales_train.csv')
    print kk.shape
    kk = kk.rename(columns={'date_block_num':'bloc_num', 'item_cnt_day':'target'})


    get_alldata_from_pickle = True



    if not get_alldata_from_pickle:
    # if not get_from_pickle:
    #     grid = []
    #     for bloc_num in kk['bloc_num'].unique():
    #         cur_shops = kk[kk['bloc_num']==bloc_num].shop_id.unique()
    #         cur_items = kk[kk['bloc_num']==bloc_num].item_id.unique()
    #         grid.append(list(product(*[cur_shops, cur_items, [bloc_num]])))
    #     grid = pd.DataFrame(np.vstack(grid), columns=ind_cols)
    #     grid.to_pickle('prid.pkl')
    # else:
        grid = pd.read_pickle('prid.pkl')

        kk.drop('item_price',axis=1,inplace=True)

        gb = kk.groupby(ind_cols,as_index=False).sum()
        all_data = pd.merge(grid, gb, on=ind_cols, how='left').fillna(0)

        gb_item = kk.groupby(['bloc_num','item_id'],as_index=False).target.sum().fillna(0)
        gb_item = gb_item.rename(columns={'target':'target_item'})
        all_data = pd.merge(all_data, gb_item, on=['bloc_num','item_id'], how='left')

        gb_shop = kk.groupby(['bloc_num','shop_id'],as_index=False).target.sum().fillna(0)
        gb_shop = gb_shop.rename(columns={'target':'target_shop'})
        all_data = pd.merge(all_data, gb_shop, on=['bloc_num','shop_id'], how='left')

        lags = [1,2,3,12]

        print all_data.shape
        for lag in lags:
            #print ind_cols + ['target','target_item','target_shop']
            #kk_lag = all_data[ind_cols + ['target','target_item','target_shop']].copy()
            kk_lag = all_data[ind_cols + ['target','target_item','target_shop']].copy()
            kk_lag['bloc_num'] -= lag
            kk_lag = kk_lag.rename(columns={'target':'target_lag_' + str(lag)})
            kk_lag = kk_lag.rename(columns={'target_item':'target_item_lag' + str(lag)})
            kk_lag = kk_lag.rename(columns={'target_shop':'target_shop_lag' + str(lag)})

            all_data = pd.merge(all_data, kk_lag, on=ind_cols, how='left').fillna(0)

        all_data.to_pickle('all_data.pkl')
    else:
        all_data = pd.read_pickle('all_data.pkl')

    print 'all_data.shape:', all_data.shape

    ## mean encoder
    mean_encoder = all_data[ind_cols+['target']].groupby(ind_cols,as_index=False).target.mean().fillna(0)
    mean_encoder = mean_encoder.rename(columns={'target':'mean_code'})
    # print 'mean_encoder.shape:', mean_encoder.shape
    # print mean_encoder.head()
    all_data = pd.merge(all_data, mean_encoder, on=ind_cols, how='left')
    all_data.drop(ind_cols,axis=1,inplace=True)

    all_data.to_pickle('all_data_mean_code_no_inds.pkl')
    print 'all_data.shape:', all_data.shape
    print all_data.head()

    X_train, X_valid, y_train, y_valid = train_test_split(
                all_data.loc[:, all_data.columns != 'target'],
                all_data['target'],
                test_size=0.2,
                random_state=42)

    X_train.to_pickle('X_train.pkl')
    X_valid.to_pickle('X_valid.pkl')
    y_train.to_pickle('y_train.pkl')
    y_valid.to_pickle('y_valid.pkl')

else:

    X_train = pd.read_pickle('X_train.pkl')
    X_valid = pd.read_pickle('X_valid.pkl')
    y_train = pd.read_pickle('y_train.pkl')
    y_valid = pd.read_pickle('y_valid.pkl')

print MSE(y_valid, np.zeros(y_valid.shape))

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_valid)
print MSE(y_valid, y_pred)

test_df = pd.read_csv(data_folder + 'test.csv')
print test_df.head()
#mean_encoder_test = test_df[ind_cols].groupby(ind_cols,as_index=False).target.mean().fillna(0)