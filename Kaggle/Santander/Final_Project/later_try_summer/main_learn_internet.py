


import pandas as pd
from itertools import product
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt

def downcast_dtypes(df):
    '''
        Changes column types in the dataframe:

                `float64` type to `float32`
                `int64`   type to `int32`
    '''

    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]

    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)

    return df

data_folder = 'data/'

ind_cols = ['shop_id','item_id','bloc_num']

use_alldata_pickle = False

if not use_alldata_pickle:
    # 1. Drop useless cols

    kk = pd.read_csv(data_folder + 'sales_train.csv')
    kk = kk.rename(columns={'date_block_num':'bloc_num', 'item_cnt_day':'target'})
    kk = kk.drop(['date', 'item_price'],axis=1)

    test_df = pd.read_csv(data_folder + 'test.csv')
    test_df['target'] = pd.Series(np.zeros(test_df.shape[0],), index=test_df.index)
    test_df['bloc_num'] = pd.Series(np.ones(test_df.shape[0],)*34, index=test_df.index)

    # print test_df.head()

    # 2. Build grid

    grid = []
    for bloc_num in kk['bloc_num'].unique():
        cur_shops = kk[kk['bloc_num']==bloc_num].shop_id.unique()
        cur_items = kk[kk['bloc_num']==bloc_num].item_id.unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [bloc_num]])),dtype='int32'))
    grid = pd.DataFrame(np.vstack(grid), columns=ind_cols, dtype=np.int32) # TODO: order much match!!!



    grid['ID'] = pd.Series(np.ones(grid.shape[0],)*-1, index=grid.index)

    gb = kk.groupby(ind_cols,as_index=False).target.sum()#.clip(0, 20) #TODO: CNM!

    grid = pd.merge(grid, gb, how='left', on=ind_cols).fillna(0)

    # print all_data.shape
    # print all_data.head()
    # quit()


    # 3. Concat df
    all_data = pd.merge(grid, test_df, how='outer')


    #del grid, test_df

    # TODO: as type int32


    # 3.5 FE
    gb_shop = all_data.groupby(['bloc_num','shop_id'],as_index=False).target.sum().fillna(0)
    gb_shop = gb_shop.rename(columns={'target':'target_shop'})

    all_data = pd.merge(all_data, gb_shop, on=['bloc_num','shop_id'], how='left')



    gb_item = all_data.groupby(['bloc_num','item_id'],as_index=False).target.sum().fillna(0)
    gb_item = gb_item.rename(columns={'target':'target_item'})
    all_data = pd.merge(all_data, gb_item, on=['bloc_num','item_id'], how='left')




    # print all_data['target'].value_counts()
    # print all_data['target_item'].value_counts()
    # print all_data['target_shop'].value_counts()
    # quit()

    # 4. lags

    lags = [1,2,3,12]

    print all_data.shape
    for lag in lags:
        #print ind_cols + ['target','target_item','target_shop']
        cur_lag = all_data[ind_cols + ['target','target_item','target_shop']].copy()
        cur_lag['bloc_num'] = cur_lag['bloc_num'] + lag # TODO: wtf?!

        cur_lag = cur_lag.rename(columns={'target':'target_lag_' + str(lag)})
        cur_lag = cur_lag.rename(columns={'target_item':'target_item_lag' + str(lag)})
        cur_lag = cur_lag.rename(columns={'target_shop':'target_shop_lag' + str(lag)})
        # print cur_lag['target_lag_' + str(1)].value_counts()
        # print cur_lag['target_item_lag' + str(1)].value_counts()
        # print cur_lag['target_shop_lag' + str(1)].value_counts()

        #
        # print cur_lag.head()

        # print cur_lag.shape, cur_lag.dtypes
        # print all_data.shape, all_data.dtypes

        all_data = pd.merge(all_data, cur_lag, on=ind_cols, how='left').fillna(0)

        # print all_data['target_lag_' + str(1)].value_counts()
        # print all_data['target_item_lag' + str(1)].value_counts()
        # print all_data['target_shop_lag' + str(1)].value_counts()

        # print all_data.head()
        # print all_data.shape, all_data.dtypes


    print all_data['target_lag_' + str(1)].value_counts()
    print all_data['target_item_lag' + str(1)].value_counts()
    print all_data['target_shop_lag' + str(1)].value_counts()

    # print all_data['target_lag_' + str(12)].value_counts()
    # print all_data['target_item_lag' + str(12)].value_counts()
    # print all_data['target_shop_lag' + str(12)].value_counts()

    #all_data.to_pickle('all_data.pkl')
    all_data = all_data[all_data['bloc_num']>=12]

    all_data = downcast_dtypes(all_data)
    all_data.to_pickle('all_data_after12mo.pkl')

else:
    all_data = pd.read_pickle('all_data_after12mo.pkl')


quit()









# 5.

# ## mean encoder
# mean_encoder = all_data[ind_cols+['target']].groupby(ind_cols,as_index=False).target.mean().fillna(0)
# mean_encoder = mean_encoder.rename(columns={'target':'mean_code'})
# all_data = pd.merge(all_data, mean_encoder, on=ind_cols, how='left')
# all_data.drop(ind_cols,axis=1,inplace=True)

lags = [1,2,3,12]
lag_features = []
for lag in lags:
    lag_features = lag_features + ['target_lag_' + str(lag),
                     'target_item_lag' + str(lag),
                     'target_shop_lag' + str(lag)]

X_train, X_valid, y_train, y_valid = train_test_split(
            all_data.loc[all_data['ID']<0, lag_features],
            all_data.loc[all_data['ID']<0, 'target'],
            test_size=0.2,
            random_state=42)

# X_train.to_pickle('X_train.pkl')
# X_valid.to_pickle('X_valid.pkl')
# y_train.to_pickle('y_train.pkl')
# y_valid.to_pickle('y_valid.pkl')


print MSE(y_valid, np.zeros(y_valid.shape))

# lr = LinearRegression()
# lr.fit(X_train, y_train)
# y_valid_pred = lr.predict(X_valid)

xgdmat = xgboost.DMatrix(X_train,y_train)
our_params = {'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':3,'min_child_weight':1}
final_gb = xgboost.train(our_params,xgdmat)

tesdmat = xgboost.DMatrix(X_valid)
y_valid_pred=final_gb.predict(tesdmat)

print MSE(y_valid, y_valid_pred)


# 6.predict
test_df = pd.read_csv(data_folder + 'test.csv')
X_test = all_data.loc[all_data['ID']>=0, lag_features]
tesdmat = xgboost.DMatrix(X_test)
y_test_pred = final_gb.predict(tesdmat)
#y_test = lr.predict()
del all_data

test_df['item_cnt_month'] = pd.Series(y_test_pred, index=test_df.index)
test_df[['ID', 'item_cnt_month']].to_csv("new_submission.csv", index=False)

#mean_encoder_test = test_df[ind_cols].groupby(ind_cols,as_index=False).target.mean().fillna(0)