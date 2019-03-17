
import pandas as pd
import numpy as np


tmp = pd.read_pickle('../pkls/test_predicted.pkl')


lags = [1,2,3,12]
lag_features = []
for lag in lags:
    lag_features = lag_features + ['target_lag_' + str(lag),
                     'target_item_lag' + str(lag),
                     'target_shop_lag' + str(lag)]

# tmp['label_pred'] = pd.Series(np.zeros(tmp.shape[0],), index=tmp.index, dtype='int32')

hah = tmp['label_pred'].apply(lambda x: lag_features[x])

res_pred = tmp.lookup(hah.index, hah.values)
tmp['item_cnt_month'] = pd.Series(res_pred, index=tmp.index).clip(0,20)

tmp['ID'] = tmp['ID'].astype('int32')


print tmp['item_cnt_month'].value_counts()
# tmp['item_cnt_month'] = pd.Series(np.zeros(tmp.shape[0],),index=tmp.index)
# for index, test_row in tmp.iterrows():
#     tmp.loc[index,'item_cnt_month'] = test_row[hah.values[index-10913850]]
# print tmp.head()

tmp[['ID', 'item_cnt_month']].to_csv('newsubmission.csv', index=False)