
import pandas as pd

#all_data = pd.read_pickle('mid_data/all_data_after12mo.pkl')

tmp = pd.read_pickle('all_data_after12mo.pkl') #'tmp.pkl'


lags = [1,2,3,12]
# lag_features = []
absdiff_features = []

dict_feature_to_label = {}

cnt_label = 0
for lag in lags:
    # lag_features += ['target_lag_' + str(lag),
    #                  'target_item_lag' + str(lag),
    #                  'target_shop_lag' + str(lag)]

    absdiff_features.append('absdiff_lag_'+str(lag))
    dict_feature_to_label['absdiff_lag_'+str(lag)] = cnt_label
    cnt_label += 1

    absdiff_features.append('absdiff_item_lag_'+str(lag))
    dict_feature_to_label['absdiff_item_lag_'+str(lag)] = cnt_label
    cnt_label += 1

    absdiff_features.append('absdiff_shop_lag_'+str(lag))
    dict_feature_to_label['absdiff_shop_lag_'+str(lag)] = cnt_label
    cnt_label += 1

    tmp['absdiff_lag_'+str(lag)] = (tmp['target'] - tmp['target_lag_' + str(lag)]).abs()
    tmp['absdiff_item_lag_'+str(lag)] = (tmp['target'] - tmp['target_item_lag' + str(lag)]).abs()
    tmp['absdiff_shop_lag_'+str(lag)] = (tmp['target'] - tmp['target_shop_lag' + str(lag)]).abs()
print 'cnt_label:', cnt_label


argmin_absdiff_features = tmp[absdiff_features].idxmin(axis=1)

tmp['label'] = argmin_absdiff_features.apply(lambda x:dict_feature_to_label[x])

tmp[tmp['ID']<0].to_pickle('trainval_labeled.pkl')
tmp[tmp['ID']>=0].to_pickle('test_labeled.pkl')