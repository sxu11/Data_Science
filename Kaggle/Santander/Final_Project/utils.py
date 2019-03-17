
import numpy as np
import pandas as pd
from itertools import product

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

lags = [1,2,3,12]
target_names = ['target','target_item','target_shop']

def get_lag_feature():
    lag_features = []

    for lag in lags:
        curr_lag_names = [_+'_lag_'+str(lag) for _ in target_names]
        lag_features = lag_features + curr_lag_names

    return lag_features


folder_pkl = 'pkls/'
def load_FE_data(file_train, file_test):
    ind_cols = ['shop_id','item_id','bloc_num']

    use_alldata_pickle = True

    if not use_alldata_pickle:
        # 1. Drop useless cols

        kk = pd.read_csv(file_train)
        kk = kk.rename(columns={'date_block_num':'bloc_num', 'item_cnt_day':'target'})
        kk = kk.drop(['date', 'item_price'],axis=1)

        test_df = pd.read_csv(file_test)
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



        # 4. lags




        print all_data.shape
        cnt_label = 0
        dict_feature_to_label = {}
        absdiff_features = []
        for lag in lags:
            curr_lag_names = [_+'_lag_'+str(lag) for _ in target_names]

            #print ind_cols + ['target','target_item','target_shop']
            cur_lag = all_data[ind_cols + target_names].copy()
            cur_lag['bloc_num'] = cur_lag['bloc_num'] + lag # TODO: wtf?!


            for i in range(len(target_names)) :
                cur_lag = cur_lag.rename(columns={target_names[i]:curr_lag_names[i]})

            all_data = pd.merge(all_data, cur_lag, on=ind_cols, how='left').fillna(0)




            absdiff_features.append('absdiff_lag_'+str(lag))
            dict_feature_to_label['absdiff_lag_'+str(lag)] = cnt_label
            cnt_label += 1

            absdiff_features.append('absdiff_item_lag_'+str(lag))
            dict_feature_to_label['absdiff_item_lag_'+str(lag)] = cnt_label
            cnt_label += 1

            absdiff_features.append('absdiff_shop_lag_'+str(lag))
            dict_feature_to_label['absdiff_shop_lag_'+str(lag)] = cnt_label
            cnt_label += 1

            all_data['absdiff_lag_'+str(lag)] = (all_data['target'] - all_data['target_lag_' + str(lag)]).abs()
            all_data['absdiff_item_lag_'+str(lag)] = (all_data['target'] - all_data['target_item_lag_' + str(lag)]).abs()
            all_data['absdiff_shop_lag_'+str(lag)] = (all_data['target'] - all_data['target_shop_lag_' + str(lag)]).abs()


        all_data = all_data[all_data['bloc_num']>=12]

        argmin_absdiff_features = all_data[absdiff_features].idxmin(axis=1)

        all_data['label'] = argmin_absdiff_features.apply(lambda x:dict_feature_to_label[x])
        all_data = downcast_dtypes(all_data)

        all_data[all_data['ID']<0].to_pickle('trainval_labeled.pkl')
        all_data[all_data['ID']>=0].to_pickle('test_labeled.pkl')

        return all_data[all_data['ID']<0], all_data[all_data['ID']>=0]

        # all_data.to_pickle('all_data_after12mo.pkl')

    else:
        return pd.read_pickle(folder_pkl + 'trainval_labeled.pkl'), \
               pd.read_pickle(folder_pkl + 'test_labeled.pkl')
