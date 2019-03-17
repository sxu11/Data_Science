import numpy as np
import pandas as pd
import sklearn

import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)

# import lightgbm as lgb
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from tqdm import tqdm_notebook

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


use_saved_data = False

if not use_saved_data:
    data_folder = 'data/'
    sales = pd.read_csv(data_folder + 'sales_train_small.csv')
    #shops = pd.read_csv('../readonly/final_project_data/shops.csv')
    #items = pd.read_csv('../readonly/final_project_data/items.csv')
    #item_cats = pd.read_csv('../readonly/final_project_data/item_categories.csv')

    # Create "grid" with columns
    index_cols = ['shop_id', 'item_id', 'date_block_num']

    # For every month we create a grid from all shops/items combinations from that month
    grid = []
    for block_num in sales['date_block_num'].unique():
        cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
        cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

    # Turn the grid into a dataframe
    grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)


    # Groupby data to get shop-item-month aggregates
    gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})
    # Fix column names
    gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
    # Join it to the grid
    all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)


    test_df = pd.read_csv(data_folder + 'test.csv')
    test_df['target'] = pd.Series(np.zeros(test_df.shape[0],), index=test_df.index)
    test_df['date_block_num'] = pd.Series(np.ones(test_df.shape[0],)*34, index=test_df.index)

    all_data['ID'] = pd.Series(np.ones(all_data.shape[0],)*-1, index=grid.index)
    all_data = pd.merge(all_data, test_df, how='outer').fillna(0)


    # Same as above but with shop-month aggregates
    gb = sales.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_shop':'sum'}})
    gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
    all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)



    # Same as above but with item-month aggregates
    gb = sales.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_item':'sum'}})
    gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

    # Downcast dtypes from 64 to 32 bit to save memory
    all_data = downcast_dtypes(all_data)
    del grid, gb
    gc.collect()

    # List of columns that we will use to create lags
    cols_to_rename = list(all_data.columns.difference(index_cols))

    shift_range = [1, 2, 3, 12]

    for month_shift in shift_range:
        train_shift = all_data[index_cols + cols_to_rename].copy()

        train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift

        foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
        train_shift = train_shift.rename(columns=foo)

        all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

    del train_shift

    # Don't use old data from year 2013
    all_data = all_data[all_data['date_block_num'] >= 12]

    # List of all lagged features
    fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]]
    # We will drop these at fitting stage
    to_drop_cols = list(set(list(all_data.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num']

    # Category for each item
    # item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()
    #
    # all_data = pd.merge(all_data, item_category_mapping, how='left', on='item_id')
    all_data = downcast_dtypes(all_data)
    gc.collect()

    all_data.to_pickle('internet_all_data.pkl')

else:
    all_data = pd.read_pickle('internet_all_data.pkl')
print all_data.head(5)
print all_data.shape

