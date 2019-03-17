
import utils
import pandas as pd
pd.set_option('display.width', 500)
data_folder = 'data/'
#

# 1. FE like online
file_train = data_folder + 'sales_train.csv'
file_test = data_folder + 'test.csv'
train_valid_data, test_df = utils.load_FE_data(file_train, file_test)
lag_features = utils.get_lag_feature()

print train_valid_data.head
print test_df.head
