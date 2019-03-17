
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import my_funcs as mf

import operator

train = pd.read_csv('small_train.csv')
print train[train['class']=='DATE']
quit()

X_train, X_test, y_train, y_test = \
    train_test_split(train['before'].values, train['after'].values,
                     test_size=0.2, random_state=42)

transformed_test = (X_test != y_test)

dict_name = 'en_dict'
my_dict = mf.get_dict_fromXY(X_train, y_train, dict_name)

def is_year(year_str):
    return len(year_str)==4 and year_str.isdigit()

#def handle_year(year_str):
#    res = []
#    if year_str[:2]=='19'
#        res += 'nineteen'

## Look at how test is predicted
all_stats = []
# most scores (sth or None),
# 2nd-most scores (sth or None),
# which one is True (most, 2nd, ...,  or None)
for i in range(len(X_test)):
    if my_dict.has_key(X_test[i]):
        all_candi_s = my_dict[X_test[i]]
        all_candi_s_sorted = sorted(all_candi_s, key=operator.itemgetter(1), reverse=True)#[0][0]



