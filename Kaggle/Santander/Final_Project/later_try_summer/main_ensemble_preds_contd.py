import pandas as pd
from xgboost import XGBClassifier as XGBC

tmp = pd.read_pickle('trainval_labeled.pkl') #'tmp.pkl'
test_df = pd.read_pickle('test_labeled.pkl')
# print tmp.head()

#print tmp['label'].value_counts()

lags = [1,2,3,12]
lag_features = []
for lag in lags:
    lag_features = lag_features + ['target_lag_' + str(lag),
                     'target_item_lag' + str(lag),
                     'target_shop_lag' + str(lag)]

clf = XGBC()
X_train = tmp[lag_features]
y_train = tmp['label']
X_test = test_df[lag_features]


clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)

print test_df.head()
test_df['label_pred'] = pd.Series(y_test_pred, index=test_df.index)
print test_df.head()

test_df.to_pickle('test_predicted.pkl')