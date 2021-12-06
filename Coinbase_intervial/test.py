import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier

df = pd.read_csv("data/interview_dataset_v1.csv")
print(df.head())

"""
1.
features: 
- ipaddress
- country_code
- eareason: ip score
- email_domain
- idology_alerts
- jumio_: 3rd party data
- num_signin_ips
- return_info

fraud_bad

2. e2e workflow: training, test
- check distribution of f/t
- split


3. optimize model
"""

print(df.shape)
print(df.columns)

print(df["fraud_bad"].sum()) # 6748/ 141698: 5% fraud

# df = df.loc[:1000]

# print(df[""])
# quit()

"""
split data into train/test
"""
# fraudDf = df[df["fraud_bad"]==1]
# goodDf = df[df["fraud_bad"]==0]
# print(fraudDf.shape, goodDf.shape)
y = df["fraud_bad"]
# X = df.loc[:, df.columns != 'fraud_bad']

features = ["buy_count", "buy_volume_btc", "buy_volume_eth", "buy_volume_ltc", "buy_volume_usd",
            "dob_year", "eascore", "jumio_failure_count", "num_signin_ips", "sell_count", "sell_volume_usd"]
X = df[features]
# print(y.shape, X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)

# X_train["weight"] = X_train["fraud_bad"].apply(lambda x: 5 if True else 1)

etas = [0.1, 0.01, 0.3]
max_depths = [3,4,5]
min_child_weight = [1,5,10]
subsamples = [1,0.7,0.5]
colsample_bytree = [1,0.7,0.5]


clf = XGBClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:,1]
print(y_pred, y_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
print(metrics.auc(fpr, tpr))


