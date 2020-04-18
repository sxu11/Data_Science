"""
https://www.youtube.com/watch?v=jA7FY4itT6s&list=PLlH6o4fAIji6FEsjFeo7gRgiwhPUkJ4ap&index=15
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/50_Startups.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

X[:,3] = LabelEncoder().fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regression = LinearRegression()
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)
r2_score(y_test, y_pred)

df = pd.DataFrame(data=y_test, columns=["y_test"])
df["y_pred"] = y_pred

from sklearn.externals import joblib
joblib.dump(regression, "output/multiple_linear_model.pkl")