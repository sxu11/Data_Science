

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from joblib import dump

class Evaluator:
    def __init__(self):
        pass

    def evalBySign(self, y_pred, y_true):
        """the 1st record doesn't count"""
        score = 0
        for i in range(1, len(y_pred)):
            diff_pred = y_pred[i] - y_pred[i-1]
            diff_true = y_true[i] - y_true[i-1]
            score += (diff_pred * diff_true) >= 0
        return score / float(len(y_pred)-1)



"""
x1[i]: cpi[i]
x2[i]: m1[i]
x3[i]: baa[i]
x4[i]: gdp[i]

y[i]: sp[i] - sp[i-1]
"""
filenames = ["SP500","CPALTT01USM657N", "WM1NS", "BAA10Y", "GDP"]
intermediate_folder = "intermediate_data"

trainTestSplitRatio = 0.75

yPre = pd.read_csv(os.path.join(intermediate_folder, "SP500.csv"))["SP500"].values
y = yPre[1:] - yPre[:-1]



xs = []
for filename in filenames[1:]:
    curDs = pd.read_csv(os.path.join(intermediate_folder, filename+".csv"))[filename]
    x = curDs[:-1] #.values[:-1].tolist()
    xs.append(x)
X = np.array(xs).transpose()


# print(X.shape, y.shape)
# np.savetxt("X.csv", X, delimiter=",")
# np.savetxt("y.csv", y, delimiter=",")

X = X[y==y,:]
y = y[y==y]
# print(X.shape, y.shape)

splitInd = int(trainTestSplitRatio * len(y))

y_train, y_test = y[:splitInd], y[splitInd:]
X_train, X_test = X[:splitInd, :], X[splitInd:, :]

reg = LinearRegression().fit(X_train,y_train)
print("reg.coef_: ", reg.coef_)
dump(reg, os.path.join("res_data", "reg.joblib"))

print("R2 score for Train data:", reg.score(X_train, y_train))
print("R2 score for Test data:", reg.score(X_test, y_test))


y_pred = reg.predict(X_train)
eval_score = Evaluator().evalBySign(y_pred, y_train)
print("evalBySign score for Train data:", eval_score)


y_pred = reg.predict(X_test)
eval_score = Evaluator().evalBySign(y_pred, y_test)
print("evalBySign score for Test data:", eval_score)