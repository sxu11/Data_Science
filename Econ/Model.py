

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

class Evaluator:
    def __init__(self):
        pass

    def evalBySign(self, y_pred, y_true):
        """the 1st record doesn't count"""
        score = 0
        for i in range(1, len(y_pred)):
            diff_pred = y_pred[i] - y_pred[i-1]
            diff_true = y_true[i] - y_true[i-1]
            score += (diff_pred * diff_true) > 0
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


yPre = pd.read_csv(os.path.join(intermediate_folder, "SP500.csv"))["SP500"].values
y = yPre[1:] - yPre[:-1]
print(y)

xs = []
for filename in filenames[1:]:
    curDs = pd.read_csv(os.path.join(intermediate_folder, filename+".csv"))[filename]
    x = curDs[:-1] #.values[:-1].tolist()
    xs.append(x)
X = np.array(xs).transpose()

print(X.shape, y.shape)
np.savetxt("X.csv", X, delimiter=",")
np.savetxt("y.csv", y, delimiter=",")

reg = LinearRegression().fit(X,y)
print(reg.coef_)


