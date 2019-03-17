

import pandas as pd
import numpy as np

data1 = pd.read_csv('ridge_sol.csv')#.as_matrix()
data2 = pd.read_csv('submission.csv')#.as_matrix()

# data3 = np.copy(data1)
# row, col = data3.shape
# for i in range(row):
#     data3[i,1] = (data3[i,1]+data2[i,1])/2.

ha = (data1['SalePrice']+data2['SalePrice'])/2.
print ha

solution = pd.DataFrame({"id":data1['id'], "SalePrice":ha})
solution.to_csv("ensembled.csv", index = False)
