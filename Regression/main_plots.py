

import numpy as np
import pandas as pd
import os
from sklearn import linear_model

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns

data_folderpath = 'house_data'

train_df = pd.read_csv(os.path.join(data_folderpath, 'kc_house_train_data.csv'))
print train_df.head()

X=train_df['sqft_living'].values
y=train_df['price'].values

# regr = linear_model.LinearRegression()
# regr.fit(X, y)
# print regr.intercept_, regr.coef_

sns.residplot(X, y, lowess=True)

plt.xlabel('sqft_living')
plt.ylabel('price')
plt.savefig('figures/residual_plot.png')