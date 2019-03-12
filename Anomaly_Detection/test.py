
'''
Won't work that easy!
PCA-detected feature do not necessarily capture predictive features for y
Essentially, key info is missing from EMR for HIV, Pregnancy tests
'''

import pandas as pd
pd.set_option('display.width', 1000)
from medinfo.dataconversion.FeatureMatrixIO import FeatureMatrixIO

fm_io = FeatureMatrixIO()
df = fm_io.read_file_to_data_frame('LABHIVWBL-normality-matrix-processed.tab')

print df.shape

y = df.pop('all_components_normal')
X = df.copy().values


from sklearn import preprocessing, decomposition
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)


mask_1s = (y==1.).values
mask_0s = (y==0.).values



df_scaled = pd.DataFrame(X_scaled)

pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print pca.explained_variance_ratio_

features_scores = zip(df.columns.values.tolist(), pca.components_[0])
print sorted(features_scores, key=lambda x:x[1])[::-1]

print X_pca.shape

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


plt.scatter(X_pca[mask_1s,0], X_pca[mask_1s,1], color='b')
plt.scatter(X_pca[mask_0s,0], X_pca[mask_0s,1], color='orange')
plt.show()
