

import numpy as np

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input

import my_funcs as mf
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNC
from xgboost import XGBClassifier as XGBC

from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB as GNB

import pywt, csv

#CHANNEL_IND = channels[ind_ds] #
data_set = 'ecgiddb' # 'ecgiddb', 'mitdb'
channel = 1
num_persons = 10
records, labels, fss = mf.load_data(data_set, channel, num_persons=num_persons)

USE_BIOSPPY_FILTERED = True
segs_all, labels_all = mf.get_seg_data(records, labels, fss, USE_BIOSPPY_FILTERED)

print 'method_feat: ' + 'orig' + ', method_clsfy: ' + 'DL'

X_all, y_all = np.array(segs_all), np.array(labels_all)


feat_dim = X_all.shape[1]

X_train,X_test,y_train,y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42)
#print X_train.shape, y_train.shape

num_categs = len(set(y_train))
Y_train = np_utils.to_categorical(y_train, num_categs)
Y_test = np_utils.to_categorical(y_test, num_categs)

## start process the classification

model = Sequential()
model.add(Dense(feat_dim,activation='relu',input_shape=(feat_dim,)))
#model.add(Dense(input_dim,activation='relu'))
model.add(Dense(num_categs,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train, validation_split=0.2,
          batch_size=32, nb_epoch=50, verbose=0)
#score = model.evaluate(X_test, Y_test, verbose=0)
res_pred = model.predict_classes(X_test)
print ''

#fig = plt.figure(figsize=(8,6))
#print zip(y_test, res_pred)

stat_matrix = np.zeros((10,10))
all_categs = range(num_persons)
i = 0
for one_categ in all_categs:
    one_preds = res_pred[y_test == one_categ]
    cnts_pred = np.bincount(one_preds)
    stat_matrix[i, :len(cnts_pred)] = cnts_pred
    #print cnts_pred
    i += 1
print stat_matrix

res_by_seg = mf.get_corr_ratio(res_pred=res_pred, y_test=y_test, type='by_seg')
res_by_categ = mf.get_corr_ratio(res_pred=res_pred, y_test=y_test, type='by_categ')
print res_by_seg, res_by_categ

with open('demo_classifier_ress.txt','w') as fw:
    writer = csv.writer(fw,delimiter='\t')
    for one_row in stat_matrix:
        writer.writerow(one_row)
