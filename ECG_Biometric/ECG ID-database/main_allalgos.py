


import numpy as np
import glob
import re
import wfdb
from biosppy.signals import ecg

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
#import xgboost as xgb
import csv
import my_funcs as mf
from mpl_toolkits.mplot3d import Axes3D

from math import exp, log

from scipy.spatial.distance import euclidean

from sklearn import svm

# Sampling freq: 0.002s = 2ms
# Segment period: 256ms (determined by QRS width), so there are 256/2 = 128 datapoints
# Num Persons: 90
# Num recs: 310
# Num segs: 780

num_persons = 90

Person_IDs = range(1,num_persons+1)
SEG_LEN_IN_SEC = 1 # PQRST segment length, unit: s

#TO_PLOT_AGING = False
#if TO_PLOT_AGING:
#    FIRST_TIME = True

labels = []
records = []
#segs_allqual = []
for i in range(num_persons):
    curr_ID_str = str(i+1)
    if i+1 < 10:
        curr_ID_str = '0' + curr_ID_str
    curr_foldername = 'Person_' + curr_ID_str

    curr_filenames =  glob.glob(curr_foldername + "/*.dat")

    #j = 0
    for one_filename in curr_filenames: # One rec has many pulses
        filename = one_filename[:-4]

        sig, fields = wfdb.rdsamp(filename)#, sampto=1000) #, pbdl=0)
        #print fields
        fs = fields['fs']
        records.append(sig[:,1])
        labels.append(i)

X_all = []
y_all = []
num_recs = len(records)
for i in range(num_recs):
    dt = 1./fs
    seg_len = SEG_LEN_IN_SEC / (1./fs) # unit: num of bins
    #print seg_len

    out = ecg.ecg(signal=records[i], sampling_rate=fs, show=False)
    ecg_filtered, peaks = out[1], out[2] ## TODO: get peaks by ecg.ecg

    segs = mf.get_segs(ecg_filtered, peaks, seg_len, fs=fs)
    for seg in segs:
        X_all.append(seg)
        y_all.append(labels[i])

X_all, y_all = np.array(X_all), np.array(y_all)

X_train,X_test,y_train,y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42)
print X_train.shape, y_train.shape
input_dim = X_train.shape[1]

Y_train = np_utils.to_categorical(y_train, num_persons)
Y_test = np_utils.to_categorical(y_test, num_persons)

methods = ['DL','SVC']
method = 'SVC'

## start process the classification

if method == 'DL':
    print 'Method: DL'

    model = Sequential()
    model.add(Dense(input_dim,activation='relu',input_shape=(input_dim,)))
    #model.add(Dense(input_dim,activation='relu'))
    model.add(Dense(num_persons,activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_split=0.2,
              batch_size=32, nb_epoch=20, verbose=1)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print score
elif method == 'SVC':
    print 'Method: SVC'
    clf = svm.SVC(kernel='rbf', C=10., gamma=0.1)
    # ["linear", "poly", "rbf", "sigmoid", "precomputed"]

    clf.fit(X_train, y_train)
    res = clf.predict(X_test)
    print res
    print y_test
    corr_num = sum([res[i]==y_test[i] for i in range(len(res))])
    print float(corr_num)/len(res)

    plt.scatter(res,y_test)
    plt.show()

