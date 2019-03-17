

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
records, labels, fss = mf.load_data(data_set, channel, num_persons=30, record_time=20)

USE_BIOSPPY_FILTERED = True
segs_all, labels_all = mf.get_seg_data(records, labels, fss, USE_BIOSPPY_FILTERED)

methods_feat = ['DL']
#ind_fts = 0
#method_feat = 'orig' #method_feats[ind_fts]

methods_clsf = ['Logit', 'GNB', 'kNN', 'SVC', 'boosting', 'DL']
#ind_cls = 0
#method_clsf = 'Logit' #methods_clsfy[ind_cls]

num_feats = len(methods_feat)
num_clsfy = len(methods_clsf)


X_all, y_all = np.array(segs_all), np.array(labels_all)

#print X_all.shape, y_all.shape # (3102, 250) (3102,)

            #print X_all.shape
input_dim = X_all.shape[1]
#model = Sequential()
#model.add(Dense(feat_dim,activation='relu',input_shape=(feat_dim,)))
#model.add(Dense(feat_dim,activation='sigmoid'))
#print segs_all.shape
amp_max, amp_min = np.max(X_all), np.min(X_all)

#print X_all.shape
X_all = X_all.reshape((len(X_all), np.prod(X_all.shape[1:])))
#print X_all.shape

#print X_all.shape
X_all = X_all.astype('float32')
X_all = (X_all-amp_min)/(amp_max-amp_min)

encoding_dim = 1

input_sig = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_sig)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input=input_sig, output=decoded)

encoder = Model(input=input_sig, output=encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mse')
## TODO: options for loss:
# mse (mean_squared_error)
# mae (mean_absolute_error)
# mape (mean_absolute_percentage_error)
# msle (mean_squared_logarithmic_error)
# squared_hinge
# hinge
# binary_crossentropy
# categorical_crossentropy
# sparse_categorical_crossentropy
# kld (kullback_leibler_divergence)
# poisson
# cosine_proximity
autoencoder.fit(X_all, X_all, nb_epoch=10,
                batch_size=32, shuffle=True, validation_split=0.2)

test_eg = np.array([X_all[0]])
encoded_eg = encoder.predict(test_eg)
decoded_eg = decoder.predict(encoded_eg)

plt.plot(test_eg[0],'g')
plt.plot(decoded_eg[0],'r')
plt.show()
#quit()


