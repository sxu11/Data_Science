

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

methods_feat = ['PCA'] #['orig', 'PCA', 'fiducial', 'wavelet']#, 'DL']
#ind_fts = 0
#method_feat = 'orig' #method_feats[ind_fts]

methods_clsf = ['DL'] #['Logit', 'GNB', 'kNN', 'SVC', 'boosting', 'DL']
#ind_cls = 0
#method_clsf = 'Logit' #methods_clsfy[ind_cls]

num_feats = len(methods_feat)
num_clsfy = len(methods_clsf)

all_ress = []
for ind_cls in range(num_clsfy):
    one_ress = []
    for ind_fts in range(num_feats):
        method_feat = methods_feat[ind_fts]
        method_clsf = methods_clsf[ind_cls]
        print 'method_feat: ' + method_feat + ', method_clsfy: ' + method_clsf

        X_all, y_all = np.array(segs_all), np.array(labels_all)


        if method_feat == 'orig':
            #X_all, y_all = np.array(segs_all), np.array(labels_all)
            feat_dim = X_all.shape[1]
        elif method_feat == 'PCA':
            n_compon = 30

            feat_dim = n_compon
            pca = PCA(n_components=n_compon)
            X_all = pca.fit(X_all).transform(X_all)
            X_all = np.array(X_all)

        elif method_feat == 'fiducial':
            all_feats = []
            y_all_new = []
            i = 0
            for one_X in X_all:
                one_feats = []
                # first: locate P, Q, S, T
                #plt.plot(one_X)
                #plt.show()

                R_ind = np.argmax(one_X)
                if R_ind <= 30:
                    continue
                Q_ind = np.argmin(one_X[R_ind-30:R_ind]) + R_ind-30 #TODO: empirical
                P_ind = np.argmax(one_X[:Q_ind])
                S_ind = np.argmin(one_X[R_ind:])
                T_ind = np.argmax(one_X[S_ind:])

                # calculate features
                # 0: RP amp
                one_feats.append(abs(one_X[P_ind]-one_X[R_ind]))
                # 1: RQ amp
                one_feats.append(abs(one_X[Q_ind]-one_X[R_ind]))
                # 2: RS amp
                one_feats.append(abs(one_X[S_ind]-one_X[R_ind]))
                # 3: RT amp
                one_feats.append(abs(one_X[R_ind]-one_X[R_ind]))

                # 4: RS interv, todo: dt adjust?
                one_feats.append(abs(S_ind-R_ind))
                # 5: RQ interv
                one_feats.append(abs(Q_ind-R_ind))
                # 6: RP interv (todo: heart rate adjust)
                one_feats.append(abs(P_ind-R_ind))
                # 7: RT interv
                one_feats.append(abs(T_ind-R_ind))

                y_all_new.append(y_all[i])
                i += 1

                # todo: feature normalization?

                # 8: S angle

                # 9: Q angle

                # 10: R angle
                all_feats.append(one_feats)
            X_all = np.array(all_feats)
            y_all = np.array(y_all_new)
            feat_dim = X_all.shape[-1]
            #print X_all.shape, y_all.shape,

        elif method_feat == 'wavelet':
            lvl = 4
            wav_coefs = []
            for one_X in X_all:
                curr_coefs = pywt.wavedec(one_X, 'db3', level=lvl)
                wav_coefs.append(curr_coefs[0]) # cA
            X_all = np.array(wav_coefs)
            feat_dim = X_all.shape[1]

            #print X_all.shape
        elif method_feat == 'DL': # converging too slow, aborted
            feat_dim = X_all.shape[1]
            #model = Sequential()
            #model.add(Dense(feat_dim,activation='relu',input_shape=(feat_dim,)))
            #model.add(Dense(feat_dim,activation='sigmoid'))
            #print segs_all.shape
            amp_max, amp_min = np.max(X_all), np.min(X_all)

            X_all = X_all.reshape((len(X_all), np.prod(X_all.shape[1:])))
            #print X_all.shape
            X_all = X_all.astype('float32')
            X_all = (X_all-amp_min)/(amp_max-amp_min)

            encoding_dim = 125

            input_sig = Input(shape=(feat_dim,))
            encoded = Dense(encoding_dim, activation='relu')(input_sig)
            decoded = Dense(feat_dim, activation='sigmoid')(encoded)
            autoencoder = Model(input=input_sig, output=decoded)

            encoder = Model(input=input_sig, output=encoded)
            encoded_input = Input(shape=(encoding_dim,))
            decoder_layer = autoencoder.layers[-1]
            decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

            autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
            autoencoder.fit(X_all, X_all, nb_epoch=50,
                            batch_size=32, shuffle=True, validation_split=0.2)
            ha = np.array(decoder_layer.get_weights()[1])
            #print ha.shape # output
            #quit()
        else:
            X_all, y_all = None, None

        X_train,X_test,y_train,y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42)
        #print X_train.shape, y_train.shape

        num_categs = len(set(y_train))
        Y_train = np_utils.to_categorical(y_train, num_categs)
        Y_test = np_utils.to_categorical(y_test, num_categs)

        ## start process the classification

        if method_clsf == 'DL':
            #print 'Method: DL'

            model = Sequential()
            model.add(Dense(feat_dim,activation='relu',input_shape=(feat_dim,)))
            #model.add(Dense(input_dim,activation='relu'))
            model.add(Dense(num_categs,activation='softmax'))

            print X_train.shape
            print Y_train.shape
            quit()

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            model.fit(X_train, Y_train, validation_split=0.2,
                      batch_size=32, nb_epoch=50, verbose=0)
            #score = model.evaluate(X_test, Y_test, verbose=0)
            res_pred = model.predict_classes(X_test)
        else:
            if method_clsf == 'SVC':
                #print 'Method: SVC'
                clf = svm.SVC(kernel='rbf', C=10., gamma=0.1)

                # ["linear", "poly", "rbf", "sigmoid", "precomputed"]
                #print dfs[0], len(dfs), len(X_test)
                #for i in range(len(X_test)):
                #    print np.argmax(dfs[i]), res_pred[i]
            elif method_clsf == 'Logit':
                clf = LR(C=10.)
            elif method_clsf == 'kNN':
                clf = KNC()
            elif method_clsf == 'boosting':
                clf = XGBC()
            elif method_clsf == 'GNB':
                clf = GNB()
            else:
                clf = None

            clf.fit(X_train, y_train)
            res_pred = clf.predict(X_test)

            #dfs = clf.decision_function(X_test)

        res_by_seg = mf.get_corr_ratio(res_pred=res_pred, y_test=y_test, type='by_seg')
        res_by_categ = mf.get_corr_ratio(res_pred=res_pred, y_test=y_test, type='by_categ')
        one_res = (float(format(res_by_seg, '.3f')), float(format(res_by_categ, '.3f')))
        one_ress.append(one_res)
    all_ress.append(one_ress)

with open('all_ress.txt','w') as fw:
    writer = csv.writer(fw,delimiter='\t')
    for one_ress in all_ress:
        writer.writerow(one_ress)
