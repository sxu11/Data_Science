

from scipy.optimize import curve_fit

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from math import pi as PI

from sklearn.cross_validation import train_test_split
from sklearn import svm

import my_funcs as mf
import wfdb

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.utils import np_utils

from collections import Counter

from sklearn.decomposition import PCA
import random

class Seg:
    def __init__(self, sig, label):
        self.sig = sig
        self.label = label
class Rec:
    def __init__(self, segs, ID, anns=None):
        self.segs = segs
        self.ID = ID
        self.anns = anns

num_Gaussians = 6

def multiGaussFunc(x, a1,b1,t1,a2,b2,t2,a3,b3,t3,a4,b4,t4,a5,b5,t5,a6,b6,t6):
    # TODO may need to be divmod(x-t1,PI)[1]

    return a1*exp(-((x-t1)/b1)**2./2.) \
           + a2*exp(-((x-t2)/b2)**2./2.) \
           + a3*exp(-((x-t3)/b3)**2./2.) \
           + a4*exp(-((x-t4)/b4)**2./2.) \
           + a5*exp(-((x-t5)/b5)**2./2.) \
           + a6*exp(-((x-t6)/b6)**2./2.)
    #return a1*exp(-(divmod(x-t1,PI)[1]/b1)**2./2.) \
    #       + a2*exp(-(divmod(x-t2,PI)[1]/b2)**2./2.) \
    #       + a3*exp(-(divmod(x-t3,PI)[1]/b3)**2./2.) \
    #       + a4*exp(-(divmod(x-t4,PI)[1]/b4)**2./2.) \
    #       + a5*exp(-(divmod(x-t5,PI)[1]/b5)**2./2.) \
    #       + a6*exp(-(divmod(x-t6,PI)[1]/b6)**2./2.)

    #for i in range(num_Gaussians):
    #    del_tht = t_s[i] - x
    #    z_res += 2 * a_s[i] * del_tht * exp(-(del_tht/b_s[i])**2./2.)

data_set = 'mitdb' # 'ecgiddb', 'mitdb'
channel = 0
records, labels, fss, annss = mf.load_data(data_set, channel)#, num_persons=60, record_time=20)
fs = fss[0] # 500. # 500 cycles/second

records = np.array(records)
labels = np.array(labels)
annss = np.array(annss)
NV_inds = [6,15,18,23,24,26,29,31,33,35,39,41,42,46]
#for i in NV_inds: #range(annss.shape[0]): #
#    print i, Counter(annss[i][1])['V']

constrain_time = True
if constrain_time:
    look_time = 30. # in s
    look_ind = int(look_time * fs)

    records = records[NV_inds, :look_ind]
    labels = labels[NV_inds]
    annss = annss[NV_inds, :look_ind]
else:
    records = records[NV_inds, :]
    labels = labels[NV_inds]
    annss = annss[NV_inds, :]


USE_BIOSPPY_FILTERED = True
segs_all, labels_all = mf.get_seg_data(records, labels, fss, USE_BIOSPPY_FILTERED
                                       , annss=annss)
#for one_label in labels_all:
#    if ('N' in one_label) or ('V' in one_label):
#        print one_label
#quit()

segs_all, labels_all = np.array(segs_all), np.array(labels_all)

X_all = []
y_all = []
method_feat = 'PCA' # 'template_matching'

if method_feat == 'template_matching':
    dt = 1. #1./fs
    ts = np.linspace(0,len(segs_all[0])*dt,num=len(segs_all[0]))

    #print len(segs_all[0])
    co = 1./250 * len(segs_all[0])

    t1_lower_bnd = 18.*co*dt
    t1_upper_bnd = 21.*co*dt
    t2_lower_bnd = 75.*co*dt
    t2_upper_bnd = 78.*co*dt
    t3_lower_bnd = 89.*co*dt
    t3_upper_bnd = 92.*co*dt
    t4_lower_bnd = 100.*co*dt
    t4_upper_bnd = 103.*co*dt
    t5_lower_bnd = 160.*co*dt
    t5_upper_bnd = 250.*co*dt
    t6_lower_bnd = 220.*co*dt
    t6_upper_bnd = 250*co*dt

    p0 = [.12, PI/16, -PI/2,   #P
          -.25, PI/16, -PI/16,   #Q
          1.2, PI/16, 0,   #R
          -.25, PI/16, PI/16,   #S
          .12, PI/16, PI/3,   #T
          .12, PI/16, PI*2/3   #U
          ]
    for i in range(len(segs_all)):
        seg_curr = segs_all[i]
        #plt.plot(ts,seg_curr)
        #plt.show()

        #print seg_curr
        #print annss[i][0]
        #quit()

        #xs = np.linspace(0,250,num=len(seg_curr))
        try:
            popt, pcov = curve_fit(multiGaussFunc, ts, seg_curr,
                               bounds=([-np.inf,-np.inf,t1_lower_bnd,
                                       -np.inf,-np.inf,t2_lower_bnd,
                                       -np.inf,-np.inf,t3_lower_bnd,
                                       -np.inf,-np.inf,t4_lower_bnd,
                                       -np.inf,-np.inf,t5_lower_bnd,
                                       -np.inf,-np.inf,t6_lower_bnd],
                                       [np.inf,np.inf,t1_upper_bnd,
                                       np.inf,np.inf,t2_upper_bnd,
                                       np.inf,np.inf,t3_upper_bnd,
                                       np.inf,np.inf,t4_upper_bnd,
                                       np.inf,np.inf,t5_upper_bnd,
                                       np.inf,np.inf,t6_upper_bnd])
                               #p0=p0
                               ) # Parameter extraction

            seg_fitted = multiGaussFunc(ts, popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],
                      popt[6],popt[7],popt[8],popt[9],popt[10],popt[11],
                      popt[12],popt[13],popt[14],popt[15],popt[16],popt[17])
            # Feature extraction

            X_all.append(seg_fitted)

            curr_label = str(labels_all[i]) #TODO
            y_all.append(curr_label)
        except:
            pass

elif method_feat == 'PCA':
    feat_dim = 20
    pca = PCA(n_components=feat_dim)
    X_all = np.array(segs_all)
    X_all = pca.fit(X_all).transform(X_all)
    y_all = np.array(labels_all)

X_all = np.array(X_all)

## TODO: data selection
do_data_selection = False
if not do_data_selection:
    X_train,X_test,y_train,y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42)
else:
    IDs_all = np.array([int(x[:-1]) for x in labels_all])
    mrks_all = np.array([x[-1] for x in labels_all])
    X_train,X_test,y_train,y_test = [],[],[],[]
    for i in NV_inds:
        curr_mrks = mrks_all[IDs_all==i] #current people's mrks
        curr_segs = segs_all[IDs_all==i]
        curr_labels = labels_all[IDs_all==i]

        curr_inds_Vs = np.where(curr_mrks=='V')[0]
        curr_inds_Ns = np.where(curr_mrks=='N')[0]

        curr_num_Vs = sum(curr_mrks=='V') #all his Vs

        train_num_Vs = int(curr_num_Vs*0.8)
        train_num_Ns = train_num_Vs

        train_inds_Vs = random.sample(curr_inds_Vs, train_num_Vs)
        test_inds_Vs = [x for x in curr_inds_Vs if not (x in train_inds_Vs)]

        #test_inds_Vs = curr_inds_Vs[~ train_inds_Vs]
        train_inds_Ns = random.sample(curr_inds_Ns, train_num_Ns)
        test_inds_Ns = [x for x in curr_inds_Ns if not (x in train_inds_Ns)]
        #test_inds_Ns = curr_inds_Vs[~ train_inds_Ns]
#        print train_inds_Ns
#        print test_inds_Ns

        if len(train_inds_Vs) > 0:
            X_train += curr_segs[train_inds_Vs].tolist()
            y_train += curr_labels[train_inds_Vs].tolist()
        if len(train_inds_Ns) > 0:
            X_train += curr_segs[train_inds_Ns].tolist()
            y_train += curr_labels[train_inds_Ns].tolist()

        if len(test_inds_Vs) > 0:
            X_test += curr_segs[test_inds_Vs].tolist()
            y_test += curr_labels[test_inds_Vs].tolist()
        if len(test_inds_Ns) > 0:
            X_test += curr_segs[test_inds_Ns].tolist()
            y_test += curr_labels[test_inds_Ns].tolist()

        #print i, curr_num_Vs
        #print i,train_num_Vs,train_num_Ns


y_mrks = np.array([y[-1] for y in y_all])
y_train_mrks = np.array([y[-1] for y in y_train])
y_test_IDs = np.array([y[:-1] for y in y_test])
y_test_mrks = np.array([y[-1] for y in y_test])

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='string').ravel()
    print y
    quit()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

method_clsf = 'DL'
if method_clsf == 'SVM':
    not_trained = True
    from sklearn.externals import joblib
    if not_trained:
        clf = svm.SVC(kernel='rbf', C=10., gamma=0.1)
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'test_clf.pkl')
    else:
        clf = joblib.load('test_clf.pkl')

    res_pred = clf.predict(X_test)
elif method_clsf == 'DL':
    not_trained = True
    from sklearn.externals import joblib

    if not_trained:
        model = Sequential()
        model.add(Dense(feat_dim,activation='relu',input_shape=(feat_dim,)))
        #model.add(Dense(input_dim,activation='relu'))

        num_categs = len(set(y_train))

        Y_train = to_categorical(y_train, num_categs)
        Y_test = to_categorical(y_test, num_categs)

        model.add(Dense(num_categs,activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X_train, Y_train, validation_split=0.2,
                  batch_size=32, nb_epoch=50, verbose=0)
        model.save('test_clf_DL.pkl')
    else:
        model = keras.models.load_model('test_clf_DL.pkl')
    #score = model.evaluate(X_test, Y_test, verbose=0)
    res_pred = model.predict_classes(X_test)

res_look = []
for i in range(len(res_pred)):
    res_look.append((res_pred[i], y_test[i]))
#print res_look

res_pred_IDs = np.array([y[:-1] for y in res_pred])
res_pred_mrks = np.array([y[-1] for y in res_pred])

only_test_ID = True
if only_test_ID:
    to_be_predct = res_pred_IDs
    to_be_tested = y_test_IDs
else:
    to_be_predct = res_pred
    to_be_tested = y_test

look_stat = 'V'
to_be_predct = to_be_predct[y_test_mrks == look_stat]
to_be_tested = to_be_tested[y_test_mrks == look_stat]

res_by_seg = mf.get_corr_ratio(res_pred=to_be_predct, y_test=to_be_tested, type='by_seg')
res_by_categ = mf.get_corr_ratio(res_pred=to_be_predct, y_test=to_be_tested, type='by_categ')
one_res = (float(format(res_by_seg, '.3f')), float(format(res_by_categ, '.3f')))
print one_res

