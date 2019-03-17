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

from sklearn.neighbors import KNeighborsClassifier as KNC
from xgboost import XGBClassifier as XGBC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC

from time import time as Time

class Seg:
    def __init__(self, sig, fs, ID, mrk):
        self.sig = sig
        self.fs = fs
        self.ID = ID
        self.mrk = mrk

        self.ind = -1 # local ind w/i the segs list
        self.feat = []

class Rec:
    def __init__(self, record, fs, ID, anns=None):
        self.record = record
        self.fs = fs
        self.anns = anns

        self.segs = []

######################
#TODO Step 0: Parameter input
######################
strategies = ['allN_data', 'NV_data', 'all_data', 'combine_IDs']
# only 'NV_data' is special, where there is need to consider both 'N' and 'V' correct rate
clsfy_methods = ['DTC', 'SVM', 'DL', 'boosting', 'kNN', 'Logit', 'GNB']

strategy = 'NV_data'
ratio = 0.8
method_clsf = 'boosting'

##

def optimCurveFit(strategy, method_clsf, ratio=0.8, NV_type='NVequals'):
    constrain_time = False

    ######################
    #TODO Step 1: Data input
    ######################
    data_set = 'mitdb' # 'ecgiddb', 'mitdb'
    channel = 0
    records, IDs, fss, annss = mf.load_data(data_set, channel)#, num_persons=60, record_time=20)
    fs = fss[0]

    records = np.array(records)
    IDs = np.array(IDs)
    annss = np.array(annss)
    ######################


    ######################
    #TODO Step 2: Data selection
    ######################

    if (strategy=='allN_data') or (strategy=='all_data'):
        '' # do nothing here
    elif strategy == 'NV_data':
        NV_inds = [6,15,18,23,24,26,29,31,33,35,39,41,42,46]
        #for i in NV_inds: #range(annss.shape[0]): #
        #    print i, Counter(annss[i][1])['V']

        records = records[NV_inds, :]
        IDs = IDs[NV_inds]
        annss = annss[NV_inds, :]

        ## re-numbering the IDs... wtf
        for i in range(len(NV_inds)):
            IDs[i] = i
    elif strategy == 'combine_IDs':
        print IDs
        for i in range(int(len(records)/2)):
            IDs[i*2+1] = IDs[i*2]
        print IDs

    if constrain_time:
        look_time = 100. # in s
        look_ind = int(look_time * fs)
        records = records[:, :look_ind]
        annss = annss[:, :look_ind]


    recs = []
    for i in range(len(records)):
        curr_rec = Rec(records[i], fs, IDs[i], annss[i])
        recs.append(curr_rec)
    ######################


    ######################
    #TODO Step 3: Data filtering
    ######################

    ######################


    ######################
    #TODO Step 4: Data segmentation
    ######################
    USE_BIOSPPY_FILTERED = True
    sigs, labels_bySegs = mf.get_seg_data(records, IDs, fss, USE_BIOSPPY_FILTERED
                                           , annss=annss)
    sigs, labels_bySegs = np.array(sigs), np.array(labels_bySegs)
    mrks_bySegs = np.array([x[-1] for x in labels_bySegs])

    if strategy == 'allN_data':
        N_masks = (mrks_bySegs == 'N')
        sigs = sigs[N_masks,:]
        labels_bySegs = labels_bySegs[N_masks]

    IDs_bySegs = [int(x[:-1]) for x in labels_bySegs]
    mrks_bySegs = [x[-1] for x in labels_bySegs]
    IDs_bySegs, mrks_bySegs = np.array(IDs_bySegs), np.array(mrks_bySegs)


    segs = []
    for i in range(len(sigs)):
        curr_seg = Seg(sig=sigs[i], fs=fs, ID=IDs_bySegs[i], mrk=mrks_bySegs[i])
        segs.append(curr_seg)
    segs = np.array(segs)
    ######################



    #for one_label in labels_all:
    #    if ('N' in one_label) or ('V' in one_label):
    #        print one_label
    #quit()

    #segs_all, labels_all = np.array(segs_all), np.array(labels_all)

    ######################
    #TODO Step 5: feature extraction
    ######################
    X_all = []
    y_all = []
    method_feat = 'PCA' # 'template_matching'

    if method_feat == 'PCA':
        feat_dim = 20
        pca = PCA(n_components=feat_dim)
        X_all = np.array([x.sig for x in segs])
        X_all = pca.fit(X_all).transform(X_all)

        for i in range(len(segs)):
            segs[i].feat = X_all[i,:]
        y_all = np.array([x.ID for x in segs])

    X_all = np.array(X_all)
    ######################

    ######################
    #TODO Step 6: Data split
    ######################
    if strategy != 'NV_data':
        X_train,X_test,y_train,y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42)
    else:
        X_train,X_test,y_train,y_test = [],[],[],[]
        y_test_mrks = []
        for i in range(len(NV_inds)):
            curr_mrks = mrks_bySegs[IDs_bySegs==i] #current people's mrks\
            #print curr_mrks

            curr_segs = segs[IDs_bySegs==i]
            curr_labels = labels_bySegs[IDs_bySegs==i]

            curr_inds_Vs = np.where(curr_mrks=='V')[0]
            curr_inds_Ns = np.where(curr_mrks=='N')[0]

            curr_num_Vs = sum(np.array(curr_mrks)=='V') #all his Vs
            curr_num_Ns = sum(np.array(curr_mrks)=='N')

            if NV_type == 'fixV':
                train_num_Vs = int(curr_num_Vs*.8)
                train_num_Ns = min([int(curr_num_Ns*.8), int(ratio * train_num_Vs)])
            elif NV_type == 'NVequals':
                train_num_Vs = int(curr_num_Vs*ratio)
                train_num_Ns = train_num_Vs

            train_inds_Vs = random.sample(curr_inds_Vs, train_num_Vs)
            test_inds_Vs = [x for x in curr_inds_Vs if not (x in train_inds_Vs)]

            #test_inds_Vs = curr_inds_Vs[~ train_inds_Vs]
            train_inds_Ns = random.sample(curr_inds_Ns, train_num_Ns)
            test_inds_Ns = [x for x in curr_inds_Ns if not (x in train_inds_Ns)]

            #print len(train_inds_Vs), len(test_inds_Vs)
            #print len(train_inds_Ns), len(test_inds_Ns)

            #test_inds_Ns = curr_inds_Vs[~ train_inds_Ns]
    #        print train_inds_Ns
    #        print test_inds_Ns

            curr_IDs = IDs_bySegs[IDs_bySegs==i]
            #print curr_IDs

            for one_seg in curr_segs[train_inds_Vs]:
                X_train.append(one_seg.feat.tolist())
            for one_lab in curr_IDs[train_inds_Vs]:
                y_train.append(one_lab)

            for one_seg in curr_segs[train_inds_Ns]:
                X_train.append(one_seg.feat.tolist())
            for one_lab in curr_IDs[train_inds_Ns]:
                y_train.append(one_lab)

            for one_seg in curr_segs[test_inds_Vs]:
                X_test.append(one_seg.feat.tolist())
            for one_lab in curr_IDs[test_inds_Vs]:
                y_test.append(one_lab)
            for one_mrk in curr_mrks[test_inds_Vs]:
                y_test_mrks.append(one_mrk)

            for one_seg in curr_segs[test_inds_Ns]:
                X_test.append(one_seg.feat.tolist())
            for one_lab in curr_IDs[test_inds_Ns]:
                y_test.append(one_lab)
            for one_mrk in curr_mrks[test_inds_Ns]:
                y_test_mrks.append(one_mrk)

            #print i
            #print len(X_train), len(y_train), len(X_test), len(y_test)

    X_train, y_train, X_test, y_test = \
    np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    ######################

    #print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    #quit()
    #print X_train
    #print X_test
    #y_train = [int(y[:-1]) for y in y_train]
    #y_test = [int(y[:-1]) for y in y_test]

    ######################
    #TODO Step 7: Model training
    ######################
    time_before_training = Time()

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
    elif method_clsf == 'Logit':
        clf = LR(C=10.)
        clf.fit(X_train, y_train)
        res_pred = clf.predict(X_test)
    elif method_clsf == 'kNN':
        clf = KNC()
        clf.fit(X_train, y_train)
        res_pred = clf.predict(X_test)
    elif method_clsf == 'DTC':
        clf = DTC()
        clf.fit(X_train, y_train)
        res_pred = clf.predict(X_test)
    elif method_clsf == 'boosting':
        clf = XGBC()
        clf.fit(X_train, y_train)
        res_pred = clf.predict(X_test)
    elif method_clsf == 'GNB':
        clf = GNB()
        clf.fit(X_train, y_train)
        res_pred = clf.predict(X_test)
    elif method_clsf == 'DL':
        not_trained = True
        from sklearn.externals import joblib

        if not_trained:
            model = Sequential()
            model.add(Dense(feat_dim,activation='relu',input_shape=(feat_dim,)))
            #model.add(Dense(input_dim,activation='relu'))

            num_categs = len(set(y_train))

            Y_train = np_utils.to_categorical(y_train, num_categs)
            Y_test = np_utils.to_categorical(y_test, num_categs)

            model.add(Dense(num_categs,activation='softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            #print X_train.shape
            #print Y_train.shape

            model.fit(X_train, Y_train, validation_split=0.2,
                      batch_size=32, nb_epoch=50, verbose=0)
            #model.save('test_clf_DL.pkl')
        else:
            model = keras.models.load_model('test_clf_DL.pkl')
        #score = model.evaluate(X_test, Y_test, verbose=0)

    time_after_training = Time()

    ######################
    #TODO Step 8: Model testing
    ######################
    if method_clsf != 'DL':
        res_pred = clf.predict(X_test)
    else:
        res_pred = model.predict_classes(X_test)
    ######################

    ######################
    #TODO Step 9: Result output
    ######################
    train_time = time_after_training - time_before_training

    print_res = False
    if print_res:
        print ''
        print 'Parameters:'
        print 'strategy:', strategy
        print 'constrain_time:', constrain_time
        print 'ratio:', ratio
        print 'method_clsf:', method_clsf

        #print ''

        print 'Results:'
        print 'Used time for training:', time_after_training - time_before_training

    res_look = []
    for i in range(len(res_pred)):
        res_look.append((res_pred[i], y_test[i]))
    #print res_look

    if False:
        res_pred_IDs = np.array([y[:-1] for y in res_pred])
        res_pred_mrks = np.array([y[-1] for y in res_pred])

        only_test_ID = True
        if only_test_ID:
            to_be_predct = res_pred_IDs
            to_be_tested = y_test
        else:
            to_be_predct = res_pred
            to_be_tested = y_test


    ##TODO: adjust accordingly
    if strategy == 'NV_data':
        look_stat = 'V'
        y_test_mrks = np.array(y_test_mrks)
        #print y_test_mrks
        to_be_predct = res_pred[y_test_mrks == look_stat]
        to_be_tested = y_test[y_test_mrks == look_stat]

        res_by_seg = mf.get_corr_ratio(res_pred=to_be_predct, y_test=to_be_tested, type='by_seg')
        res_by_categ = mf.get_corr_ratio(res_pred=to_be_predct, y_test=to_be_tested, type='by_categ')
        one_res = (float(format(res_by_seg, '.3f')), float(format(res_by_categ, '.3f')))
        accuBySeg_V = one_res[0]
        #print len(to_be_predct), one_res

        look_stat = 'N'
        to_be_predct = res_pred[y_test_mrks == look_stat]
        to_be_tested = y_test[y_test_mrks == look_stat]

        res_by_seg = mf.get_corr_ratio(res_pred=to_be_predct, y_test=to_be_tested, type='by_seg')
        res_by_categ = mf.get_corr_ratio(res_pred=to_be_predct, y_test=to_be_tested, type='by_categ')
        one_res = (float(format(res_by_seg, '.3f')), float(format(res_by_categ, '.3f')))
        accuBySeg_N = one_res[0]
        #print len(to_be_predct), one_res
        return [accuBySeg_V, accuBySeg_N, train_time]
    else:
        to_be_predct = res_pred
        to_be_tested = y_test

        res_by_seg = mf.get_corr_ratio(res_pred=to_be_predct, y_test=to_be_tested, type='by_seg')
        res_by_categ = mf.get_corr_ratio(res_pred=to_be_predct, y_test=to_be_tested, type='by_categ')
        one_res = (float(format(res_by_seg, '.3f')), float(format(res_by_categ, '.3f')))
        return [one_res[0], train_time]