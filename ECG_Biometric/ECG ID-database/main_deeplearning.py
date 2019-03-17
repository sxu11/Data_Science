
import numpy as np
import glob
import re
import wfdb
from biosppy.signals import ecg
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
#import xgboost as xgb
import csv
import my_funcs
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution1D, MaxPooling1D

from keras.utils import np_utils

from sklearn.cross_validation import train_test_split

# preparing data: each segment is a ...

# Sampling freq: 0.002s = 2ms
# Segment period: 256ms (determined by QRS width), so there are 256/2 = 128 datapoints
# Num Persons: 90
# Num recs: 310
# Num segs: 780

Person_IDs = range(1,90+1)
SEG_LEN_MS = 1000 # PQRST segment length, unit: ms

TO_PLOT_AGING = False
if TO_PLOT_AGING:
    FIRST_TIME = True

labels_all = []
segs_all = []
segs_allqual = []
for i in Person_IDs:
    curr_ID_str = str(i)
    if i < 10:
        curr_ID_str = '0' + curr_ID_str
    curr_foldername = 'Person_' + curr_ID_str

    curr_filenames =  glob.glob(curr_foldername + "/*.dat")

    j = 0
    for one_filename in curr_filenames: # One rec has many pulses
        filename = one_filename[:-4]

        sig, fields = wfdb.rdsamp(filename)#, sampto=1000) #, pbdl=0)
        #print fields

        fs = fields['fs']
        sig_use = sig[:,1] ## TODO: treated by wfdb

        dt = 1./fs
        seg_len = SEG_LEN_MS/1000. / (1./fs) # unit: num of bins
        #print seg_len

        TO_PLOT_ORIGIN_SIG = False
        if TO_PLOT_ORIGIN_SIG and 'Person_01/rec_7' in filename:
            ts = np.linspace(0,dt*len(sig_use),len(sig_use))
            fig = plt.figure(figsize=(8,6))
            plt.plot(ts, sig[:,0])
            #my_funcs.config_plot('Time /s', 'Amplitude /mV')
            #plt.show()
            plt.plot(ts, sig[:,1])
            my_funcs.config_plot('Time /s', 'Amplitude /mV')

            print 'filename: '+filename
            plt.show()

            ## baseline drift handle
            x = sig[:,0]
            my_funcs.write_to_file(x)
            quit()
            y = my_funcs.my_butter(x,freqs=0.95,type='high')
            plt.plot(x,y)
            plt.show()
            ##
            quit()

        out = ecg.ecg(signal=sig_use, sampling_rate=fs, show=False)
        peaks = out[2] ## TODO: get peaks by ecg.ecg
        ecg_filtered = out[1] ## TODO: treated by ecg.ecg

        TO_PLOT_DIFF_FILTERS = False
        if TO_PLOT_DIFF_FILTERS:
            ts = np.linspace(0,dt*len(sig_use),len(sig_use))
            look_inds = int(len(sig_use)/24.)
            gs = gridspec.GridSpec(3, 1, hspace=0.2, wspace=0.2)
            #fig = plt.figure()
            #ax1 = fig.add_subplot(gs[0,0])
            #ax1.plot(ts[:look_inds],sig[:look_inds,0])
            #ax2 = fig.add_subplot(gs[1,0])
            #ax2.plot(ts[:look_inds],sig[:look_inds,1])
            #ax3 = fig.add_subplot(gs[2,0])
            #ax3.plot(ts[:look_inds],ecg_filtered[:look_inds])

            print 'filename: '+filename
            if 'Person_01/rec_10' in filename:
                ''
                x = sig[:look_inds,0]
                y = my_funcs.my_butter(x, freqs=0.1)
                plt.plot(x)
                plt.plot(y)
                plt.show()
                quit()
            #plt.show()

        TO_PLOT_PEAKS = False
        if TO_PLOT_PEAKS:
            ts = np.linspace(0,dt*len(sig_use),len(sig_use))
            peak_xs = [x*dt for x in peaks]
            peak_ys = [sig_use[x] for x in peaks]
            fig = plt.figure(figsize=(8,6))
            plt.plot(ts, sig[:,1])
            plt.plot(peak_xs, peak_ys, 'ro')
            my_funcs.config_plot('Time /s', 'Amplitude /mV')
            plt.show()

        ## after all these things, get segs by my own function

        ## Three options:
        # sig[:,0], accuracy 87.8%
        # sig[:,1], accuracy 87.8%
        # ecg_filtered, accuracy 92.2%
        segs = my_funcs.get_segs(ecg_filtered, peaks, seg_len, fs=fs)


        qual_len = int(round(seg_len/1000. * fs)) # len of a complete QRS seg
        qual_num = 0
        seg_sum = [0] * qual_len

        TO_PLOT_SEGS = False
        if TO_PLOT_SEGS:
            fig = plt.figure(figsize=(8,6))

        for seg in segs:
            segs_all.append(seg)
            if len(seg) == qual_len:
                qual_num += 1

                segs_allqual.append(seg)
                TO_PLOT_ONE_SEG = False
                if TO_PLOT_ONE_SEG:
                    ts = np.linspace(0,dt*len(seg),len(seg))
                    fig = plt.figure(figsize=(8,6))
                    plt.plot(ts, seg)
                    my_funcs.config_plot('Time /s', 'Amplitude /mV')
                    plt.show()

                if TO_PLOT_AGING:
                    ts = np.linspace(0,dt*len(seg),len(seg))
                    if 'rec_1' in one_filename:
                        plt.plot(ts, seg, 'b')
                        if FIRST_TIME:
                            fig = plt.figure(figsize=(8,6))
                            FIRST_TIME = False
                    if 'rec_2' in one_filename:
                        plt.plot(ts, seg, 'g')
                    if 'Person_02' in one_filename:
                        my_funcs.config_plot('Time /s', 'Amplitude /mV')
                        plt.show()

                labels_all.append(i)


                if TO_PLOT_SEGS:
                    ts = np.linspace(0,dt*len(seg),len(seg))
                    plt.plot(ts, seg)

        if TO_PLOT_SEGS:
            my_funcs.config_plot('Time /s', 'Amplitude /mV')
            plt.show()
            print one_filename

        seg_avg = [x/float(qual_num) for x in seg_sum]
        j += 1


segs_all, labels_all = np.array(segs_all), np.array(labels_all)

#print segs_all.shape
#print labels_all.shape

X_train,X_test,y_train,y_test = train_test_split(
    segs_all, labels_all, test_size=0.2, random_state=42)
#print X_train.shape

#X_train = X_train.reshape(X_train.shape[0],1,250,)
#X_test = X_test.reshape(X_test.shape[0],1,250,)

Y_train = np_utils.to_categorical(y_train, 91)
Y_test = np_utils.to_categorical(y_test, 91)


print X_train.shape, Y_train.shape
#print X_test.shape, Y_test.shape

# DL part
model = Sequential()
model.add(Dense(250,activation='relu',input_shape=(250,)))
model.add(Dense(250,activation='relu',input_shape=(250,)))
model.add(Dense(91,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train, validation_split=0.2,
          batch_size=250, nb_epoch=50, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)
print score
