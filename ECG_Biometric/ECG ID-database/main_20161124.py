
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
import csv
import my_funcs
from mpl_toolkits.mplot3d import Axes3D

from math import exp, log

from scipy.spatial.distance import euclidean

# Sampling freq: 0.002s = 2ms
# Segment period: 256ms (determined by QRS width), so there are 256/2 = 128 datapoints
# Num Persons: 90
# Num recs: 310
# Num segs: 780

Person_IDs = range(1,50+1)
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
        sig_use = sig[:,1]

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
        peaks = out[2]
        ecg_filtered = out[1]

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


TO_CHECK_SEGS = False
if TO_CHECK_SEGS:
    print 'Total number of segs: '+str(len(segs_allqual)) # 7850
    seg_inds_to_look = range(20) # [10,100,1000,2000,3000,5000,7000]
    for seg_ind in seg_inds_to_look:
        plt.plot(segs_allqual[seg_ind])
    plt.legend()
    plt.show()
    quit()

#print len(segs_all), len(segs_allqual)

X = segs_allqual
X = np.array(X)

#n_compon = 2

n_compons = [20] # [int(exp(x)) for x in np.linspace(log(1), log(100), num=20)]
correct_rates = []
max_rates = []
for n_compon in n_compons:
    num_repeats = 1

    curr_correct_rates = []
    for j1 in range(num_repeats):
        pca = PCA(n_components=n_compon)
        X_r = pca.fit(X).transform(X)

        TO_PLOT_PCA = False
        if TO_PLOT_PCA:
            for i in range(3):
                fig = plt.figure(figsize=(8,6))
                plt.plot(X[i,:])
                my_funcs.config_plot('Time /s', 'Amplitude /mV')
                plt.show()

                fig = plt.figure(figsize=(8,6))
                plt.plot(X_r[i,:])
                my_funcs.config_plot('Features', 'Amplitude /mV')
                plt.show()

        X_r = np.array(X_r)
        #print X_r.shape

        labels_all = np.array(labels_all)

        TO_PLOT_PCA_scatter = False
        if TO_PLOT_PCA_scatter:
            ''
            NUM_PERSONS = 20

            NUM_PT_EACH_PERSON = 10

            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 1, NUM_PERSONS)]

            if n_compon == 2:
                fig = plt.figure(figsize=(8,6))
                #curr_segs = []
                for i in range(0,NUM_PERSONS):
                    curr_person_recs = X_r[labels_all==i+1,:][:NUM_PT_EACH_PERSON].tolist()
                    for one_rec in curr_person_recs:
                        plt.scatter(one_rec[0], one_rec[1], color=colors[i])

                my_funcs.config_plot('Component 1', 'Component 2')
                plt.show()
            elif n_compon == 3:
                fig = plt.figure(figsize=(8,6))
                ax = fig.add_subplot(111, projection='3d')
                for i in range(0,NUM_PERSONS):
                    curr_person_recs = X_r[labels_all==i+1,:][:NUM_PT_EACH_PERSON].tolist()
                    for one_rec in curr_person_recs:
                        ax.scatter(one_rec[0], one_rec[1], one_rec[2], color=colors[i])
                        #plt.scatter(one_rec[0], one_rec[1], color=colors[i])
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_zlabel('Component 3')
                #my_funcs.config_plot()
                plt.show()

        training_set = np.empty([0,n_compon])
        training_y = []
        testing_set = np.empty([0,n_compon])
        testing_y = []
        train_ratio = 0.67
        for i in Person_IDs:

            curr_segs = X_r[labels_all==i,:]
            train_num = int(round(len(curr_segs) * train_ratio))

            #training_set += curr_segs[:train_num]
            training_set = np.concatenate((training_set, np.array(curr_segs[:train_num])))

            training_y += [i]*train_num
            #testing_set += curr_segs[train_num:]
            testing_set = np.concatenate((testing_set, np.array(curr_segs[train_num:])))
            testing_y += [i]*(len(curr_segs)-train_num)

        training_set_refined = np.empty([0,n_compon])
        training_y_refined = []

        training_y = np.array(training_y)

        for i in Person_IDs:
            curr_trainingset = training_set[training_y==i]
            if len(curr_trainingset) == 0:
                continue

            curr_trainingset_avg = [0] * len(curr_trainingset[0])
            for j in range(len(curr_trainingset[0])):
                for one_training in curr_trainingset:
                    curr_trainingset_avg[j] += one_training[j]
                curr_trainingset_avg[j] /= float(len(curr_trainingset[0]))

            curr_dists = []
            for one_training in curr_trainingset:
                curr_dists.append(euclidean(one_training, curr_trainingset_avg))
            curr_dists_sorted_inds = np.array(curr_dists).argsort()
            num_selected = min(35,len(curr_trainingset))
            curr_dists_selected_inds = curr_dists_sorted_inds[:num_selected]

            training_set_refined = np.concatenate((training_set_refined, curr_trainingset[curr_dists_selected_inds,:]))
            training_y_refined += [i] * num_selected
        training_y_refined = np.array(training_y_refined)

        training_set = np.array(training_set)
        training_y = np.array(training_y)
        clf = SGDClassifier(loss="hinge",penalty='l2')
        clf.fit(training_set_refined, training_y_refined)

        testing_set = np.array(testing_set)
        pred_res = clf.predict(testing_set)

        testing_y = np.array(testing_y)


        OLD_METHOD = False

        if OLD_METHOD:
            num_same = 0.
            num_diff = 0.
            for i in range(len(pred_res)):
                if pred_res[i] == testing_y[i]:
                    num_same += 1
                else:
                    num_diff += 1
            print num_same/(num_same + num_diff)

            with open('res.txt','w') as fw:
                writer = csv.writer(fw,delimiter='\t')
                writer.writerow(pred_res)
                writer.writerow(testing_y.tolist())
        else:
            num_same = 0.
            num_diff = 0.
            predicted_IDs = []
            for i in Person_IDs:
                curr_inds = np.where([testing_y == i])[1]
                counts = np.bincount(pred_res[curr_inds])
                if np.argmax(counts) == i:
                    num_same += 1
                else:
                    num_diff += 1
                predicted_IDs.append(np.argmax(counts))
        curr_correct_rates.append(num_same/(num_same + num_diff))

    correct_rates.append(curr_correct_rates)
    max_rates.append(max(curr_correct_rates))

if len(max_rates) > 1:
    fig = plt.figure(figsize=(8,6))
    plt.plot(n_compons, max_rates, '*-', linewidth=2.)
    my_funcs.config_plot('Number of components', 'True positive rate')
    plt.show()
else:
    print num_same/(num_same + num_diff)
    with open('res.txt','w') as fw:
        writer = csv.writer(fw,delimiter='\t')
        writer.writerow(Person_IDs)
        writer.writerow(predicted_IDs)

    fig = plt.figure(figsize=(8,6))
    plt.scatter(Person_IDs, predicted_IDs)
    my_funcs.config_plot('Real ID', 'Predicted ID')
    plt.show()


