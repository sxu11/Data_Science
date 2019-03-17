
import numpy as np
import glob
import re
import wfdb
from biosppy.signals import ecg

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.linear_model import SGDClassifier
import csv
import my_funcs

from scipy.spatial.distance import euclidean

# Sampling freq: 0.002s = 2ms
# Segment period: 256ms (determined by QRS width), so there are 256/2 = 128 datapoints
# Num Persons: 90
# Num recs: 310
# Num segs: 780

Person_IDs = range(1,90+1)
seg_len = 360

labels_all = []
segs_all = []
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

        fs = fields['fs']
        sig_use = sig[:,1]

        out = ecg.ecg(signal=sig_use, sampling_rate=fs, show=False)
        peaks = out[2]

        segs = my_funcs.get_segs(sig_use, peaks, seg_len, fs=fs)

        qual_len = int(round(seg_len/1000. * fs)) # len of a complete QRS seg
        qual_num = 0
        seg_sum = [0] * qual_len
        for seg in segs:
            if len(seg) == qual_len:
                qual_num += 1

                segs_all.append(seg)
                labels_all.append(i)

        seg_avg = [x/float(qual_num) for x in seg_sum]

        j += 1


TO_CHECK_SEGS = False
if TO_CHECK_SEGS:
    print 'Total number of segs: '+str(len(segs_all)) # 7850
    seg_inds_to_look = range(20) # [10,100,1000,2000,3000,5000,7000]
    for seg_ind in seg_inds_to_look:
        plt.plot(segs_all[seg_ind])
    plt.legend()
    plt.show()
    quit()

X = segs_all

n_compon = 15

pca = PCA(n_components=n_compon)
X_r = pca.fit(X).transform(X)

X_r = np.array(X_r)
print X_r.shape

labels_all = np.array(labels_all)

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

num_same = 0
num_diff = 0
for i in range(len(pred_res)):
    if pred_res[i] == testing_y[i]:
        num_same += 1
    else:
        num_diff += 1
print num_same/(num_same + num_diff + .0)


with open('res.txt','w') as fw:
    writer = csv.writer(fw,delimiter='\t')
    writer.writerow(pred_res)
    writer.writerow(testing_y.tolist())
