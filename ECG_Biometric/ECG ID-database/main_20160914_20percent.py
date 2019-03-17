
import numpy as np
import glob
import re
import wfdb
from biosppy.signals import ecg
import pywt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.linear_model import SGDClassifier
#from matplotlib.mlab import PCA
import csv

# Sampling freq: 0.002s = 2ms
# Segment period: 256ms (determined by QRS width), so there are 256/2 = 128 datapoints
def get_segs(sig, peaks, seg_len):
    seg_halflen = seg_len/2 # unit: ms

    segs = []
    seg_Lbnds = [0] * len(peaks)
    seg_Rbnds = [0] * len(peaks)
    seg_Lbnds[0] = max(0, int(round(peaks[0]-seg_halflen/1000. * fs)))
    for i in range(1,len(peaks)):
        seg_Lbnds[i] = int(round(peaks[i] - seg_halflen/1000. * fs))
    for i in range(0,len(peaks)-1):
        seg_Rbnds[i] = int(round(peaks[i] + seg_halflen/1000. * fs))
    seg_Rbnds[-1] = min(int(round(peaks[-1] + seg_halflen/1000. * fs)), len(sig))

    for i in range(len(peaks)):
        segs.append(sig[seg_Lbnds[i]:seg_Rbnds[i]]) # Warning: bnds are shown in float, weird
    return segs

def dist_sig(sig1, sig2, lvl):
    coeffs1 = pywt.wavedec(sig1, 'db3', level=lvl)
    coeffs2 = pywt.wavedec(sig2, 'db3', level=lvl)
    dist = 0
    tau = .6
    for i in range(len(coeffs1)):
        for j in range(len(coeffs1[i])):
            dist += abs(coeffs1[i][j]-coeffs2[i][j])/max(coeffs1[i][j], coeffs2[i][j], tau)
    return dist

Person_IDs = range(1,90+1)
seg_len = 256

avg_labels = []
all_seg_avgs = []
all_targetnames = []

labels = []
all_segs = []
for i in Person_IDs:
    curr_ID_str = str(i)
    if i < 10:
        curr_ID_str = '0' + curr_ID_str
    curr_foldername = 'Person_' + curr_ID_str
    all_targetnames.append(curr_foldername)

    all_recs =  glob.glob(curr_foldername + "/*.dat")
    all_seg_avgs_i = []
    for one_rec in all_recs: # One rec has many pulses
        filename = one_rec[:-4]

        sig, fields = wfdb.rdsamp(filename) #, pbdl=0)
        fs = fields['fs']
        sig_origin = sig[:,0]

        out = ecg.ecg(signal=sig_origin, sampling_rate=fs, show=False)
        peaks = out[2]

        segs = get_segs(sig_origin, peaks, seg_len)

        qual_len = int(round(seg_len/1000. * fs)) # len of a complete QRS seg
        qual_num = 0
        seg_sum = [0] * qual_len
        for seg in segs:
            if len(seg) == qual_len:
                qual_num += 1
                seg_sum = [x+y for x,y in zip(seg, seg_sum)]

                all_segs.append(seg)
                labels.append(i)

        seg_avg = [x/float(qual_num) for x in seg_sum]
        all_seg_avgs_i.append(seg_avg)
        all_seg_avgs.append(seg_avg)
        avg_labels.append(i)

    PLOT_WITHINPERSON_CMAP = False
    if PLOT_WITHINPERSON_CMAP:
        dist_matrix_withinperson = [[dist_sig(seg_avg1, seg_avg2, 4) \
                                    for seg_avg2 in all_seg_avgs_i] for seg_avg1 in all_seg_avgs_i]
        plt.imshow(dist_matrix_withinperson, cmap='hot', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.show()
        quit()

#dist_matrix_interperson = [[dist_sig(seg_avg1, seg_avg2,4) \
#                            for seg_avg2 in all_seg_avgs] for seg_avg1 in all_seg_avgs]
#plt.imshow(dist_matrix_interperson, cmap='hot', interpolation='nearest')
#plt.gca().invert_yaxis()
#plt.show()

TO_CHECK_SEGS = False
if TO_CHECK_SEGS:
    print 'Total number of segs: '+str(len(all_segs))
    seg_inds_to_look = [10,100,1000,2000,3000,5000,7000]
    for seg_ind in seg_inds_to_look:
        plt.plot(all_segs[seg_ind])
    plt.legend()
    plt.show()
    quit()

X = all_segs

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
#print X_r

#dist_matrix_interperson = [[distance.euclidean(seg_avg1, seg_avg2) \
#                            for seg_avg2 in X_r] for seg_avg1 in X_r]
#plt.imshow(dist_matrix_interperson, cmap='hot', interpolation='nearest')
#plt.gca().invert_yaxis()
#plt.show()

all_segs = np.array(all_segs)
print all_segs.shape

labels = np.array(labels)

training_set = np.empty([0,128])
training_y = []
testing_set = np.empty([0,128])
testing_y = []
train_ratio = 0.67
for i in Person_IDs:

    curr_segs = all_segs[labels==i,:]
    train_num = int(round(len(curr_segs) * train_ratio))

    #print train_num
    #print curr_segs[:train_num].shape
#    quit()

    #training_set += curr_segs[:train_num]
    training_set = np.concatenate((training_set, np.array(curr_segs[:train_num])))

    training_y += [i]*train_num
    #testing_set += curr_segs[train_num:]
    testing_set = np.concatenate((testing_set, np.array(curr_segs[train_num:])))
    testing_y += [i]*(len(curr_segs)-train_num)

training_set = np.array(training_set)
training_y = np.array(training_y)
clf = SGDClassifier(loss="hinge",penalty='l2')
#print training_set.shape, training_y.shape
clf.fit(training_set, training_y)

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
