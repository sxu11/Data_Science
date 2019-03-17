
from scipy.signal import butter, lfilter
import pywt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from scipy.stats import itemfreq
import matplotlib as mpl


import glob
import os
import wfdb
import re

from biosppy.signals import ecg


def load_data(data_name, channel_ind, num_persons = 30, record_time = None):
    curr_tot_dir = os.path.dirname(__file__)

    records, labels, fss = [], [], []

    if data_name == 'ecgiddb':
        curr_data_dir = os.path.join(curr_tot_dir + '/ecgiddb/')
        os.chdir(curr_data_dir)

        #Person_IDs = range(1,num_persons+1)
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
                sig = np.array(sig)

                #print fields
                fss.append(fields['fs'])
                records.append(sig[:,channel_ind])
                labels.append(i)
        os.chdir(curr_tot_dir)
        return records, labels, fss
    elif data_name == 'mitdb':
        curr_data_dir = os.path.join(curr_tot_dir + '/mitdb/')
        os.chdir(curr_data_dir)
        filenames = glob.glob('*.dat')

        annss = []

        ID = 0
        for filename in filenames:
            regex = re.compile(r'\d+')
            curr_ID = regex.findall(filename)[0] # string
            labels.append(ID)

            sig, fields = wfdb.rdsamp(curr_ID)

            fss.append(fields['fs'])

            anns = wfdb.rdann(curr_ID, 'atr')
            #ann_inds = ann[0]
            #ann_mrks = ann[1]

            if record_time != None: # DID set a limit
                sig_len = int(record_time * fss[-1])
                records.append(sig[:sig_len,channel_ind])

                j2 = 0 #
                for j1 in anns[0]: # j1 is annotated index
                    j2 += 1 # j2 is the num of anns before sig_len
                    if j1 > sig_len:
                        break
                j2 -= 1 #TODO:??
                annss.append([anns[0][:j2], anns[1][:j2]])
            else:
                records.append(sig[:,channel_ind])
                annss.append(anns)
            ID += 1
        os.chdir(curr_tot_dir)
        return records, labels, fss, annss

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def my_butter(sig, freqs, type = 'low'):
    b, a = signal.butter(8, freqs, type)
    y = signal.filtfilt(b, a, sig, padlen=150)
    return y

def my_baseline():
    ''

SEG_LEN_IN_SEC = .5 # TODO: may not be enough!
def get_seg_data(records, labels, fss, USE_FILTERED = True, annss = None):
    X_all = []
    y_all = []
    num_recs = len(records)
    for i in range(num_recs):
        fs = fss[i]
        seg_len = SEG_LEN_IN_SEC / (1./fs)

        out = ecg.ecg(signal=records[i], sampling_rate=fs, show=False)
        #print len(records[i])

        ecg_filtered, peaks = out[1], out[2] ## TODO: get peaks by ecg.ecg

        if USE_FILTERED:
            sig_use = ecg_filtered
        else:
            sig_use = records[i]

        if annss is not None:
            print 'Current ID:', labels[i]
            segs, seg_mrks = get_segs(sig_use, peaks, seg_len, fs=fs, anns=annss[i])
        else:
            segs = get_segs(sig_use, peaks, seg_len, fs=fs)
        segs = np.array(segs)
        #print segs.shape

        #for seg in segs:

        for j in range(len(segs)):
            curr_seg = segs[j]
            if annss is not None:
                if (seg_mrks[j] == 'N') or (seg_mrks[j] == 'V'):
                    curr_lab = str(labels[i])+seg_mrks[j]
                else:
                    curr_lab = str(labels[i])+'O'
            else:
                curr_lab = str(labels[i])
            X_all.append(curr_seg)
            y_all.append(curr_lab)

    return X_all, y_all


def get_segs(sig, peaks, seg_len, fs, anns=None):
    seg_prelen = int(round(seg_len * 180./500))
    seg_prolen = int(round(seg_len * 320./500))

    segs = []
    seg_Lbnds = [0] * len(peaks)
    seg_Rbnds = [0] * len(peaks)

    seg_mrks = []

    #todo_num = 1000. # should be 1000.
    seg_Lbnds[0] = max(0, int(round(peaks[0]-seg_prelen)))
    for i in range(1,len(peaks)):
        seg_Lbnds[i] = int(round(peaks[i] - seg_prelen))
    for i in range(0,len(peaks)-1):
        seg_Rbnds[i] = int(round(peaks[i] + seg_prolen))
    seg_Rbnds[-1] = min(int(round(peaks[-1] + seg_prolen)), len(sig))

    for i in range(len(peaks)):
        curr_seg = sig[seg_Lbnds[i]:seg_Rbnds[i]]

        if anns is not None:
            curr_anns = []
            for j in range(1,len(anns[0])):
                if anns[0][j] >= seg_Lbnds[i] and anns[0][j] <= seg_Rbnds[i]:
                    #seg_mrks.append(anns[1][j])
                    curr_anns.append(anns[1][j])

                    look_at_Vs = True
                    if look_at_Vs and anns[1][j]=='V':
                        plt.plot([x * 1./fs for x in range(seg_Lbnds[i],seg_Rbnds[i])],curr_seg)
                        #plt.scatter(anns[0][j],0)
                        #plt.scatter(peaks[i],0,color='r')
                        plt.xlim([seg_Lbnds[i]* 1./fs, seg_Rbnds[i] * 1./fs])
                        config_plot('Time','ECG segment')
                        plt.show()
            if len(curr_anns) == 1:
                seg_mrks.append(curr_anns[0])
                segs.append(curr_seg)
                #print i, anns[0][j]
        #curr_seg = butter_bandpass_filter(curr_seg, 0, 30, fs, order=3)
        else:
            segs.append(curr_seg) # Warning: bnds are shown in float, weird

        INCLUDE_QT_ADJUST = False
        if INCLUDE_QT_ADJUST:
            ### TODO: get Q valley and T peak ###
            # within 0.1ms, the lowest point is Q
            # after 0.1ms, the highest point is T
            ind_div = peaks[i] - seg_Lbnds[i] + int(round(0.1/0.002))
            ind_Q_valley = peaks[i] - seg_Lbnds[i] + \
                           np.argmin(curr_seg[peaks[i] - seg_Lbnds[i]:ind_div])
            ind_T_peak = np.argmax(curr_seg[ind_div:]) + ind_div
            curr_QT = (ind_T_peak-ind_Q_valley)*0.002 # in s

            if i < len(peaks) - 1:
                curr_RRinterval = peaks[i+1] - peaks[i]
            curr_RR = curr_RRinterval * 0.002
            # else: use the previous RRinterval
            Fra_QT = curr_QT + 0.154 * (1-curr_RR)
            #print curr_QT, Fra_QT

            ## Take the whole curr_seg[ind_Q_valley:]
            # resize to curr_len * Fra_QT/curr_QT
            curr_len = len(curr_seg[ind_Q_valley:])
            xvals = np.linspace(0, 1, int(round(curr_len * Fra_QT/curr_QT)))
            x = np.linspace(0,1, curr_len)
            y = curr_seg[ind_Q_valley:]
            yinterp = np.interp(xvals, x,y)
            #print len(x), len(y)
            #print len(xvals), len(yinterp)
            #print '___'

            if len(xvals) >= len(x):
                curr_seg[ind_Q_valley:] = yinterp[:len(curr_seg[ind_Q_valley:])]
            else:
                curr_seg[ind_Q_valley:ind_Q_valley+len(yinterp)] = yinterp

            ######

    if anns is not None:
        return segs, seg_mrks
    else:
        return segs

def get_corr_ratio(res_pred, y_test, type='by_seg'):
    res_pred, y_test = np.array(res_pred), np.array(y_test)
    if type == 'by_seg':
        #print res_pred, y_test
        corr_seg_num = sum(res_pred == y_test)
        return float(corr_seg_num)/len(res_pred)
    elif type == 'by_categ':
        all_categs = set(y_test)
        corr_categ_num = 0
        #print all_categs
        for one_categ in all_categs:
            one_preds = res_pred[y_test == one_categ]

            cnts_pred = itemfreq(one_preds)
            curr_preds = cnts_pred[:,0]
            curr_cnts = [int(x) for x in cnts_pred[:,1]]

            categ_pred_ind = np.argmax(curr_cnts)
            categ_pred = curr_preds[categ_pred_ind]
            corr_categ_num += (categ_pred == one_categ)
        return float(corr_categ_num)/len(all_categs)

def dist_sig(sig1, sig2, lvl):
    coeffs1 = pywt.wavedec(sig1, 'db3', level=lvl)
    coeffs2 = pywt.wavedec(sig2, 'db3', level=lvl)
    dist = 0
    tau = .6
    for i in range(len(coeffs1)):
        for j in range(len(coeffs1[i])):
            dist += abs(coeffs1[i][j]-coeffs2[i][j])/max(coeffs1[i][j], coeffs2[i][j], tau)
    return dist

def config_plot(xlab='', ylab='', legend_loc=(0.,0.)):
    FONTSIZE_MATH = 24
    FONTSIZE_TEXT = 22
    FONTSIZE_TICK = 20

    plt.rc('text', usetex=True)
    plt.tick_params(labelsize=FONTSIZE_TICK)

    x_is_math = '$' in xlab
    y_is_math = '$' in ylab

    plt.legend(bbox_to_anchor=(legend_loc),bbox_transform=plt.gcf().transFigure, fontsize=15)

    plt.xlabel(xlab, fontsize=FONTSIZE_MATH if x_is_math else FONTSIZE_TEXT)
    plt.ylabel(ylab, fontsize=FONTSIZE_MATH if y_is_math else FONTSIZE_TEXT)
    plt.tight_layout()

def use_sci_nota(axis, usesci=True):
    ax = plt.gca()

    if axis == 'x':
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        if usesci:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        if usesci:
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))



import csv
def write_to_file(sth, filename='test'):
    with open(filename+'.txt','w') as fw:
        writer = csv.writer(fw,delimiter='\t')
        writer.writerow(sth)
        #tot_cnt = 0
        #for one_Seq in Seqs:
        #    writer.writerow(one_Seq.rec)
        #    tot_cnt += one_Seq.get_cnt()
        #print(filename, 'has tot_cnt:', tot_cnt)
def read_from_file(filename='test'):
    with open(filename+'.txt') as fr:
        reader = csv.reader(fr, delimiter='\t')
        for row in reader:
            return row