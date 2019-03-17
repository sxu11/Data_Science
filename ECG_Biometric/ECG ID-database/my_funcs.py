
from scipy.signal import butter, lfilter
import pywt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

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

def get_segs(sig, peaks, seg_len, fs):
    seg_prelen = int(round(seg_len * 180./500))
    seg_prolen = int(round(seg_len * 320./500))

    segs = []
    seg_Lbnds = [0] * len(peaks)
    seg_Rbnds = [0] * len(peaks)


    todo_num = 1000. # should be 1000.
    seg_Lbnds[0] = max(0, int(round(peaks[0]-seg_prelen/todo_num * fs)))
    for i in range(1,len(peaks)):
        seg_Lbnds[i] = int(round(peaks[i] - seg_prelen/todo_num * fs))
    for i in range(0,len(peaks)-1):
        seg_Rbnds[i] = int(round(peaks[i] + seg_prolen/todo_num * fs))
    seg_Rbnds[-1] = min(int(round(peaks[-1] + seg_prolen/todo_num * fs)), len(sig))

    for i in range(len(peaks)):
        curr_seg = sig[seg_Lbnds[i]:seg_Rbnds[i]]
        #curr_seg = butter_bandpass_filter(curr_seg, 0, 30, fs, order=3)

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

        segs.append(curr_seg) # Warning: bnds are shown in float, weird
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

def config_plot(xlab='', ylab='', legend_loc=(0.,0.)):
    FONTSIZE_MATH = 24
    FONTSIZE_TEXT = 22
    FONTSIZE_TICK = 16

    plt.rc('text', usetex=True)
    plt.tick_params(labelsize=FONTSIZE_TICK)

    x_is_math = '$' in xlab
    y_is_math = '$' in ylab

    plt.legend(bbox_to_anchor=(legend_loc),bbox_transform=plt.gcf().transFigure, fontsize=18)

    plt.xlabel(xlab, fontsize=FONTSIZE_MATH if x_is_math else FONTSIZE_TEXT)
    plt.ylabel(ylab, fontsize=FONTSIZE_MATH if y_is_math else FONTSIZE_TEXT)
    plt.tight_layout()

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