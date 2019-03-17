
import numpy as np
import matplotlib.pyplot as plt
import os, glob, wfdb
from biosppy.signals import ecg
import my_funcs as mf
import random

curr_tot_dir = os.path.dirname(__file__)
curr_data_dir = os.path.join(curr_tot_dir + '/ecgiddb/')
os.chdir(curr_data_dir)

Person_ID = 2
curr_ID_str = str(Person_ID)
if Person_ID < 10:
    curr_ID_str = '0' + curr_ID_str
curr_foldername = 'Person_' + curr_ID_str
#print curr_foldername + ' is chosen'

curr_filenames =  glob.glob(curr_foldername + "/*.dat")
curr_recnum = len(curr_filenames)
print 'Number of records for ' + curr_foldername + ': ' + str(curr_recnum) + '.'

channel_ind = 0
SEG_LEN_IN_SEC = 0.5

rec_ind_look = 9 # random.sample(range(curr_recnum), 1)[0]
# TODO: 9-th is a good example of primary filter
# TODO: 19-th is a good example of secondary filter

print 'The ' + str(rec_ind_look) + '-th record is chosen.'
records = []
one_filename = curr_filenames[rec_ind_look] # One rec has many pulses
filename = one_filename[:-4]
sig, fields = wfdb.rdsamp(filename)#, sampto=1000) #, pbdl=0)

fs = fields['fs']
seg_len = SEG_LEN_IN_SEC / (1./fs)

sig = np.array(sig)
sig_use = sig[:,channel_ind]

out = ecg.ecg(signal=sig_use, sampling_rate=fs, show=False)
ecg_filtered, peaks = out[1], out[2]
segs = mf.get_segs(sig_use, peaks, seg_len, fs=fs)

sigs_all = [sig[:,0], sig[:,1], ecg_filtered]
desp_all = ['original signal', 'primary filtered signal', 'secondary filtered signal']
i = 0
for curr_sig in sigs_all:
    fig = plt.figure(figsize=(8,6))
    plt.plot(curr_sig)
    mf.config_plot('time', desp_all[i])
    plt.show()
    curr_segs = mf.get_segs(curr_sig, peaks, seg_len, fs=fs)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,8))
    j = 0
    for ax in axes.flat:
        ax.plot(curr_segs[i])
        #ax.axis('off')
        j += 1
    plt.show()
    i += 1

records += segs



