
# this is for publishing a conference paper

import numpy as np
import glob
import wfdb
import os
from biosppy.signals import ecg
import my_funcs as mf
import matplotlib.pyplot as plt

def load_data(data_name, channel_ind, num_persons = 30, record_time = None):
    curr_tot_dir = os.path.dirname(__file__)

    records, labels, fields = [], [], []

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

            sig, field = wfdb.rdsamp(filename)#, sampto=1000) #, pbdl=0)

            sig = np.array(sig)

            #print fields
            fields.append(field)
            records.append(sig[:,channel_ind])
            labels.append(i)

    os.chdir(curr_tot_dir)
    return records, labels, fields

class Time():
    def __init__(self, dy, mo, yr):
        self.mo = mo
        self.dy = dy
        self.yr = yr
    def get_time_str(self):
        return '{}.{}.{}'.format(self.mo, self.dy, self.yr)
    def get_time_abs(self):
        return self.mo*30 + self.dy + self.yr*365
    def set_person_ID(self, ID):
        self.ID = ID


class ECG():
    def __init__(self, record, ID, field):
        self.record = record
        self.ID = ID
        self.field = field

        self.fs = field['fs']

        curr_date = field['comments'][2]
        [curr_dy, curr_mo, curr_yr] = [curr_date[-10:-8], curr_date[-7:-5], curr_date[-4:]]
        self.time = Time(int(curr_dy), int(curr_mo), int(curr_yr))
        self.time.set_person_ID(self.ID)

        SEG_LEN_IN_SEC = .5
        seg_len = SEG_LEN_IN_SEC / (1./self.fs)
        out = ecg.ecg(signal=self.record, sampling_rate=self.fs, show=False)
        ecg_filtered, peaks = out[1], out[2]

        USE_FILTERED = True
        if USE_FILTERED:
            sig_use = ecg_filtered
        else:
            sig_use = records[i]
        segs = mf.get_segs(sig_use, peaks, seg_len, fs=self.fs)
        self.segs = np.array(segs)


data_set = 'ecgiddb' # 'ecgiddb', 'mitdb'
channel = 1
records, labels, fields = load_data(data_set, channel, num_persons=10, record_time=20)

#num_labels = len(set(labels))
#all_ID_time = []
look_ID = 1
look_times = []

ecgs = []
for i in range(len(labels)): # i is the index for record, not for person
    curr_ecg = ECG(records[i], labels[i], fields[i])
    ecgs.append(curr_ecg)

    if labels[i] == look_ID:
        print curr_ecg.time.get_time_str()
        look_times.append(curr_ecg.time.get_time_abs())

look_times = sorted(look_times)
look_time_init = look_times[0]
look_times_rela = [(look_time - look_time_init) for look_time in look_times]

print(look_times_rela)
