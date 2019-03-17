
import numpy as np
import matplotlib.pyplot as plt
import os, glob, re, wfdb

import my_funcs as mf

data_set = 'mitdb' # 'ecgiddb', 'mitdb'
channel = 1

curr_tot_dir = os.path.dirname(__file__)
curr_data_dir = os.path.join(curr_tot_dir + '/mitdb/')
os.chdir(curr_data_dir)
filenames = glob.glob('*.dat')

ID = 0
filename = filenames[ID]
regex = re.compile(r'\d+')

curr_ID = regex.findall(filename)[0] # string
sig, fields = wfdb.rdsamp(curr_ID)
fs = fields['fs']

ann = wfdb.rdann(curr_ID, 'atr')

ann_inds = ann[0]
ann_mrks = ann[1]

#print ann_mrks

#num_anns_look = len(ann_inds)
#plt.plot(sig[:ann_inds[num_anns_look], channel])
#for i in range(num_anns_look):
#    plt.text(ann_inds[i], 0, ann_mrks[i])
#plt.show()


offset_look = 600
# Look: plot the area around a peak
for i in range(len(ann_inds)): # ann[0]
    #print sig[ann_inds[i]-10:ann_inds[i]+10,channel]
    if ann_mrks[i] == 'V':
        ts = [1./fs*x for x in range(2*offset_look)]
        plt.plot(ts, sig[ann_inds[i]-offset_look:ann_inds[i]+offset_look,channel])
        plt.text(offset_look,0, ann_mrks[i])
        plt.show()

## extract features from all
