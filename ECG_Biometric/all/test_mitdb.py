

import numpy as np
import matplotlib.pyplot as plt
import os, glob, re, wfdb
from biosppy.signals import ecg

import my_funcs as mf

data_set = 'mitdb' # 'ecgiddb', 'mitdb'
channel = 0

curr_tot_dir = os.path.dirname(__file__)
curr_data_dir = os.path.join(curr_tot_dir + '/mitdb/')
os.chdir(curr_data_dir)
filenames = glob.glob('*.dat')

ID = 0
filename = filenames[ID]
regex = re.compile(r'\d+')

curr_ID = regex.findall(filename)[0] # string
sig, fields = wfdb.rdsamp(curr_ID)
print fields

fs = fields['fs']
dt = 1./fs
ann = wfdb.rdann(curr_ID, 'atr')
print ann[0][:5]
print ann[1][:5]

sig_len = 600
ts = np.arange(sig_len)*1./fs
curr_sig = sig[:sig_len, channel]

out = ecg.ecg(signal=curr_sig, sampling_rate=fs, show=False)
ecg_filtered, peaks = out[1], out[2]

seg_len = mf.SEG_LEN_IN_SEC/dt
segs = mf.get_segs(ecg_filtered, peaks, seg_len, fs)
#segs = np.array(segs)
#print segs.shape
#plt.plot(segs[1])
#plt.show()

# Display for looking
def tim2ind(t):
    return int(round(t/dt))

## TODO: define a template fitting function, including parameter preparations
R_loc, R_color = peaks[0], 'blue' # color='blue'
plt.scatter(R_loc*dt, 0, color=R_color)

whole_range_ind = range(len(ecg_filtered)) #range(R_loc-50,R_loc+50,1)
whole_range_tim = [x*dt for x in whole_range_ind]
plt.plot(whole_range_tim,
    ecg_filtered[whole_range_ind])

all_pts = []
# P: R_loc-250 < P_loc < R_loc-150, color='red'
P_left, P_right, P_color = R_loc-tim2ind(.25), R_loc-tim2ind(.15), 'red'
all_pts.append([P_left, P_right, P_color])
# Q: R_loc-60 < Q_loc < R_loc-10, color='green'
Q_left, Q_right, Q_color = R_loc-tim2ind(.06), R_loc-tim2ind(.01), 'green'
all_pts.append([Q_left, Q_right, Q_color])
# S: R_loc+10 < S_loc < R_loc+60, color='yellow'
S_left, S_right, S_color = R_loc+tim2ind(.01), R_loc+tim2ind(.06), 'yellow'
all_pts.append([S_left, S_right, S_color])
# T: R_loc+200 < R_loc+360, color='purple'
T_left, T_right, T_color = R_loc+tim2ind(.20), R_loc+tim2ind(.36), 'purple'
all_pts.append([T_left, T_right, T_color])

for one_pts in all_pts:
    look_range_ind = range(one_pts[0],one_pts[1],1)
    look_range_tim = [x*dt for x in look_range_ind]
    plt.plot(look_range_tim,
            ecg_filtered[look_range_ind],color=one_pts[2])
plt.show()

#plt.plot(ts,ecg_filtered)
#plt.show()

#records, labels, fss = mf.load_data(data_set, channel, num_persons=30, record_time=20)
#records, labels, fss = np.array(records), np.array(labels), np.array(fss)



