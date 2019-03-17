
import wfdb
import pywt
from biosppy.signals import ecg

import matplotlib.pyplot as plt

### segment len: 240ms,
# segment bnds: always 120ms left, 120ms right ###

print("i'm curios")
print('he as a "monkey"')

print int(round(12.8))
quit()

def get_segs(sig, peaks):
    seg_halflen = 120.

    segs = []
    seg_Lbnds = [0] * len(peaks)
    seg_Rbnds = [0] * len(peaks)
    seg_Lbnds[0] = max(0, round(peaks[0]-seg_halflen/1000 * fs))
    for i in range(1,len(peaks)):
        seg_Lbnds[i] = round(peaks[i] - seg_halflen/1000 * fs)
    for i in range(0,len(peaks)-1):
        seg_Rbnds[i] = round(peaks[i] + seg_halflen/1000 * fs)
    seg_Rbnds[-1] = min(round(peaks[-1] + seg_halflen/1000 * fs), max_ind)

    for i in range(len(peaks)):
        segs.append(sig[seg_Lbnds[i]:seg_Rbnds[i]])
    return segs

def dist_sig(sig1, sig2, lvl):
    coeffs1 = pywt.wavedec(sig1, 'db3', level=lvl)
    coeffs2 = pywt.wavedec(sig2, 'db3', level=lvl)
    dist = 0
    tau = 0.1
    for i in range(len(coeffs1)):
        for j in range(len(coeffs1[i])):
            dist += (coeffs1[i][j]-coeffs2[i][j])/max(coeffs1[i][j], coeffs2[i][j], tau)
    return dist

max_ind = 1000
sig, fields = wfdb.rdsamp('Person_01/rec_1', sampto=max_ind, pbdl=0)
sig_person1 = sig[:,0]

#wfdb.plotwfdb(sig, fields)

fs = fields['fs']

out = ecg.ecg(signal=sig_person1, sampling_rate=fs, show=False)
peaks_person1 = out[2]

segs_person1 = get_segs(sig_person1, peaks_person1)

print dist_sig(segs_person1[0], segs_person1[1], 4)


sig, fields = wfdb.rdsamp('Person_02/rec_1', sampto=max_ind, pbdl=0)
sig_person2 = sig[:,0]

fs = fields['fs']

out = ecg.ecg(signal=sig_person2, sampling_rate=fs, show=False)
peaks_person2 = out[2]

segs_person2 = get_segs(sig_person2, peaks_person2)

print dist_sig(segs_person2[0], segs_person2[1], 4)


print dist_sig(segs_person1[0], segs_person2[0], 4)
print dist_sig(segs_person1[0], segs_person2[1], 4)
print dist_sig(segs_person1[1], segs_person2[0], 4)
print dist_sig(segs_person1[1], segs_person2[1], 4)