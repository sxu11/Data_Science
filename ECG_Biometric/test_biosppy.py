
import numpy as np
from biosppy.signals import ecg

# load raw ECG signal
#signal = np.loadtxt('./examples/ecg.txt')
signal = np.loadtxt('100m1.txt')

# process it and plot
out = ecg.ecg(signal=signal, sampling_rate=360., show=True)