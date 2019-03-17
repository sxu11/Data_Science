
import my_funcs as mf
import numpy as np
from collections import Counter
import time

start_time = time.time()

print 1!=2

data_set = 'mitdb' # 'ecgiddb', 'mitdb'
channel = 0
records, IDs, fss, annss = mf.load_data(data_set, channel)#, num_persons=60, record_time=20)
fs = fss[0]

records = np.array(records)
IDs = np.array(IDs)
annss = np.array(annss)

for i in range(annss.shape[0]): #
    print i, Counter(annss[i][1])

print time.time() - start_time
