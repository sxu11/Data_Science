

import pywt
import matplotlib.pyplot as plt

ts = [1,2,3,4]
cA, cD = pywt.dwt(ts, 'db1')
plt.plot(cD)
plt.show()