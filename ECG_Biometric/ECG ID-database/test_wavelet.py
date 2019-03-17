
import pywt
import my_funcs
import matplotlib.pyplot as plt
import numpy as np

x = my_funcs.read_from_file()
#plt.plot(x)
#plt.show()

w = pywt.Wavelet('db8')
coeffs = pywt.wavedec(x, w, level=5)
plt.plot(coeffs[0])
plt.show()

coeffs[0] = np.array([0] * len(coeffs[0]))
y = pywt.waverec(coeffs, w)
plt.plot(y)
plt.show()
quit()

