from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1.0, 2001)
xlow = np.sin(2 * np.pi * 5 * t)
xhigh = np.sin(2 * np.pi * 250 * t)
print 1./5, 1./250

x = xlow + xhigh
#plt.plot(t, x)
#plt.show()
#quit()
plt.plot(t, x)
plt.show()

b, a = signal.butter(8, 0.1, 'high')
y = signal.filtfilt(b, a, x, padlen=150)
plt.plot(t, y)
plt.show()