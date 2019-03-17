
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

a = [1,2,3]
b = [3,5,6]

nrows = 2
ncols = 1
gs = gridspec.GridSpec(nrows, ncols, hspace=0.2, wspace=0.2)
fig = plt.figure()
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(a,b)
ax2 = fig.add_subplot(gs[1,0])
ax2.plot(b,a)
plt.show()