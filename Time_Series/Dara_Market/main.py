
'''

Daily minimum temperatures in Melbourne, Australia, 1981-1990
https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

'''

from pandas import Series
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

series = Series.from_csv('data/daily-minimum-temperatures-in-me.csv', header=0)
# print series
# series.plot()
# plt.show()

'''
General rules:
AR (autocorrelation, Y(t)=\sum Y(t-k)) process: ACF Geometric, PACF: Significant till p lags

MA (moving average, Y(t)=\sum xi(t-k)) process: PACF Geometric, ACF: Significant till p lags
e.g. Y(t) = xi(t) + xi(t-1), Y(t-1) = xi(t-1) + xi(t-2), Y(t-2) = xi(t-2) + xi(t-3).
Then Y(t) and Y(t-2) are totally unrelated (0 on ACF). 
To calculate PACF, one regress Y(t) = a0 + a1 * Y(t-1) + a2 * Y(t-2), now a2 cannot be 0!!

ARMA: ACF Geometric, PACF: Geometric



https://www.youtube.com/watch?v=-vSzKfqcTDg&t=285s
'''

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(series)
plt.savefig('results/acf.png')

plot_pacf(series)
plt.savefig('results/pacf.png')