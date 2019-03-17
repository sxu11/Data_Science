

from scipy.optimize import curve_fit
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from math import pi as PI

from sklearn.cross_validation import train_test_split
from sklearn import svm

import my_funcs as mf

num_Gaussians = 6

def func(x, a1,b1,t1,a2,b2,t2,a3,b3,t3,a4,b4,t4,a5,b5,t5,a6,b6,t6):
    # TODO may need to be divmod(x-t1,PI)[1]

    return a1*exp(-((x-t1)/b1)**2./2.) \
           + a2*exp(-((x-t2)/b2)**2./2.) \
           + a3*exp(-((x-t3)/b3)**2./2.) \
           + a4*exp(-((x-t4)/b4)**2./2.) \
           + a5*exp(-((x-t5)/b5)**2./2.) \
           + a6*exp(-((x-t6)/b6)**2./2.)
    #return a1*exp(-(divmod(x-t1,PI)[1]/b1)**2./2.) \
    #       + a2*exp(-(divmod(x-t2,PI)[1]/b2)**2./2.) \
    #       + a3*exp(-(divmod(x-t3,PI)[1]/b3)**2./2.) \
    #       + a4*exp(-(divmod(x-t4,PI)[1]/b4)**2./2.) \
    #       + a5*exp(-(divmod(x-t5,PI)[1]/b5)**2./2.) \
    #       + a6*exp(-(divmod(x-t6,PI)[1]/b6)**2./2.)

    #for i in range(num_Gaussians):
    #    del_tht = t_s[i] - x
    #    z_res += 2 * a_s[i] * del_tht * exp(-(del_tht/b_s[i])**2./2.)



data_set = 'ecgiddb' # 'ecgiddb', 'mitdb'
channel = 1
records, labels, fss = mf.load_data(data_set, channel, num_persons=30, record_time=20)
fs = 500. # 500 cycles/second

USE_BIOSPPY_FILTERED = True
segs_all, labels_all = mf.get_seg_data(records, labels, fss, USE_BIOSPPY_FILTERED)
segs_all, labels_all = np.array(segs_all), np.array(labels_all)

dt = 1. #1./fs
ts = np.linspace(0,len(segs_all[0])*dt,num=len(segs_all[0]))

print len(segs_all[0])
quit()

t1_lower_bnd = 18.*dt
t1_upper_bnd = 21.*dt
t2_lower_bnd = 75.*dt
t2_upper_bnd = 78.*dt
t3_lower_bnd = 89.*dt
t3_upper_bnd = 92.*dt
t4_lower_bnd = 100.*dt
t4_upper_bnd = 103.*dt
t5_lower_bnd = 160.*dt
t5_upper_bnd = 250.*dt
t6_lower_bnd = 220.*dt
t6_upper_bnd = 250*dt

p0 = [.12, PI/16, -PI/2,   #P
      -.25, PI/16, -PI/16,   #Q
      1.2, PI/16, 0,   #R
      -.25, PI/16, PI/16,   #S
      .12, PI/16, PI/3,   #T
      .12, PI/16, PI*2/3   #U
      ]

#print p0

X_all = []
y_all = []
for i in range(len(segs_all)):
    seg_curr = segs_all[i]
    plt.plot(ts,seg_curr)
    #plt.show()

    #xs = np.linspace(0,250,num=len(seg_curr))

    try:
        popt, pcov = curve_fit(func, ts, seg_curr,
                           bounds=([-np.inf,-np.inf,t1_lower_bnd,
                                   -np.inf,-np.inf,t2_lower_bnd,
                                   -np.inf,-np.inf,t3_lower_bnd,
                                   -np.inf,-np.inf,t4_lower_bnd,
                                   -np.inf,-np.inf,t5_lower_bnd,
                                   -np.inf,-np.inf,t6_lower_bnd],
                                   [np.inf,np.inf,t1_upper_bnd,
                                   np.inf,np.inf,t2_upper_bnd,
                                   np.inf,np.inf,t3_upper_bnd,
                                   np.inf,np.inf,t4_upper_bnd,
                                   np.inf,np.inf,t5_upper_bnd,
                                   np.inf,np.inf,t6_upper_bnd])
                           #p0=p0
                           )
        seg_fitted = func(ts, popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],
                  popt[6],popt[7],popt[8],popt[9],popt[10],popt[11],
                  popt[12],popt[13],popt[14],popt[15],popt[16],popt[17])
        X_all.append(seg_fitted)
        y_all.append(labels_all[i])
    except:
        pass
#print popt
#print pcov


    #plt.plot(ts,seg_fitted)
    #plt.show()

X_all = np.array(X_all)
X_train,X_test,y_train,y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42)
clf = svm.SVC(kernel='rbf', C=10., gamma=0.1)
clf.fit(X_train, y_train)
res_pred = clf.predict(X_test)

res_by_seg = mf.get_corr_ratio(res_pred=res_pred, y_test=y_test, type='by_seg')
res_by_categ = mf.get_corr_ratio(res_pred=res_pred, y_test=y_test, type='by_categ')
one_res = (float(format(res_by_seg, '.3f')), float(format(res_by_categ, '.3f')))
print one_res