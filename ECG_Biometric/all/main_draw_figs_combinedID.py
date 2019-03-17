

import matplotlib.pyplot as plt
from main_optimCurveFit_mitdb_OO_func_combineID import optimCurveFit
import numpy as np
import json
import my_funcs as mf

clsfy_methods = ['DTC', 'SVM', 'DL', 'boosting', 'kNN', 'Logit', 'GNB']
color_methods = ['r', 'b', 'g', 'k', 'm', 'c', 'y']
facecolors_methods = ['r', 'none', 'none', 'none', 'none', 'none', 'y']
marker_methods = ['x', 'o', '^', 'd', 's', 'p', '+']

ratios_testV = [.1, .3, .5, .7, .9]
ratios_NfoldV = [.25, .5, .1, .2, .4]
NV_type = 'NVequals'
if NV_type == 'NVequals':
    ratios = ratios_testV
elif NV_type == 'fixV':
    ratios = ratios_NfoldV

#inds_used = 2
#clsfy_methods, color_methods = \
#    [clsfy_methods[inds_used]], [color_methods[inds_used]]

strategy = 'combine_IDs'
#if strategy == 'combine_IDs':
#    ratios = ratios_testV

res_filename = 'draw_figs_combineID_4'
using_file = True

if using_file:
    with open(res_filename) as fr:
        output_all = json.load(fr)
    all_accuBySegs, all_train_times = output_all[0], output_all[1]
else:
# fig1: how the rate depends on the data amount of NV?

    #accuBySegs_SVM = [.951, .963, .977, .979, .982]
    #accuBySegs_DL = [.709, .757, .781, .781, .794]
    #method_clsf = 'SVM'
    #ratio_testV = .8

    all_accuBySegs, all_train_times = [], []
    for method_clsf in clsfy_methods:
        #for ratio in ratios:
        print '#########################'
        print 'method_clsf:', method_clsf

        [accuBySegs, train_time] = \
            optimCurveFit(strategy, method_clsf)
        all_accuBySegs.append(accuBySegs)
        all_train_times.append(train_time)
    #all_accuBySeg_Vs, all_accuBySeg_Ns, all_train_times = \
    #    np.array(all_accuBySeg_Vs), np.array(all_accuBySeg_Ns), np.array(all_train_times)

    output_all = [all_accuBySegs, all_train_times]

    with open(res_filename, 'w') as fw:
        json.dump(output_all, fw)

print all_train_times
print all_accuBySegs

                #color=color_methods[i], marker=marker_methods[i],
                #s=150, label=clsfy_methods[i], facecolors=facecolors_methods[i],
                #linewidth = 1.5)
#plt.ylim([.5,1.])
#mf.config_plot('Ratio of V segments in Test', 'Accuracy of V by segment',
#               legend_loc=(.9,.55))

                #color=color_methods[i], marker=marker_methods[i],
                #s=150, label=clsfy_methods[i], facecolors=facecolors_methods[i],
                #linewidth = 1.5)
#plt.ylim([-1.,12.])

re_inds = [6, 5, 0, 2, 3, 4, 1]
clsfy_methods, all_accuBySegs, all_train_times = \
    np.array(clsfy_methods), np.array(all_accuBySegs), np.array(all_train_times)

fig, ax1 = plt.subplots()
#mf.config_plot()
FONTSIZE_MATH = 24
FONTSIZE_TEXT = 22
FONTSIZE_TICK = 16

x = np.array(range(7))
plt.xticks(x, clsfy_methods[re_inds])
ax1.plot(x, all_accuBySegs[re_inds], color='r', linewidth=2)
#ax1.set_ylim([.7,1.])
ax1.set_ylim([.6,1.])
ax1.set_xlabel('Classification methods', fontsize=FONTSIZE_TEXT)
ax1.set_ylabel('Accuracy', fontsize=FONTSIZE_TEXT, color='r')

#mf.use_sci_nota('x',usesci=False)
mf.use_sci_nota('y',usesci=False)

for tl in ax1.get_yticklabels():
    tl.set_color('r')

ax2 = ax1.twinx()
ax2.plot(x, all_train_times[re_inds], color='b', linewidth=2)
#ax2.set_ylim([0,60])
ax2.set_ylim([0,30])
ax2.set_ylabel('Training times', fontsize=FONTSIZE_TEXT, color='b')

for tl in ax2.get_yticklabels():
    tl.set_color('b')
#mf.use_sci_nota('x',usesci=False)
mf.use_sci_nota('y',usesci=False)

plt.tight_layout()
plt.show()
