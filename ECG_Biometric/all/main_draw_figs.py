

import matplotlib.pyplot as plt
from main_optimCurveFit_mitdb_OO_func import optimCurveFit
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
#clsfy_methods, color_methods, ratios = \
#    clsfy_methods[:inds_used], color_methods[:inds_used], ratios[:inds_used]

strategy = 'NV_data'
#if strategy == 'combine_IDs':
#    ratios = ratios_testV

res_filename = 'draw_figs'
using_file = True

if using_file:
    with open(res_filename) as fr:
        output_all = json.load(fr)
    all_accuBySeg_Vs, all_accuBySeg_Ns, all_train_times = output_all[0], output_all[1], output_all[2]
else:
# fig1: how the rate depends on the data amount of NV?

    #accuBySegs_SVM = [.951, .963, .977, .979, .982]
    #accuBySegs_DL = [.709, .757, .781, .781, .794]
    #method_clsf = 'SVM'
    #ratio_testV = .8

    all_accuBySeg_Vs, all_accuBySeg_Ns, all_train_times = [], [], []
    for method_clsf in clsfy_methods:
        one_accuBySegs_Vs = []
        one_accuBySegs_Ns = []
        one_train_times = []
        for ratio in ratios:
            print '#########################'
            print 'method_clsf:', method_clsf, ', ratios:', ratios

            [accuBySeg_V, accuBySeg_N, train_time] = \
                optimCurveFit(strategy, method_clsf, ratio, NV_type=NV_type)
            one_accuBySegs_Vs.append(accuBySeg_V)
            one_accuBySegs_Ns.append(accuBySeg_N)
            one_train_times.append(train_time)
        all_accuBySeg_Vs.append(one_accuBySegs_Vs)
        all_accuBySeg_Ns.append(one_accuBySegs_Ns)
        all_train_times.append(one_train_times)
    #all_accuBySeg_Vs, all_accuBySeg_Ns, all_train_times = \
    #    np.array(all_accuBySeg_Vs), np.array(all_accuBySeg_Ns), np.array(all_train_times)

    output_all = [all_accuBySeg_Vs, all_accuBySeg_Ns, all_train_times]

    with open(res_filename, 'w') as fw:
        json.dump(output_all, fw)

print all_train_times
print all_accuBySeg_Vs
print all_accuBySeg_Ns

for i in range(len(all_accuBySeg_Vs)):
    plt.scatter(ratios, all_accuBySeg_Vs[i],
                color=color_methods[i], marker=marker_methods[i],
                s=150, label=clsfy_methods[i], facecolors=facecolors_methods[i],
                linewidth = 1.5)
plt.ylim([.5,1.])
mf.config_plot('Ratio of V segments in Test', 'Accuracy of V by segment',
               legend_loc=(.9,.55))
plt.show()

for i in range(len(all_accuBySeg_Ns)):
    plt.scatter(ratios, all_accuBySeg_Ns[i],
                color=color_methods[i], marker=marker_methods[i],
                s=150, label=clsfy_methods[i], facecolors=facecolors_methods[i],
                linewidth = 1.5)
plt.ylim([.4,1.])
mf.config_plot('Ratio of V segments in Test', 'Accuracy of N by segment')#,
               #legend_loc=(.9,.55))
plt.show()

for i in range(len(all_train_times)):
    plt.scatter(ratios, all_train_times[i],
                color=color_methods[i], marker=marker_methods[i],
                s=150, label=clsfy_methods[i], facecolors=facecolors_methods[i],
                linewidth = 1.5)
plt.ylim([-1.,10.])
mf.config_plot('Ratio of V segments in Test', 'Training times',
               legend_loc=(.35,.95))
plt.show()
