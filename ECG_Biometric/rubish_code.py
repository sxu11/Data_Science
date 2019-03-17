
        #templates = out[4]
        #sum_template = [0] * len(templates[0])
        #avg_template = [0] * len(templates[0])
        #for j in range(len(templates[0])): # j is from 0 to 111111, each signal's length
        #    for i in range(len(templates)): # i is from 0 to 23, so 23 signals
        #        sum_template[j] += templates[i][j]
        #    avg_template[j] = sum_template[j]/len(templates)


        #plt.plot(avg_template)
        #plt.show()
        #quit()
        #if i==34 and j == 4:
        #    RR_avg = 0.002*(peaks[-1] - peaks[0])/(len(peaks) - 1)
        #    QTcFra = QT + 0.154*(1-RR_avg)

        #    plt.plot(sig[:,1])
        #    plt.show()
        #    quit()


                #all_seg_avgs_i.append(seg_avg)
        #all_seg_avgs.append(seg_avg)
        #avg_labels.append(i)

        #    PLOT_WITHINPERSON_CMAP = False
#    if PLOT_WITHINPERSON_CMAP:
#        dist_matrix_withinperson = [[dist_sig(seg_avg1, seg_avg2, 4) \
#                                    for seg_avg2 in all_seg_avgs_i] for seg_avg1 in all_seg_avgs_i]
#        plt.imshow(dist_matrix_withinperson, cmap='hot', interpolation='nearest')
#        plt.gca().invert_yaxis()
#        plt.show()
#        quit()

#dist_matrix_interperson = [[dist_sig(seg_avg1, seg_avg2,4) \
#                            for seg_avg2 in all_seg_avgs] for seg_avg1 in all_seg_avgs]
#plt.imshow(dist_matrix_interperson, cmap='hot', interpolation='nearest')
#plt.gca().invert_yaxis()
#plt.show()

#print X_r

#dist_matrix_interperson = [[distance.euclidean(seg_avg1, seg_avg2) \
#                            for seg_avg2 in X_r] for seg_avg1 in X_r]
#plt.imshow(dist_matrix_interperson, cmap='hot', interpolation='nearest')
#plt.gca().invert_yaxis()
#plt.show()