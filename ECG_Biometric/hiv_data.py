

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def find_ID(one_rec, all_recs):
    for i in range(len(all_recs)):
        if one_rec == all_recs[i]:
            return i

input_all = pd.read_csv('hiv_data.txt', sep='\t')
#print input_all.values[20:48,:]

#dict = {}

all_records = []
all_IDs = []
curr_ID = 0
all_labels = [] # 0 is normal, 1 is patient
group_1 = input_all.values[1:18,1:6].tolist()
for one_record in group_1:
    if one_record[-1] < 10000:
        print one_record[-1]
        continue

    all_records.append(one_record)
    all_labels.append(1)
    all_IDs.append(curr_ID)
    #dict[one_record] = [1, curr_ID]

    curr_ID += 1
print curr_ID

group_2 = input_all.values[20:46,1:6].tolist()
for one_record in group_2:
    all_records.append(one_record)
    all_labels.append(0)
    all_IDs.append(curr_ID)
    curr_ID += 1

TREC_Group1 = input_all.values[1:18,5]
TREC_Group2 =  input_all.values[20:46,5]
fig, ax = plt.subplots()
plt.errorbar(0, TREC_Group1.mean(), yerr=TREC_Group1.std(), color='r')
plt.scatter(0, TREC_Group1.mean(), s=30, color='r')
plt.errorbar(1, TREC_Group2.mean(), yerr=TREC_Group2.std(), color='b')
plt.scatter(1, TREC_Group2.mean(), s=30, color='b')
plt.xlim([-0.5,1.5])
plt.ylabel('TREC', fontsize=24)
ax.set_xticks([0,1])
ax.set_yticks([10000,20000,30000,40000,50000])
ax.set_xticklabels(['HIV', 'non-HIV'])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
plt.tight_layout()
plt.show()
quit()

#print all_labels

X = np.array(all_records)
#print 'correlations:'
#print np.corrcoef([X[:17,0],X[:17,4]])
#print np.corrcoef([X[17:,0],X[17:,4]])
#quit()

to_normalize_data = True
if to_normalize_data:
    for j in range(X.shape[1]):
        max_X = max(X[:,j])
        X[:,j] /= max_X

X_all, y_all = X, np.array(all_labels)
y_ticklabels = ['#CD4','#CD8','Memory CD4','#CD45RA+CD31+','TREC/1e6 CD4']
y_ticklabels = np.array(y_ticklabels)

repeat_times = 1000

problem_ones = []

for repeat_ind in range(repeat_times):
    X_train,X_test,y_train,y_test = train_test_split(
            X_all, y_all, test_size=0.3)#, random_state=42)

    use_method = 'svm'

    if use_method == 'svm':
        from sklearn import svm
        clf = svm.LinearSVC(C=10.)
        clf.fit(X_train, y_train)
        res_pred = clf.predict(X_test)

        to_look_at_wrong_patient = True
        if to_look_at_wrong_patient:
            for i in range(len(X_test)):
                if res_pred[i] != y_test[i]:
                    problem_ones.append(find_ID(X_test[i].tolist(), X_all.tolist()))

        to_get_single_stats = False
        if to_get_single_stats:
            print 'Accuracy:', sum(res_pred==y_test)/float(len(y_test))
            print clf.coef_[0]
            fig, ax = plt.subplots()
            plt.barh(range(len(y_ticklabels)),clf.coef_[0])
            ax.set_yticklabels(y_ticklabels)
            plt.xlabel('coefs from linear SVM',fontsize=20)
            plt.tight_layout()
            plt.show()
    elif use_method == 'xgbc':
        from xgboost import XGBClassifier as XGBC
        from xgboost import plot_importance
        clf = XGBC()
        clf.fit(X_train, y_train)
        res_pred = clf.predict(X_test)
        #print res_pred
        #print y_test

        print 'Accuracy:', sum(res_pred==y_test)/float(len(y_test))

        print(clf.feature_importances_)
        plot_importance(clf)
        ax = plt.gca()
        curr_labels = ax.get_yticklabels()
        curr_inds = []
        for one_label in curr_labels:
            curr_label = one_label.get_text()
            one_ind = int(curr_label[1])
            curr_inds.append(one_ind)
        y_ticklabels = y_ticklabels[curr_inds]
        ax.set_yticklabels(y_ticklabels)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        plt.tight_layout()
        plt.show()

print problem_ones
plt.hist(problem_ones, bins=len(all_IDs), normed=True)
plt.xlabel('Tester ID', fontsize=20)
plt.ylabel('Misclassification ratio', fontsize=20)
plt.show()
#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#print kmeans.labels_.tolist()