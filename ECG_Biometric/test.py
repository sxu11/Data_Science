
from sklearn.linear_model import SGDClassifier
import numpy as np
import csv

from numpy import exp

print 100.**0.001, 10.**0.001
quit()

print 1/exp(200)
quit()

MISSING_VAL = -0

patient_input = []
col_num = 0
with open('patient_data.csv') as fr:
    reader = csv.reader(fr, delimiter=',')
    for row in reader:
        row_pro = []
        for ch in row:
            if ch == 'NaN':
                row_pro.append(MISSING_VAL)
            else:
                row_pro.append(float(ch))

        patient_input.append(row_pro)
        col_num = len(row)

patient_num = len(patient_input)
patient_data = np.array(patient_input)
num_feat = 11

max_pulse_num = int(float(col_num)/(num_feat+1)) # 20
rec_num = patient_num * max_pulse_num
# patient_data = patient_data.reshape(patient_num, int(float(col_num)/(num_feat+1)), num_feat+1)
patient_data = patient_data.reshape(rec_num, num_feat+1)

#valid_data = []
#for i in range(rec_num):
#    if not sum(patient_data[i,:]) == 0:
#        valid_data.append(patient_data[i,:])

patent_inds = [100, 101, 102, 103, 105, 106, 109, 111, 113]
num_rec_patient = []
for one_ind in patent_inds:
    num_rec_patient.append(np.count_nonzero(patient_data[:,-1]==one_ind))

set_training = []
set_testing = []
training_ratio = 0.67
for i in range(patient_num):
    curr_train_num = int(training_ratio * num_rec_patient[i])
    curr_test_num = num_rec_patient[i] - curr_train_num
    for j in range((i)*max_pulse_num, curr_train_num+(i)*max_pulse_num):
        set_training.append(patient_data[j,:].tolist())
    for j in range(curr_train_num+(i)*max_pulse_num, curr_train_num+curr_test_num+(i)*max_pulse_num):
        set_testing.append(patient_data[j,:].tolist())

set_training = np.array(set_training)
set_testing = np.array(set_testing)

clf = SGDClassifier(loss="hinge",penalty='l2')
clf.fit(set_training[:,:-1], set_training[:,-1])

pred_res = clf.predict(set_testing[:,:-1])
print pred_res
real_res = set_testing[:,-1]
print real_res

print sum(pred_res == real_res)/ float(len(pred_res))