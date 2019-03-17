

import os, sys
import numpy as np
import mxnet
import my_utils
import pandas as pd


train_folder = os.getcwd() + '/train/'
test_folder = os.getcwd() + '/test/'
#print 'num_trains:', len(os.listdir(train_folder))

##############################

from scipy import misc
img1 = misc.imread(train_folder + '1.png')
print 'img1.shape:', img1.shape

num_trains, num_tests = 1000,10
num_total = num_trains + num_tests

train_imgs = []
for i in range(1,num_trains+1):
    curr_png_file = str(i) + '.png'
    curr_img = misc.imread(train_folder + curr_png_file)
    curr_img = curr_img.transpose()
    train_imgs.append(curr_img)

all_labels = pd.read_csv('trainLabels.csv')['label'].astype('category').cat.codes.values[:num_total].tolist()

#train_labels = pd.read_csv('trainLabels.csv')['label'].values[:100].tolist()

train_labels = all_labels[:num_trains]

train_data = np.array(train_imgs), np.array(train_labels)
batch_size = 10
train_data_DL = my_utils.DataLoader(dataset = train_data,
                                 batch_size = batch_size,
                                 shuffle=False)

test_imgs = [] # TODO
for i in range(num_trains+1, num_total+1):
    curr_img = misc.imread(train_folder + str(i) + '.png')
    curr_img = curr_img.transpose()
    test_imgs.append(curr_img)
test_labels = all_labels[num_trains:num_total]


test_data = np.array(test_imgs), np.array(test_labels)
test_data_DL = my_utils.DataLoader(dataset = test_data,
                                   batch_size = batch_size,
                                   shuffle=False)

# import matplotlib.pyplot as plt
# plt.imshow(img1)
# plt.show()

##############################

from mxnet import gluon
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(256, flatten=True, activation='relu'))
    net.add(gluon.nn.Dense(10))



ctx = my_utils.try_gpu()
net.initialize(ctx=ctx)

# a total of 100 training, and 1 testing


print net.collect_params()

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate':0.1})
my_utils.train(train_data_DL, test_data_DL, net, loss,
            trainer, ctx, num_epochs=5)


