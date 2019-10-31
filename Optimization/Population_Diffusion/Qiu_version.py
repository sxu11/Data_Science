
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import Theano_functions as Tf
import Flow_functions as Ff

import pandas as pd

def get_init_paras():
    vlm_embedding = pd.read_csv(
        'Qiu/vl_embedding.csv',
        #"https://www.dropbox.com/s/d30qh2s2yzk7ng0/vl_embedding.csv?dl=1",
        sep = '\t', header = None)#, sep = '\t', header = F)
    delta_embedding = pd.read_csv(
        'Qiu/delta_embedding.csv',
        sep = '\t', header = None
        #"https://www.dropbox.com/s/oxbdmuhduzhu1sj/delta_embedding.csv?dl=1"
    )#,

    #  sep = '\t', header = F)
    ca = pd.read_csv("https://www.dropbox.com/s/yj0j068mnjyc9g6/ca.csv?dl=1", sep = ',', header = 0)
    ra = pd.read_csv("https://www.dropbox.com/s/ltqmo0qsw41irqe/ra.csv?dl=1")
    # print(vlm_embedding.T.values)
    # print(ca.head())


    np.random.seed(0)
    sdin=1.0
    # sample_dat = vlm_embedding.sample(100,  random_state = 0).T.values

    print vlm_embedding.shape, delta_embedding.shape
    dt = 0.1
    vlm_future = vlm_embedding + delta_embedding * dt

    tmp = np.random.choice(vlm_embedding.shape[0], size=100, replace=True, p=None)
    sample_cur = vlm_embedding.iloc[tmp, :].T.values
    sample_fut = vlm_future.iloc[tmp, :].T.values

    velocity_test = Tf.run_all([sample_cur, sample_fut], [0, 1], Tf.relu_pack, sdin=sdin,dtin=0.1, tau=0.7,
                            n1=1,n2=3,lossfun=Tf.sinkhorn_error, Knum=100, eps_base=0.1,scale_base=0.00001)

    w = velocity_test[1].W_matrix
    b = velocity_test[1].b_vec[:,np.newaxis]
    g = velocity_test[1].g_vec[:,np.newaxis]

    return w, b, g
# import pickle
# with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     velocity_test = pickle.load(f)

# x_test = np.linspace(-25,25,num=50)
# y_test = np.linspace(-25,25,num=50)
# plt.figure()
# velocity_test[1].plot([-25,25],[-25,25])
# plt.show()