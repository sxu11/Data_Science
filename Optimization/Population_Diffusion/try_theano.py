

import theano
import theano.tensor as T
import Theano_functions

import numpy as np
import pandas as pd

#
#
# W_mat = pot_out.W_matrix
# b_v = pot_out.b_vec[:, np.newaxis]
# g_v = pot_out.g_vec[:, np.newaxis]


# num_steps = int(time/dt)
# pp = p_samp(init,ns)
# z = rng.randn(num_steps,pp.shape[0],pp.shape[1])*sd
# return self.potin['simulate'](z, pp, W_mat, b_v, g_v, dt, num_steps)

def matrix_cross_prod(X, Y):
    res = (X * Y.T) - (Y * X.T)
    return (res)

vlm_embedding = pd.read_csv(
    'Qiu/vl_embedding.csv',
    #"https://www.dropbox.com/s/d30qh2s2yzk7ng0/vl_embedding.csv?dl=1",
    sep = '\t', header = None)#, sep = '\t', header = F)

print vlm_embedding.head()

delta_embedding = pd.read_csv(
    'Qiu/delta_embedding.csv',
    sep = '\t', header = None
    #"https://www.dropbox.com/s/oxbdmuhduzhu1sj/delta_embedding.csv?dl=1"
)#,

tmp = np.random.choice(vlm_embedding.shape[0], size=100, replace=True, p=None)
sample_X = vlm_embedding.iloc[tmp, :].T.values
sample_f = delta_embedding.iloc[tmp, :].T.values


# relu_pack = Theano_functions.theano_meta_factory(Tsimul_relu,
#                                                  Tdrift_relu,
#                                                  Tpot_relu,
#                                                  'ramp potential')

def get_grad(X_data, f_data, w, b, g):
    print X_data.shape, f_data.shape
    print w.shape, b.shape, g.shape

    X_data = T._shared(X_data)
    f_data = T._shared(f_data)
#
    psi = T.sum(Theano_functions.Tpot_relu(x_p=X_data, w=w, b=b, g=g))
    print 'psi:', psi

    nabla_psi = theano.gradient.grad(psi, X_data)
    print 'nabla_psi:', nabla_psi
#
    D_hat = - (T.dot(f_data, nabla_psi) / T.dot(nabla_psi, nabla_psi) * T.eye(X_data.shape[1]))
    print 'D_hat:', D_hat
#
    Q_hat = - matrix_cross_prod(f_data, nabla_psi) / T.dot(nabla_psi, nabla_psi)
    print 'Q_hat:', Q_hat
    #
#
    f_hat = -(D_hat+Q_hat)*nabla_psi
    print 'f_hat:', f_hat
#
    loss = T.sum(T.square(f_hat - f_data))

#
    return theano.gradient.grad(loss, [w, b, g])

import Qiu_version

use_cached = True

if not use_cached:
    w, b, g = Qiu_version.get_init_paras()
    print w, b, g

    pd.DataFrame(w).to_csv('w.csv', index=False)
    pd.DataFrame(b).to_csv('b.csv', index=False)
    pd.DataFrame(g).to_csv('g.csv', index=False)

else:
    w = (pd.read_csv('w.csv').values)
    b = (pd.read_csv('b.csv').values)
    g = (pd.read_csv('g.csv').values)

# print 'w', w
# print 'b', b
# print 'g', g
res = get_grad(sample_X, sample_f, T._shared(w), T._shared(b), T._shared(g))
print [x.eval(x) for x in res]