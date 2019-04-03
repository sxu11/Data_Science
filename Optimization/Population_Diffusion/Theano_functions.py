import random
import numpy as np
import scipy as sp
import math
from sklearn import metrics
from sklearn import svm
from sklearn import manifold
from sklearn.datasets import *
from sklearn.neighbors import NearestNeighbors
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import time
import seaborn as sns

from time import sleep


import os
if os.environ.get('THEANO_FLAGS') is not None:
    del os.environ['THEANO_FLAGS']

'''
GPU test
'''

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')


import numpy
import theano
import theano.tensor as T
rng = numpy.random

'''
Helper fns
'''

def Tilogit(x):
    return 1/(1+T.exp(-x))

def T_relu_dprime(x):
    return -1*Tilogit(x)*(1-Tilogit(x))

def T_relu_prime(x):
    return -1*Tilogit(x)

def T_relu(x):
    return -1*T.log(1+T.exp(x))

def Tdrift_relu_prime(x_p, w, b, g):
    linterm = g*T_relu_dprime(T.dot(w,x_p)+b)
    return T.dot(w.T,linterm)

def Tsimul_relu_prime(z, x_p, w, b, g, dt):
    return x_p + Tdrift_relu_prime(x_p,w,b,g)*dt + T.sqrt(dt)*z

def Tpot_relu_prime(x_p, w, b, g):
    linterm = g*T_relu_prime(T.dot(w,x_p)+b)
    return T.sum(linterm, 0)

def Tdrift_relu(x_p, w, b, g):
    linterm = g*T_relu_prime(T.dot(w,x_p)+b)
    return T.dot(w.T,linterm)

def Tsimul_relu(z, x_p, w, b, g, dt):
    return x_p + Tdrift_relu(x_p,w,b,g)*dt + T.sqrt(dt)*z

def Tpot_relu(x_p, w, b, g):
    linterm = g*T_relu(T.dot(w,x_p)+b)
    return T.sum(linterm, 0)



def Tpot_quad(x_p, w, b, g):
    sqdist = (x_p ** 2).sum(0).reshape((1, x_p.shape[1])) + (w ** 2).sum(1).reshape((w.shape[0], 1)) - 2 * T.dot(w,x_p)
    kernest = T.exp(-sqdist / b)*g
    return T.sum(kernest, 0)

def Tdrift_quad(x_p, w, b, g):
    ksum = T.sum(Tpot_quad(x_p, w, b, g))
    driftterm = theano.gradient.grad(ksum, x_p)
    return driftterm

def Tsimul_quad(z, x_p, w, b, g, dt):
    return x_p + Tdrift_quad(x_p,w,b,g)*dt + T.sqrt(dt)*z


def Tpot_lin(x_p, w, b, g):
    linterm = g*T.dot(w, x_p)
    return T.sum(linterm, 0)

def Tdrift_lin(x_p, w, b, g):
    ksum = T.sum(Tpot_lin(x_p,w,b,g))
    driftterm = theano.gradient.grad(ksum, x_p)
    return driftterm

def Tsimul_lin(z, x_p, w, b, g, dt):
    return x_p + Tdrift_lin(x_p,w,b,g)*dt + T.sqrt(dt)*z


def Tpot_ou(x_p, w, b, g):
    sqdist = (x_p ** 2).sum(0).reshape((1, x_p.shape[1])) + (w ** 2).sum(1).reshape((w.shape[0], 1)) - 2 * T.dot(w,x_p)
    return T.sum(-sqdist * g, 0)

def Tdrift_ou(x_p, w, b, g):
    ksum = T.sum(Tpot_ou(x_p, w, b, g))
    driftterm = theano.gradient.grad(ksum, x_p)
    return driftterm

def Tsimul_ou(z, x_p, w, b, g, dt):
    return x_p + Tdrift_ou(x_p,w,b,g)*dt + T.sqrt(dt)*z


#theano variables
n_steps = T.iscalar('n_steps')
dt = T.fscalar('dt')
xi = T.matrix("xi")
z = T.tensor3("z")
zmat = T.matrix("zmat")
w = T.matrix("w")
b = T.TensorType(dtype='float32',broadcastable=(False,True))('b')
g = T.TensorType(dtype='float32',broadcastable=(False,True))('g')
err = T.matrix("err")

def theano_meta_factory(sim_fn, drift_fn, pot_fn, name):
    return {'potential':pot_factory(pot_fn),
           'drift':drift_factory(drift_fn),
           'trajectory':em_traj_factory(sim_fn),
           'simulate':em_final_factory(sim_fn),
           'backprop':em_lop_factory(sim_fn),
           'potential_grad':em_pot_factory(pot_fn),
           'name':name}

def drift_factory(drift_fn):
    drift_val = drift_fn(xi, w, b, g)
    return theano.function(inputs=[xi, w, b, g], outputs=drift_val, allow_input_downcast=True,on_unused_input='ignore')

def pot_factory(pot_fn):
    pot_val = pot_fn(xi, w, b, g)
    return theano.function(inputs=[xi, w, b, g], outputs=pot_val, allow_input_downcast=True,on_unused_input='ignore')

def em_traj_factory(sim_fn):
    result, updates = theano.scan(fn = sim_fn, sequences = z, outputs_info = xi, non_sequences = [w, b, g, dt], n_steps = n_steps)
    em_traj_fun = theano.function(inputs = [z, xi, w, b, g, dt, n_steps], outputs= result, updates=updates, allow_input_downcast=True,on_unused_input='ignore')
    return em_traj_fun

def em_final_factory(sim_fn):
    result, updates = theano.scan(fn = sim_fn, sequences = z, outputs_info = xi, non_sequences = [w, b, g, dt], n_steps = n_steps)
    res_final = result[-1]
    em_final_fun = theano.function(inputs = [z, xi, w, b, g, dt, n_steps], outputs=res_final, updates=updates, allow_input_downcast=True,on_unused_input='ignore')
    return em_final_fun

def em_lop_factory(sim_fn):
    result, updates = theano.scan(fn = sim_fn, sequences = z, outputs_info = xi, non_sequences = [w, b, g, dt], n_steps = n_steps)
    gradval = theano.gradient.Lop(T.flatten(result[-1]), [w, b, g], T.flatten(err), disconnected_inputs='warn')
    gradfun = theano.function(inputs = [err, z, xi, w, b, g, dt, n_steps], outputs=gradval, updates=updates, allow_input_downcast=True,on_unused_input='ignore')
    return gradfun

def em_pot_factory(pot_fn):
    pot_val = T.sum(pot_fn(xi, w, b, g))
    gradval = theano.gradient.grad(pot_val, [w, b, g], disconnected_inputs='warn')
    potfun = theano.function(inputs = [xi, w, b, g], outputs = pot_val, allow_input_downcast=True,on_unused_input='ignore')
    potgrad = theano.function(inputs = [xi, w, b, g], outputs = gradval, allow_input_downcast=True,on_unused_input='ignore')
    return potgrad, potfun


relu_pack = theano_meta_factory(Tsimul_relu,Tdrift_relu,Tpot_relu, 'ramp potential')
local_pack = theano_meta_factory(Tsimul_quad,Tdrift_quad,Tpot_quad, 'local potential')
logit_pack = theano_meta_factory(Tsimul_relu_prime,Tdrift_relu_prime,Tpot_relu_prime, 'logit potential')
ou_pack = theano_meta_factory(Tsimul_ou, Tdrift_ou, Tpot_ou, 'Orstein-Uhlenbeck potential')
lin_pack = theano_meta_factory(Tsimul_lin, Tdrift_lin, Tpot_lin, 'Linear potential')


'''
Structs
'''
import copy


class observed:
    def __init__(self, p_init, p_out):
        self.p_init = p_init
        self.p_out = p_out


class hyperpars:
    def __init__(self, NS, eps, sd, sdkern, dt, time):
        self.eps = eps
        self.NS = NS
        self.sd = sd
        self.sdkern = sdkern
        self.dt = dt
        self.time = time


class parset:
    def __init__(self, K, D, potin=relu_pack, scale=1, muzero=None):
        if muzero is None:
            muzero = np.zeros(D)
        self.potin = potin
        self.W_matrix = np.random.randn(K, D) * scale
        if 'local' not in potin['name']:
            offset = np.dot(self.W_matrix, muzero)
            self.b_vec = np.random.uniform(low=-1, high=1, size=K) - offset
        else:
            self.b_vec = np.ones(K) * 5.0
        self.g_vec = np.zeros(K)
        self.W_sqsum = np.ones(self.W_matrix.shape)
        self.b_sqsum = np.ones(self.b_vec.shape)
        self.g_sqsum = np.ones(self.g_vec.shape)
        self.fvvec = []
        self.tvec = []
        self.tnow = 0

    def gclip(self, grad, gmax=1e5):
        g_new = []
        for i in xrange(len(grad)):
            vnorm = np.sqrt(np.sum(grad[i] ** 2.0))
            sfactor = max(1, vnorm / gmax)
            g_new.append(np.copy(grad[i] / sfactor))
        return g_new

    def update(self, grad, eps_val, fv, tv, ada=1e-3):
        self.W_sqsum = self.W_sqsum + eps_val * ada * grad[0] ** 2
        self.b_sqsum = self.b_sqsum + eps_val * ada * grad[1] ** 2
        self.g_sqsum = self.g_sqsum + eps_val * ada * grad[2] ** 2
        self.W_matrix = self.W_matrix + eps_val * grad[0] / np.sqrt(self.W_sqsum)
        self.b_vec = self.b_vec + eps_val * grad[1] / np.sqrt(self.b_sqsum)
        self.g_vec = self.g_vec + eps_val * grad[2] / np.sqrt(self.g_sqsum)
        self.fvvec.append(fv)
        self.tnow = self.tnow + tv
        self.tvec.append(self.tnow)

    def reset_ada(self):
        self.W_sqsum = np.ones(self.W_matrix.shape)
        self.b_sqsum = np.ones(self.b_vec.shape)
        self.g_sqsum = np.ones(self.g_vec.shape)

    def copy(self):
        parnew = parset(K=self.b_vec.shape[0], D=self.W_matrix.shape[1], potin=self.potin)
        parnew.W_matrix = np.copy(self.W_matrix)
        parnew.b_vec = np.copy(self.b_vec)
        parnew.g_vec = np.copy(self.g_vec)
        parnew.fvvec = copy.copy(self.fvvec)
        parnew.tvec = copy.copy(self.tvec)
        parnew.tnow = self.tnow
        return parnew

    def plot(self, xpair, ypair):
        xseq = np.linspace(xpair[0], xpair[1], num=50)
        yseq = np.linspace(ypair[0], ypair[1], num=50)
        plot_flow_pot(self.potin, xseq, yseq, self.W_matrix, self.b_vec, self.g_vec)
        plot_flow_par(xseq, yseq, self.potin, self.W_matrix, self.b_vec, self.g_vec)

    def simulate(self, init, ns, time, dt, sd):
        W_mat = self.W_matrix
        b_v = self.b_vec[:, np.newaxis]
        g_v = self.g_vec[:, np.newaxis]
        num_steps = int(time / dt)
        pp = p_samp(init, ns)
        z = rng.randn(num_steps, pp.shape[0], pp.shape[1]) * sd
        return self.potin['simulate'](z, pp, W_mat, b_v, g_v, dt, num_steps)


class observed_list:
    def __init__(self, p_list, t_list):
        self.p_list = p_list
        self.t_list = t_list


def plot_flow_par(x,y,potfun,W,b,g):
    u=np.zeros((x.shape[0],y.shape[0]))
    v=np.zeros((x.shape[0],y.shape[0]))
    nrm=np.zeros((x.shape[0],y.shape[0]))
    for i in xrange(x.shape[0]):
        ptv=np.vstack((np.full(y.shape[0],x[i]),y))
        flowtmp=potfun['drift'](ptv,W,b[:,np.newaxis],g[:,np.newaxis])
        u[:,i]=flowtmp[0,:]
        v[:,i]=flowtmp[1,:]
        nrm[:,i]=np.sqrt(np.sum(flowtmp**2.0,0))
    #plt.quiver(x,y,u,v)
    plt.streamplot(x,y,u,v,density=1.0,linewidth=3*nrm/np.max(nrm))

def plot_flow_pot(pot,x,y,W,b,g):
    z=np.zeros((x.shape[0],y.shape[0]))
    for i in xrange(x.shape[0]):
        ptv=np.vstack((np.full(y.shape[0],x[i]),y))
        flowtmp= pot['potential'](ptv,W,b[:,np.newaxis],g[:,np.newaxis])
        z[:,i]=flowtmp
    plt.pcolormesh(x,y,np.exp(z))
    CS = plt.contour(x,y,z)
    plt.clabel(CS, inline=1, fontsize=10)

'''
Theano helpers
'''

def p_samp(p_in, num_samp):
    repflag = p_in.shape[1] < num_samp
    p_sub=np.random.choice(p_in.shape[1],size=num_samp,replace=repflag)
    return np.copy(p_in[:,p_sub])

def get_grad_logp(parin, samples, pp, burnin, theano_pack, dt, sd=np.sqrt(2)):
    factr = np.shape(samples)[1]/float(np.shape(pp)[1])
    W_mat = parin.W_matrix
    b_v = parin.b_vec[:,np.newaxis]
    g_v = parin.g_vec[:,np.newaxis]
    num_steps = burnin
    z = rng.randn(num_steps,pp.shape[0],pp.shape[1])*sd
    #run chain forward, get result
    result_final = theano_pack['simulate'](z, pp, W_mat, b_v, g_v, dt, num_steps)
    #logp with respect to input samples
    grad_pos = theano_pack['potential_grad'][0](samples,W_mat, b_v, g_v)
    pos_fv = theano_pack['potential_grad'][1](samples,W_mat,b_v,g_v)
    #logp with respect to contrastive divergence smaples
    grad_neg = theano_pack['potential_grad'][0](result_final,W_mat, b_v, g_v)
    neg_fv = theano_pack['potential_grad'][1](result_final,W_mat, b_v, g_v)
    fv_tot = pos_fv - factr*neg_fv
    dW = grad_pos[0]-grad_neg[0]*factr
    db = np.squeeze(grad_pos[1]-grad_neg[1]*factr)
    dg = np.squeeze(grad_pos[2]-grad_neg[2]*factr)
    return [[dW, db, dg],-1*fv_tot, result_final]

def run_logp_theano(parin,samples, niter, stepsize,theano_pack,dt=0.01, burnin=10, ns=500, ctk=True, ada_val=0,sd=np.sqrt(2)):
    for i in xrange(niter):
        pp = p_samp(samples,ns)
        t_start = time.clock()
        gradin, fv_tot, result_final = get_grad_logp(parin, samples, pp, burnin, theano_pack, dt, sd=sd)
        if ctk:
            pp = result_final
        parin.update(gradin,stepsize/np.shape(samples)[1],fv_tot,time.clock()-t_start, ada_val)
    return parin, result_final

def get_grad_marginal(parin,pp,p_target,theano_pack,time,dt,sd,sdkern,lossfun):
    W_mat = parin.W_matrix
    b_v = parin.b_vec[:,np.newaxis]
    g_v = parin.g_vec[:,np.newaxis]
    num_steps = int(time / float(dt))
    z = rng.randn(num_steps,pp.shape[0],pp.shape[1])*sd
    result_final = theano_pack['simulate'](z, pp, W_mat, b_v, g_v, dt, num_steps)
    err_out, fval=lossfun(result_final,p_target,sdkern)
    gall = theano_pack['backprop'](err_out, z, pp, W_mat, b_v, g_v, dt, num_steps)
    gall[1]=np.squeeze(gall[1])
    gall[2]=np.squeeze(gall[2])
    return gall, fval, result_final, err_out

def run_grad_theano(datin,parin,hpars,maxit,theano_pack,lossfun,tau=0,burnin=10,ctk=True,debug=True,ada_val=0):
    debug = False
    # if debug:
    #     f = FloatProgress(min=0, max=maxit)
    #     display(f)
    num_samp = hpars.NS
    for i in xrange(maxit):
        pp = p_samp(datin.p_init, num_samp)
        pneg = pp
        time_start = time.clock()
        gall, fval, result_final, err_out = get_grad_marginal(parin, pp, datin.p_out, theano_pack, hpars.time, hpars.dt, hpars.sd, hpars.sdkern,lossfun)
        if tau is not 0: #entropic regularization below.
            gall_logp, fv_logp, result_logp = get_grad_logp(parin, datin.p_out, pneg, burnin, theano_pack, hpars.dt, hpars.sd)
            if ctk:
                pneg = result_logp
            for j in xrange(3):
                gall[j] = gall[j]*(1-tau) + gall_logp[j]*tau
        parin.update(gall,hpars.eps,fval,time.clock()-time_start,ada_val/num_samp)
        if np.isneginf(fval):
            break
        if debug:
            f.value = i
    if debug:
        print(fval)
        plt.figure(1)
        plt.plot(parin.fvvec)
        plt.figure(2)
        plt.scatter(result_final[0,:],result_final[1,:],c='red')
        plt.scatter(datin.p_out[0,:],datin.p_out[1,:])
        plt.quiver(result_final[0,:],result_final[1,:],err_out[0,:],err_out[1,:])
    return parin

def run_grad_theano_list(datin_list,parin,hpars,maxit,theano_pack,lossfun, tau=0, burnin=10,ctk=True,delta=False,debug=True,ada_val=0):
    debug = False
    # if debug:
    #     f = FloatProgress(min=0, max=maxit)
    #     display(f)
    num_samp = hpars.NS
    dlast = datin_list.p_list[len(datin_list.t_list)-1]
    for i in xrange(maxit):
        pneg = p_samp(dlast, num_samp)
        db = np.zeros(parin.b_vec.shape)
        dg = np.zeros(parin.b_vec.shape)
        dW = np.zeros(parin.W_matrix.shape)
        fv_tmp = 0
        time_start = time.clock()
        for j in xrange(len(datin_list.t_list)-1):
            if not delta:
                t_cur = datin_list.t_list[j+1] - datin_list.t_list[0]
                dat_cur = datin_list.p_list[j+1]
                dat_init = datin_list.p_list[0]
            else:
                t_cur = datin_list.t_list[j+1]-datin_list.t_list[j]
                dat_cur = datin_list.p_list[j+1]
                dat_init = datin_list.p_list[j]
            pp = p_samp(dat_init, num_samp)
            gall, fval, result_final, err_out = get_grad_marginal(parin, pp, dat_cur, theano_pack, t_cur, hpars.dt, hpars.sd, hpars.sdkern,lossfun)
            dW = dW + gall[0]
            db = db + gall[1]
            dg = dg + gall[2]
            fv_tmp = fv_tmp + fval
        gnew = [dW, db, dg]
        if tau is not 0: #entropic regularization below.
            gall_logp, fv_logp, result_logp = get_grad_logp(parin, dlast, pneg, burnin, theano_pack, hpars.dt, hpars.sd)
            if ctk:
                pneg = result_logp
            for j in xrange(3):
                gnew[j] = gnew[j]*(1-tau) + gall_logp[j]*tau
        if np.isneginf(fv_tmp):
            break
        parin.update(gnew,hpars.eps,fv_tmp,time.clock()-time_start,ada_val/num_samp)
        if debug:
            f.value = i
    if debug:
        print(fval)
        plt.figure(1)
        plt.plot(parin.fvvec)
        for j in xrange(len(datin_list.t_list)-1):
            plt.figure(j+2)
            t_cur = datin_list.t_list[j+1]-datin_list.t_list[j]
            dat_cur = datin_list.p_list[j+1]
            dat_init = datin_list.p_list[j]
            W_mat = parin.W_matrix
            b_v = parin.b_vec[:,np.newaxis]
            g_v = parin.g_vec[:,np.newaxis]
            num_steps = int(t_cur / float(hpars.dt))
            z = rng.randn(num_steps,dat_init.shape[0],dat_init.shape[1])*hpars.sd
            result_final = theano_pack['simulate'](z, dat_init, W_mat, b_v, g_v, hpars.dt, num_steps)
            plt.scatter(result_final[0],result_final[1],c='red')
            plt.scatter(dat_cur[0],dat_cur[1])
    return parin


'''
Wasserstein loss stuff
'''
def checkmat(mat):
    is_finite = np.all(np.isfinite(mat))
    is_nontrivial = np.ptp(mat)>1e-5
    return is_finite and is_nontrivial

from sklearn import utils
import hungarian

def wasserstein_error(p_pred, p_true, sdkern):
    ptrue_resamp = p_samp(p_true, p_pred.shape[1])
    distsq = get_dist(p_pred,ptrue_resamp)
    #matching = utils.linear_assignment_._hungarian(distsq)
    distsq[np.isposinf(distsq)]=1e5
    if checkmat(distsq):
        matching = hungarian.lap(distsq)
    else:
        matching = [np.arange(p_pred.shape[1]), np.arange(p_pred.shape[1])]
    #m1=matching[0]
    m1=np.arange(len(matching[0]))
    #m2=matching[1]
    m2=matching[0]
    spts = p_pred[:,m1]
    dlts = ptrue_resamp[:,m2]-spts
    errs = np.sum(dlts**2.0,0)
    return dlts, -1*np.sum(errs)

def get_dist(yt, ytrue):
    ytnorm = np.sum(yt**2,0)
    ytruenorm = np.sum(ytrue**2,0)
    dotprod = np.dot(yt.T,ytrue)
    return np.add.outer(ytnorm,ytruenorm) - 2*dotprod

def sinkhorn(M, lamb, r, c, maxit=100):
    #Mp = np.array(M,dtype=np.float128)
    Mp=M
    K = np.exp(-lamb*(Mp))#-np.min(Mp)))
    rp = np.copy(r)
    cp = np.copy(c)
    for i in xrange(maxit):
        cp = 1.0/np.dot(rp,K)
        rp = 1.0/np.dot(K,cp)
    kn = rp[:,np.newaxis]*K*cp
    return cp, rp, kn#np.dot(np.dot(np.diag(rp),K),np.diag(cp))

def sinkhorn_error(p_pred, p_true, sdkern, rep=0, numit=10):
    if sdkern is None:
        sdkern = 10.0
    ptrue_resamp = p_samp(p_true, p_pred.shape[1])
    distsq = get_dist(p_pred,ptrue_resamp)
    sko = sinkhorn(distsq, sdkern, np.ones(distsq.shape[0]),np.ones(distsq.shape[1]),numit)[2]
    sko = sko / sko.sum(axis=1,keepdims=True)
    if np.all(np.isfinite(sko)):
        targ = np.dot(ptrue_resamp,np.transpose(sko))
        dlts = targ - p_pred
        return dlts, -1*np.sum(dlts**2.0)
    else:
        if rep < 10:
            return sinkhorn_error(p_pred, p_true, sdkern/2.0, rep=rep+1)
        else:
            return p_pred, -float('Inf')

def sinkhorn_hiprec(M, lamb, r, c, maxit=100):
    Mp = np.array(M,dtype=np.float128)
    K = np.exp(-lamb*(Mp-np.min(Mp)))
    rp = np.copy(r)
    cp = np.copy(c)
    for i in xrange(maxit):
        cp = 1.0/np.dot(rp,K)
        rp = 1.0/np.dot(K,cp)
    kn = rp[:,np.newaxis]*K*cp
    return cp, rp, kn#np.dot(np.dot(np.diag(rp),K),np.diag(cp))

def sinkhorn_error_hiprec(p_pred, p_true, sdkern, rep=0, numit=10):
    if sdkern is None:
        sdkern = 100.0
    ptrue_resamp = p_samp(p_true, p_pred.shape[1])
    distsq = get_dist(p_pred,ptrue_resamp)
    sko = sinkhorn_hiprec(distsq, sdkern, np.ones(distsq.shape[0]),np.ones(distsq.shape[1]),numit)[2]
    sko = sko / sko.sum(axis=1,keepdims=True)
    if np.all(np.isfinite(sko)):
        targ = np.dot(ptrue_resamp,np.transpose(sko))
        dlts = targ - p_pred
        return dlts, -1*np.sum(dlts**2.0)
    else:
        if rep < 10:
            return sinkhorn_error(p_pred, p_true, sdkern/2.0, rep=rep+1)
        else:
            return p_pred, -float('Inf')

'''
Autorun script
'''
def rescale_par(par,snew):
    parnew = par.copy()
    parnew.g_vec = np.copy(par.g_vec) * snew**2.0 / 2.0
    return parnew

def pack_sim(parin, pp, t, dt, sd, theano_pack):
    num_steps = int(t / float(dt))
    z = rng.randn(num_steps,pp.shape[0],pp.shape[1])*sd
    return theano_pack['simulate'](z, pp, parin.W_matrix, parin.b_vec, parin.g_vec, dt, num_steps)

def run_all(data_in, time_in,theano_pack,tau=0,sdin = 1.0, Knum=100, dtin=0.01,burnin=100,lossfun=sinkhorn_error, n1=5, n2=10, eps_base=0.01, scale_base=1, debug=True):
    np.random.seed(0)
    data_last = data_in[-1]
    time_last = time_in[-1]
    NS = data_last.shape[1]
    best_err = -1e8
    best_par = None
    bct = 50
    powr = 2.0
    for j in xrange(n1):
        init_par=parset(potin=theano_pack,K=Knum,D=data_last.shape[0],scale=1.0)
        init_par, p_mat = run_logp_theano(init_par,data_last,400,eps_base/float(powr**j),theano_pack,dt=(time_last)/bct, burnin=bct, ns=NS, ctk=False, ada_val=0.0)
        grad, errval = lossfun(p_mat, data_last, None)
        if debug:
            print errval
        if errval > best_err:
            best_par = init_par.copy()
            best_err = errval
    fvbase = -1e8
    best_out = None
    best_eps = None
    ada_2 = 1/100.0
    for j in xrange(n2):
        #h_par= rescale_par(best_par, sdin)
        h_par = best_par.copy()
        epsin = eps_base/(10.0*float(powr**j))*scale_base
        #print epsin
        h_hyp=hyperpars(NS=NS,eps=epsin,sd=sdin,sdkern=None,dt=dtin,time=time_in[1]-time_in[0])
        if len(time_in) is 2:
            h_dat=observed(data_in[0], data_in[1])
            parout = run_grad_theano(h_dat,h_par,h_hyp,100,theano_pack,tau=tau,burnin=burnin,lossfun=lossfun,ada_val=ada_2, debug=False)
        else:
            hl_dat=observed_list(data_in,time_in)
            parout = run_grad_theano_list(hl_dat,h_par,h_hyp,100,theano_pack,tau=tau,burnin=burnin,lossfun=lossfun,ada_val=ada_2, debug=False, delta=True)
        if debug:
            print (parout.fvvec[-1], epsin)
        #pred_output = pack_sim(parout, data_in[0], data_last, )
        if parout.fvvec[-1] > fvbase:
            best_eps = epsin
            best_out = parout.copy()
            fvbase = parout.fvvec[-1]
    #best_out= rescale_par(best_par, sdin)
    best_out2 = best_par.copy()
    epsin = best_eps
    h_hyp=hyperpars(NS=NS,eps=epsin,sd=sdin,sdkern=None,dt=dtin,time=time_in[1]-time_in[0])
    if len(time_in) is 2:
        h_dat=observed(data_in[0], data_in[1])
        best_out2 = run_grad_theano(h_dat,best_out2,h_hyp,500,theano_pack,tau=tau,burnin=burnin,lossfun=lossfun,ada_val=ada_2, debug=debug)
    else:
        hl_dat=observed_list(data_in,time_in)
        best_out2 = run_grad_theano_list(hl_dat,best_out2,h_hyp,500,theano_pack,tau=tau,burnin=burnin,lossfun=lossfun,ada_val=ada_2, debug=debug, delta=True)
    if not np.isfinite(best_out2.fvvec[-1]):
        best_out2 = best_out
    return best_out2, best_par


'''
Plot and simulation related
'''
def euler_maruyama_dist(p, flow, dt, t, sd):
    pp = np.copy(p)
    n = int(t/dt)
    sqrtdt = np.sqrt(dt)
    for i in xrange(n):
        drift = flow(pp)
        pp = pp + drift*dt + np.random.normal(scale=sd,size=p.shape)*sqrtdt
    return pp

def plot_flow(x,y,fun,ladj=5):
    u=np.zeros((x.shape[0],y.shape[0]))
    v=np.zeros((x.shape[0],y.shape[0]))
    nrm=np.zeros((x.shape[0],y.shape[0]))
    for i in xrange(x.shape[0]):
        ptv=np.vstack((np.full(y.shape[0],x[i]),y))
        flowtmp=fun(ptv)
        u[:,i]=flowtmp[0,:]
        v[:,i]=flowtmp[1,:]
        nrm[:,i]=np.sqrt(np.sum(flowtmp**2.0,0))
    plt.streamplot(x,y,u,v,density=1.0,linewidth=ladj*nrm/np.max(nrm))

from scipy.optimize import fminbound
def error_term(yt, ytrue, kern_sig, minv = 1e-4):
    distsq = get_dist(yt,ytrue)
    d=yt.shape[0]
    if kern_sig is None:
        train_size = int(0.2*yt.shape[1])+1
        indices = np.random.permutation(yt.shape[1])
        training_idx, test_idx = indices[:train_size], indices[train_size:]
        training, test = yt[:,training_idx], yt[:,test_idx]
        dist_train = get_dist(training,test)
        spo=fminbound(error_from_dmat, x1=minv, x2=max(np.max(dist_train),4.0*minv)/2.0, args=(dist_train, d), full_output=True)
        kern_sig = spo[0]
    expterm = np.exp(-distsq/(2*kern_sig))/kern_sig**(d/2.0)
    esum = np.sum(expterm,0)
    #print esum.shape
    errweight = expterm/esum
    grad_err = np.zeros(yt.shape)
    for i in xrange(errweight.shape[0]):
        grad_err[:,i]=np.sum(-2*(yt[:,i][:,np.newaxis]-ytrue)/kern_sig*errweight[i,],1)
    return grad_err, np.sum(np.log(esum))

def error_from_dmat(kern_sig, distsq, d):
    expterm = np.exp(-distsq/(2*kern_sig))/kern_sig**(d/2.0)
    fv = -1*np.sum(np.log(np.sum(expterm,0)))
    #print kern_sig, fv
    return fv