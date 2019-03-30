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

'''
Utility / plotting functions
'''
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

def euler_maruyama_dist(p, flow, dt, t, sd):
    pp = np.copy(p)
    n = int(t/dt)
    sqrtdt = np.sqrt(dt)
    for i in xrange(n):
        drift = flow(pp)
        pp = pp + drift*dt + np.random.normal(scale=sd,size=p.shape)*sqrtdt
    return pp

def euler_maruyama_dist_traj(p, flow, dt, t, sd):
    pp = np.copy(p)
    n = int(t/dt)
    pset = np.zeros((pp.shape[0],pp.shape[1],n))
    sqrtdt = np.sqrt(dt)
    for i in xrange(n):
        drift = flow(pp)
        pp = pp + drift*dt + np.random.normal(scale=sd,size=p.shape)*sqrtdt
        pset[:,:,i]=pp
    return pset

def plot_w(W_matrix, b_vec, g_vec):
    uvw=W_matrix/np.sum(W_matrix**2,1)[:,np.newaxis]
    offsets = uvw*b_vec[:,np.newaxis]
    plt.quiver(offsets[:,0],offsets[:,1],W_matrix[:,0]*g_vec,W_matrix[:,1]*g_vec)

'''
Plotting code for the output
'''
def plot_flow_par(x,y,potfun,W,b,g):
    u=np.zeros((x.shape[0],y.shape[0]))
    v=np.zeros((x.shape[0],y.shape[0]))
    nrm=np.zeros((x.shape[0],y.shape[0]))
    for i in xrange(x.shape[0]):
        ptv=np.vstack((np.full(y.shape[0],x[i]),y))
        flowtmp=drift_fun(potfun,W,b,g,ptv)
        u[:,i]=flowtmp[0,:]
        v[:,i]=flowtmp[1,:]
        nrm[:,i]=np.sqrt(np.sum(flowtmp**2.0,0))
    #plt.quiver(x,y,u,v)
    plt.streamplot(x,y,u,v,density=1.0,linewidth=3*nrm/np.max(nrm))

def plot_flow_pot(pot,x,y,W,b,g):
    z=np.zeros((x.shape[0],y.shape[0]))
    for i in xrange(x.shape[0]):
        ptv=np.vstack((np.full(y.shape[0],x[i]),y))
        flowtmp= np.sum(pot.f(np.dot(W,ptv)+b[:,np.newaxis])*g[:,np.newaxis],0)
        z[:,i]=flowtmp
    plt.pcolormesh(x,y,np.exp(z))
    CS = plt.contour(x,y,z)
    plt.clabel(CS, inline=1, fontsize=10)

def plot_flow_both(x,y,parin):
    plot_flow_pot(parin.potin,x,y,parin.W_matrix,parin.b_vec,parin.g_vec)
    plot_flow_par(x,y,parin.potin,parin.W_matrix,parin.b_vec,parin.g_vec)

'''
Potential function
'''
def ilogit(x):
    return sp.special.expit(x)
    #return 1/(1+np.exp(-x))

class logitPotential:
    """This function defines a sum-of-logits potential"""
    def f(self,x):
        return -1*ilogit(x)
    def fp(self,x):
        lx=ilogit(x)
        return -1*lx*(1-lx)
    def fpp(self,x):
        lx=ilogit(x)
        return -1*(lx*(1-lx)**2-lx**2*(1-lx))

class reluPotential:
    """This function defines a potential as log(1+exp(x))"""
    def f(self,x):
        return -1*np.log(1+np.exp(x))
    def fp(self,x):
        return -1*ilogit(x)
    def fpp(self,x):
        lx = ilogit(x)
        return -1*lx*(1-lx)

class quadraticPotential:
    """This function defines a potential as x**2"""
    def f(self,x):
        return -x**2/2
    def fp(self,x):
        return -x
    def fpp(self,x):
        return np.zeros(x.shape)-1

'''
Backprop-related
Simulating a SDE
'''

def drift_fun(pot,W,b,g,y):
    scalings = pot.fp(np.dot(W,y)+b[:,np.newaxis])*g[:,np.newaxis] #matrix, K by num_samp
    return np.dot(np.transpose(W),scalings)

def drift_fun_single(pot,W,b,g,y):
    scalings = pot.fp(np.dot(W,y)+b[:,np.newaxis])*g[:,np.newaxis] #matrix, K by num_samp
    drift = np.zeros(y.shape)
    for i in xrange(drift.shape[1]):
        drift[:,i]=np.sum(W*scalings[:,i][:,np.newaxis],0)
    return drift

def euler_maruyama_traj(p,num_samp,W_matrix,b_vec,g_vec,dt,time,sd,potfun):
    repflag = p.shape[1] < num_samp
    p_sub=np.random.choice(p.shape[1],size=num_samp,replace=repflag)
    pp = np.copy(p[:,p_sub])
    n = int(time/dt)
    ptraj = np.zeros((p.shape[0],num_samp,n))
    sqrtdt = np.sqrt(dt)
    for i in xrange(n):
        drift = drift_fun(potfun,W_matrix,b_vec,g_vec,pp)
        pp = pp + drift*dt + np.random.normal(scale=sd,size=(p.shape[0],num_samp))*sqrtdt
        ptraj[:,:,i]=pp
    return ptraj

'''
Loss function
'''
def get_dist(yt, ytrue):
    ytnorm = np.sum(yt**2,0)
    ytruenorm = np.sum(ytrue**2,0)
    dotprod = np.dot(yt.T,ytrue)
    return np.add.outer(ytnorm,ytruenorm) - 2*dotprod

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

from scipy.optimize import brent
def optim_dmat(dmat,d):
    spo=sp.optimize.brent(error_from_dmat, args=(dmat, d))

'''
Backprop + Error gradient
'''
"""Given err at time t, yt-dt, produce err at t-dt, and backpropagate"""
def backweight(pot, err, W_matrix,b_vec,g,ytp,dt):
    Wydot = np.dot(W_matrix,ytp)
    Wedot = np.dot(W_matrix,err)
    linterm = Wydot+b_vec
    pplin = pot.fp(linterm)
    pdlin = pot.fpp(linterm)
    scalings = g*pdlin*Wedot
    err_new = err + dt*np.sum(W_matrix*scalings[:,np.newaxis],0)
    dw = dt*(g*pplin)[:,np.newaxis]*err + (dt*g*pdlin)[:,np.newaxis]*W_matrix*np.dot(ytp,err)
    db = dt*g*pdlin*Wedot
    dg = dt*pplin*Wedot
    return [dw, db, dg, err_new]

'''
Old code
'''
#these functions deal with single yi/yt
def weight_deriv(pot,err,W_matrix,b_vec,g,k,yi,dt):
    linterm = np.dot(W_matrix[k,:],yi)+b_vec[k]
    plin = pot.fp(linterm)
    pdlin = pot.fpp(linterm)
    dwk = dt*g[k]*plin*err + dt*g[k]*pdlin*W_matrix[k,:]*np.dot(yi,err)
    dbk = dt*g[k]*pdlin*np.dot(W_matrix[k,:],err)
    dgk = dt*plin*np.dot(W_matrix[k,:],err)
    return [dwk, dbk, dgk]

def backprop_deriv(pot,err,W_matrix, b_vec,g,yt,dt):
    pdlin = pot.fpp(np.dot(W_matrix,yt)+b_vec)
    scalings = g*pdlin*np.dot(W_matrix,err)
    return err + dt*np.sum(W_matrix*scalings[:,np.newaxis],0)

"""Given err at time t, yt-dt, produce w gradients. Takes a single y."""
def weight_deriv_all(pot,err,W_matrix,b_vec,g,yi,dt):
    Wydot = np.dot(W_matrix,yi)
    Wedot = np.dot(W_matrix,err)
    linterm = Wydot+b_vec
    pplin = pot.fp(linterm)
    pdlin = pot.fpp(linterm)
    dw = dt*(g*pplin)[:,np.newaxis]*err + (dt*g*pdlin)[:,np.newaxis]*W_matrix*np.dot(yi,err)
    db = dt*g*pdlin*Wedot
    dg = dt*pplin*Wedot
    return [dw, db, dg]

'''
Gradient descent helpers
'''
def backweight_all(pot,err_top, W_matrix, b_vec, g_vec, traj,dt):
    grad_mat = np.zeros(W_matrix.shape)
    grad_vec = np.zeros(b_vec.shape)
    grad_g = np.zeros(g_vec.shape)
    err_mat = np.zeros(traj.shape)
    for i in xrange(traj.shape[1]):
        err_cur = np.copy(err_top[:,i])
        err_mat[:,i,traj.shape[2]-1]=err_cur
        for t in xrange(traj.shape[2]-1):
            revt = traj.shape[2]-t-2
            dw, db, dg, err_cur = backweight(pot,err_cur, W_matrix, b_vec ,g_vec, traj[:,i,revt],dt)
            grad_mat+=dw
            grad_vec+=db
            grad_g+=dg
            err_mat[:,i,revt]=err_cur
    return grad_mat, grad_vec, grad_g, err_mat

'''
These classes carry the parameters around
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
    def __init__(self, K, D, potin=logitPotential(), scale=1, muzero=None):
        if muzero is None:
            muzero = np.zeros(D)
        self.potin = potin
        self.W_matrix = np.random.randn(K, D) * scale
        offset = np.dot(self.W_matrix, muzero)
        self.b_vec = np.random.uniform(low=-1, high=1, size=K) - offset
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


class observed_list:
    def __init__(self, p_list, t_list):
        self.p_list = p_list
        self.t_list = t_list

from time import sleep

def run_grad(datin,parin,hpars,maxit,debug=True):
    if debug:
        f = FloatProgress(min=0, max=maxit)
        display(f)
    for i in xrange(maxit):
        time_start = time.clock()
        W_mat = parin.W_matrix
        b_v = parin.b_vec
        g_v = parin.g_vec
        emtj=euler_maruyama_traj(datin.p_init,hpars.NS,W_mat,b_v,g_v,hpars.dt,hpars.time,hpars.sd,parin.potin)
        err_out, fval=error_term(emtj[:,:,emtj.shape[2]-1],datin.p_out,hpars.sdkern)
        gall = backweight_all(parin.potin,err_out, W_mat, b_v, g_v, emtj, hpars.dt)
        parin.update(gall,hpars.eps,fval,time.clock()-time_start)
        if debug:
            f.value = i
    if debug:
        print(fval)
        plt.figure(1)
        plt.plot(parin.fvvec)
        plt.figure(2)
        ind = emtj.shape[2]-1
        plt.scatter(emtj[:,:,ind][0,:],emtj[:,:,ind][1,:],c='red')
        plt.scatter(datin.p_out[0,:],datin.p_out[1,:])
        plt.quiver(emtj[:,:,ind][0,:],emtj[:,:,ind][1,:],err_out[0,:],err_out[1,:])
    return parin

def run_grad_list(datin_list,parin,hpars,maxit,debug=True, delta=False, ada_val=1e-3):
    if debug:
        f = FloatProgress(min=0, max=maxit)
        display(f)
    for i in xrange(maxit):
        W_mat = parin.W_matrix
        b_v = parin.b_vec
        g_v = parin.g_vec
        dW = np.zeros(W_mat.shape)
        db = np.zeros(b_v.shape)
        dg = np.zeros(g_v.shape)
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
            emtj=euler_maruyama_traj(dat_init,hpars.NS,W_mat,b_v,g_v,hpars.dt,t_cur,hpars.sd,parin.potin)
            err_out, fval=error_term(emtj[:,:,emtj.shape[2]-1],dat_cur,hpars.sdkern)
            gall = backweight_all(parin.potin,err_out, W_mat, b_v, g_v, emtj, hpars.dt)
            dW = dW + gall[0]
            db = db + gall[1]
            dg = dg + gall[2]
            fv_tmp = fv_tmp + fval
        parin.update([dW, db, dg],hpars.eps,fv_tmp,time.clock()-time_start, ada_val)
        if debug:
            f.value = i
    if debug:
        print(fval)
        plt.figure(1)
        plt.plot(parin.fvvec)
        for j in xrange(len(datin_list.t_list)-1):
            plt.figure(j+2)
            t_cur = datin_list.t_list[j+1]
            dat_cur = datin_list.p_list[j+1]
            dat_init = datin_list.p_list[0]
            emtj=euler_maruyama_traj(dat_init,hpars.NS,W_mat,b_v,g_v,hpars.dt,t_cur,hpars.sd,parin.potin)
            ind = emtj.shape[2]-1
            plt.scatter(emtj[:,:,ind][0,:],emtj[:,:,ind][1,:],c='red')
            plt.scatter(dat_cur[0],dat_cur[1])
            #plt.quiver(emtj[:,:,ind][0,:],emtj[:,:,ind][1,:],err_out[0,:],err_out[1,:])
    return parin

'''
Old code
'''

# def backprop_all(pot,err_top, W_matrix, b_vec, g_vec, traj,dt):
#     err_mat = np.zeros(traj.shape)
#     for i in xrange(traj.shape[1]):
#         err_cur = np.copy(err_top[:,i])
#         err_mat[:,i,traj.shape[2]-1]=err_cur
#         for t in xrange(traj.shape[2]-1):
#             revt = traj.shape[2]-t-2
#             err_cur = backprop_deriv(pot,err_cur, W_matrix, b_vec ,g_vec, traj[:,i,revt],dt)
#             err_mat[:,i,revt]=err_cur
#     return err_mat
#
# def grad_all(pot,err_all, W_matrix, b_vec, g_vec, traj,dt):
#     grad_mat = np.zeros(W_matrix.shape)
#     grad_vec = np.zeros(b_vec.shape)
#     grad_g = np.zeros(g_vec.shape)
#     for i in xrange(traj.shape[1]):
#         for t in xrange(traj.shape[2]-1):
#             dw, db, dg = webight_deriv_all(pot,err_all[:,i,t+1],W_matrix,b_vec,g_vec,traj[:,i,t],dt)
#             grad_mat=grad_mat+dw
#             grad_vec=grad_vec+db
#             grad_g = grad_g+dg
#     return grad_mat, grad_vec, grad_g

'''
Initialization at equilibrium
'''
def logP(pot, W_matrix, b_vec, g_vec, x):
    """x is a matrix of (dim, n_pts), return a vector of length n_pts of logp for each point."""
    return np.sum(pot.f(np.dot(W_matrix,x)+b_vec[:,np.newaxis])*g_vec[:,np.newaxis],0)

def MALA_chain(pot, W_matrix, b_vec, g_vec, state, k, dt, sd, burnin=0):
    ptraj = np.zeros((state.shape[0],state.shape[1],k))
    sqrtdt = np.sqrt(dt)
    acc_sum = 0
    for i in xrange(k+burnin):
        drift = drift_fun(pot, W_matrix, b_vec, g_vec, state)
        state_new = state + drift*dt + np.random.normal(scale=sd,size=state.shape)*sqrtdt
        if i >= burnin:
            drift_new = drift_fun(pot, W_matrix, b_vec, g_vec, state_new)
            lpdiff = logP(pot, W_matrix, b_vec, g_vec, state_new) - logP(pot, W_matrix, b_vec, g_vec, state)
            lq1 = -1.0/(2*dt*sd**2) * np.sum(((state_new-state) - drift*dt)**2,0)
            lq2 = -1.0/(2*dt*sd**2) * np.sum(((state-state_new) - drift_new*dt)**2,0)
            lqdiff = lq1-lq2
            accpr = np.exp(lpdiff - lqdiff)
            accept_ind = np.random.uniform(size=accpr.shape[0]) < accpr
            acc_sum = acc_sum+np.sum(accept_ind)
            state_new[:,np.nonzero(1-accept_ind)[0]] = state[:,np.nonzero(1-accept_ind)[0]]
            ptraj[:,:,i-burnin]=state_new
        state = state_new
    #print acc_sum/float(state.shape[1]*k)
    return ptraj

def MALA_tester():
    W=np.eye(2)
    b_vec=np.zeros(2)
    g_vec=np.ones(2)
    state=np.zeros((2,3))+10
    return MALA_chain(quadraticPotential(),W,b_vec,g_vec,state,1000, 0.1, 1, burnin=100)

def logPGrad(pot, W_matrix, b_vec, g_vec, x, factr=1.0):
    """Derive the logP gradient for a vector of points x of size (dim, n_samples)"""
    Wdot = np.dot(W_matrix,x)
    linterm = Wdot+b_vec[:,np.newaxis] #linterm - size of K (num hidden units) by n_samples
    pplin = pot.fp(linterm)  #size K by n_samples.
    d_base = pplin*g_vec[:,np.newaxis]
    dW = np.dot(d_base, np.transpose(x))*factr
    dg = np.sum(pot.f(linterm),1)*factr
    db = np.sum(d_base,1)*factr
    return dW, dg, db

def logP_cdopt(parin, samples, niter, stepsize, dt=0.01, burnin=10, ns=500, ctk=True):
    p_ind = np.random.randint(0,np.shape(samples)[1],ns)
    p_mat = samples[:,p_ind]
    n_dat = np.shape(samples)[1]
    factr = n_dat/float(ns)
    for i in xrange(niter):
        t_start = time.clock()
        dW, dg, db = logPGrad(parin.potin, parin.W_matrix, parin.b_vec, parin.g_vec, samples, factr=1.0)
        neg_samp = MALA_chain(parin.potin, parin.W_matrix, parin.b_vec, parin.g_vec, p_mat, 1, dt, np.sqrt(2), burnin=burnin)
        if ctk:
            p_mat = neg_samp[:,:,0]
        dW_neg, dg_neg ,db_neg = logPGrad(parin.potin, parin.W_matrix, parin.b_vec, parin.g_vec, neg_samp[:,:,0], factr=factr)
        parin.update([dW-dW_neg, db-db_neg, dg-dg_neg],stepsize/n_dat,0,time.clock()-t_start)
    return parin, neg_samp

'''
Evaluation code
'''
def find_close_t(t,t_list):
    return np.nonzero(np.array(t_list)<t)[0][-1]

def interpolate_t_fitted(parout, h_list, t_list, target_sample, target_t,sd=1,sdkern=0.5,n_samp=5000,delta_t=None,startat=0):
    t_init_ind = startat#find_close_t(target_t, t_list)
    h_init = h_list[t_init_ind]
    t_delta = target_t - t_list[t_init_ind]
    if delta_t is None:
        delta_t = t_delta/float(50)
    W_mat = parout.W_matrix
    b_v = parout.b_vec
    g_v = parout.g_vec
    emtj=euler_maruyama_traj(h_init,n_samp,W_mat,b_v,g_v,delta_t,t_delta,sd,parout.potin)
    err_out, fval=error_term(emtj[:,:,emtj.shape[2]-1],target_sample,sdkern)
    return emtj[:,:,emtj.shape[2]-1], fval

