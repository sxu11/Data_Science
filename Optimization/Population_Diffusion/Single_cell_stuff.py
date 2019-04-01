

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

import Theano_functions as Tf

'''
Load single cell rnaseq- data
'''

# file_list = ['GSM1599494_ES_d0_main.csv', 'GSM1599497_ES_d2_LIFminus.csv',
#              'GSM1599498_ES_d4_LIFminus.csv', 'GSM1599499_ES_d7_LIFminus.csv']
# expt_names = ['d0','d2','d4','d7']
# times = [0.0,2.0,4.0,7.0]
#
# import pandas as pd
# import numpy as np
#
# data_folderpath = 'GSE65525_RAW/'
#
# table_list = []
# for filein in file_list:
#     table_list.append(pd.read_csv(data_folderpath+filein, header=None))
#
#
# matrix_list = []
# gene_names = table_list[0].values[:,0]
# for table in table_list:
#     matrix_list.append(table.values[:,1:].astype('float32'))
#
#
#
# '''
# Filter for genes
# '''
#
# cell_counts = [matrix.shape[1] for matrix in matrix_list]
#
# def normalize_run(mat):
#     rpm = np.sum(mat,0)/1e6
#     detect_pr = np.sum(mat==0,0)/float(mat.shape[0])
#     return np.log(mat*(np.median(detect_pr)/detect_pr)*1.0/rpm + 1.0)
#
# norm_mat = [normalize_run(matrix) for matrix in matrix_list]
# qt_mat = [np.percentile(norm_in,q=np.linspace(0,100,50),axis=1) for norm_in in norm_mat]
#
# wdiv=np.sum((qt_mat[0]-qt_mat[3])**2,0)
# w_order = np.argsort(-wdiv)
#
# print gene_names[w_order[0:100]]
#
# wid = w_order[9]
#
# yin=[norm_in[wid,:].tolist() for norm_in in norm_mat]
# xin=[[expt_names[i]]*len(yin[i]) for i in xrange(len(yin))]
#
# def flatten(l):
#     return np.array([item for sublist in l for item in sublist])
#
# nzi = np.array(flatten(yin))>0
# sns.violinplot(x=flatten(xin)[nzi],y=flatten(yin)[nzi],bw=0.05)
# plt.title(gene_names[wid])
# plt.show()
#
# wsub = w_order[0:100]
#
#
# '''
# Impute zeroes
# '''
# import numpy as np
# from scipy import linalg
# from numpy import dot
#
# def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6, print_iter=200):
#     """
#     Decompose X to A*Y
#     """
#     eps = 1e-5
#     print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
#     #X = X.toarray()   I am passing in a scipy sparse matrix
#
#     # mask
#     mask = np.sign(X)
#
#     # initial matrices. A is random [0,1] and Y is A\X.
#     rows, columns = X.shape
#     A = np.random.rand(rows, latent_features)
#     A = np.maximum(A, eps)
#
#     Y = linalg.lstsq(A, X)[0]
#     Y = np.maximum(Y, eps)
#
#     masked_X = mask * X
#     X_est_prev = dot(A, Y)
#     for i in range(1, max_iter + 1):
#         # ===== updates =====
#         # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
#         top = dot(masked_X, Y.T)
#         bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
#         A *= top / bottom
#
#         A = np.maximum(A, eps)
#         # print 'A',  np.round(A, 2)
#
#         # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
#         top = dot(A.T, masked_X)
#         bottom = dot(A.T, mask * dot(A, Y)) + eps
#         Y *= top / bottom
#         Y = np.maximum(Y, eps)
#         # print 'Y', np.round(Y, 2)
#
#
#         # ==== evaluation ====
#         if i % print_iter == 0 or i == 1 or i == max_iter:
#             print 'Iteration {}:'.format(i),
#             X_est = dot(A, Y)
#             err = mask * (X_est_prev - X_est)
#             fit_residual = np.sqrt(np.sum(err ** 2))
#             X_est_prev = X_est
#
#             curRes = linalg.norm(mask * (X - X_est), ord='fro')
#             print 'fit residual', np.round(fit_residual, 4),
#             print 'total residual', np.round(curRes, 4)
#             if curRes < error_limit or fit_residual < fit_error_limit:
#                 break
#     return A, Y, dot(A,Y)
#
# np.random.seed(0)
# norm_imputed = [nmf(normin[wsub,:], latent_features = len(wsub)*4, max_iter=500)[2] for normin in norm_mat]
#
# norm_adj = np.mean(norm_imputed[3],1)[:,np.newaxis]
# sub_len = 10
# subvec = np.array([0,9])
#
# cov_mat = np.cov(norm_imputed[3][subvec,:])
# whiten = np.diag(np.diag(cov_mat)**(-0.5))
# unwhiten = np.diag(np.diag(cov_mat)**(0.5))
#
# norm_imputed2 = [np.dot(whiten,(normin - norm_adj)[subvec,:]) for normin in norm_imputed]
#
# idin=0
# yin_imputed=[normin[idin,:].tolist() for normin in norm_imputed2]
# xin=[[expt_names[i]]*len(yin[i]) for i in xrange(len(yin))]
#
# ax = sns.violinplot(x=flatten(xin),y=flatten(yin_imputed),bw=0.05)
# ax.set(xlabel = 'Day',ylabel='Expression of Krt8')
#
# plt.show()
#
# plt.scatter(norm_imputed2[3][0],norm_imputed2[3][1])
#
# train_data = map(norm_imputed2.__getitem__,[0,1,2,3])
# train_times = map(times.__getitem__,[0,1,2,3])
#
# print train_times
#
# print train_data

#
# '''
# Baseline
# '''
# tid = 2
# qsvec = np.std(train_data[tid],axis=1)
#
# def quadratic_flow(v):
#     return -v/(2*qsvec[:,np.newaxis])
#
# np.random.seed(0)
# p_sub=np.random.choice(train_data[0].shape[1],size=5000,replace=True)
# q_in = np.copy(train_data[0][:,p_sub])
# q_out = Tf.euler_maruyama_dist(q_in,quadratic_flow,0.01,train_times[tid],np.sqrt(2))
#
#
# err_out, fval=Tf.error_term(q_out,train_data[tid],0.1)
# print fval


import numpy as np
import pandas as pd

df = pd.read_csv('Qiu/vl_embedding.csv', sep='\t', header=None)
data = np.array([df[0].values, df[1].values])
print data

'''
Run potential learning
'''
sdin=1.0

np.random.seed(0)
parout_2 = Tf.run_all([data, data],
                      [0, 1], Tf.relu_pack, sdin=sdin,
                      dtin=0.1, tau=0.7, n1=5,n2=5,lossfun=Tf.sinkhorn_error, Knum=500,
                      eps_base=0.1,scale_base=0.00001)

plt.show()

# x_test = np.linspace(-8,2,num=50)
# y_test = np.linspace(-3,8,num=50)
# plt.figure()
# Tf.plot_flow_both(x_test,y_test,parout_2[1])
#
# def nonpar_stat_drift(pts,evalpt,k):
#     dist = get_dist(evalpt,pts)
#     topk = [np.argsort(dist[i,:])[:k] for i in xrange(dist.shape[0])]
#     emu = np.vstack([(np.mean(pts[:,topk[i]],1)-evalpt[:,i])/(dist[i,topk[i][k-1]]) for i in xrange(len(topk))])
#     return emu