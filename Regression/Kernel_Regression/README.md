
Weighted kNN:
Weigh more similar houses more than less similar in list of kNN

Kernel Regression:
Weigh all points
\hat{y_q} = \sum c_qi y_i / (\sum c_qi)
    = \sum Kernel_lambda (distance(x_i, x_q)) * y_i / (\sum Kernel_lambda (distance(x_i, x_q)))
lambda is the bandwidth (like k), MUCH MORE IMPORTANT than choice of kernel!


---
Reference: UW Machine Learning on Coursera
https://www.coursera.org/learn/ml-regression/lecture/5GtJy/from-weighted-k-nn-to-kernel-regression