https://yanjin.space/blog/2020/2020305.html

1. What are overfitting/underfiting?
  - Data has 2 parts: signal + noise; goal is to fit signal, not noise
  - Overfitting is fitting the noise, usually by imposing more complicated structure than in data
  - Underfiting is no fitting signal enough, usually by imposing too simplistic
  
What are bias/variance trade off? 
  - bias is deviation from signal
  - variance is 

How to avoid overfitting?
  - Regularization (L1, L2, TODO: others?)
  
Diff of Generative & Discrimitive?
  - k

1.1 Reguarlization:
L1 vs L2, which one is which and difference
- abs vs sq

Lasso/Ridge explain (what are priors respectively）
- todo

Lasso/Ridge derivation
- todo

Why L1 is more sparse than L2?
- because gradient is const

Why regularization works
- easy to overfit 

Why regularization uses L1 L2, not L3, L4..  
- l3 l4 no foundamental diff w/ l2

1.2 Metric:
precision and recall, trade-off

label imbalance, use what metric

classification metric? and why

confusion matrix
AUC explain (the probability of ranking a randomly selected positive sample higher blablabla....)

true positive rate, false positive rate, ROC

what is Log-loss, when to use logloss
- cross-entropy, negative log-likelihood of a logistic model that returns y_pred probabilities for its training data y_true
- p^y (1-p)^(1-y), closeness of correct... across e.g.'s

context realted: ranking design use what metric? recommend?
- todo

1.3 Loss and optimize
用MSE做loss的Logistic Rregression是convex problem吗
- yes

解释并写出MSE的公式, 什么时候用到MSE?
\sum (y_true - y_pred_prob)^2

Linear Regression最小二乘法和MLE关系
- same

什么是relative entropy/crossentropy,  以及K-L divergence 他们intuition
- kl = relative entropy, 1 distr differs from reference entropy
- weighted sum of log ratio, across X?

Logistic Regression的loss是什么, 推导

SVM的loss是什么
- todo

Multiclass Logistic Regression然后问了一个为什么用cross entropy做cost function

Decision Tree split node的时候优化目标是啥

2. DL基础概念类
DNN为什么要有bias term, bias term的intuition是什么

什么是Back Propagation

梯度消失和梯度爆炸是什么，怎么解决

神经网络初始化能不能把weights都initialize成0

DNN和Logistic Regression的区别

你为什么觉得DNN的拟合能力比Logistic Regression强

how to do hyperparameter tuning in DL/ random search, grid search

Deep Learning有哪些预防overfitting的办法

什么是Dropout，why it works，dropout的流程是什么 (训练和测试时的区别)

什么是Batch Norm, why it works, BN的流程是什么 (训练和测试时的区别)

common activation functions （sigmoid, tanh, relu, leaky relu） 是什么以及每个的优缺点

为什么需要non-linear activation functions

Different optimizers (SGD, RMSprop, Momentum, Adagrad，Adam) 的区别

Batch 和 SGD的优缺点, Batch size的影响

learning rate过大过小对于模型的影响

Problem of Plateau, saddle point

When transfer learning makes sense.


3. ML模型类
3.1 Regression:
Linear Regression的基础假设是什么

what will happen when we have correlated variables, how to solve

explain regression coefficient

what is the relationship between minimizing squared error  and maximizing the likelihood

How could you minimize the inter-correlation between variables with Linear Regression?

if the relationship between y and x is no linear, can linear regression solve that

why use interaction variables

3.2 Clustering and EM:
K-means clustering (explain the algorithm in detail; whether it will converge, 收敛到global or local optimums;  how to stop)

EM算法是什么

GMM是什么，和Kmeans的关系

3.3 Decision Tree
How regression/classification DT split nodes?

How to prevent overfitting in DT?

How to do regularization in DT?

3.4 Ensemble Learning
difference between bagging and boosting

gbdt和random forest 区别，pros and cons

explain gbdt/random forest

will random forest help reduce bias or variance/why random forest can help reduce variance

3.5 Generative Model
和Discrimitive模型比起来，Generative 更容易overfitting还是underfitting

Naïve Bayes的原理，基础假设是什么

LDA/QDA是什么，假设是什么

3.6 Logistic Regression
logistic regression和svm的差别 （我想这个主要是想问两者的loss的不同以及输出的不同，一个是概率输出一个是score）

LR大部分面经集中在logloss和regularization，相关的问题在上个帖子有了这里就不重复了。

3.7 其他模型
Explain SVM, 如何引入非线性

Explain PCA

Explain kernel methods, why to use

what kernels do you know

怎么把SVM的output按照概率输出

Explain KNN

!所有模型的pros and cons （最高频的一个问题）

4. 数据处理类
怎么处理imbalanced data

high-dim classification有什么问题，以及如何处理

missing data如何处理

how to do feature selection

how to capture feature interaction

5. implementation 、推导类
写代码实现两层fully connected网络

手写CNN

手写KNN

手写K-means

手写softmax的backpropagation

给一个LSTM network的结构要你计算how many parameters

convolution layer的output size怎么算? 写出公式
       
6.项目经验类
训练好的模型在现实中不work,问你可能的原因

Loss趋于Inf或者NaN的可能的原因

生产和开发时候data发生了一些shift应该如何detect和补救

annotation有限的情況下你要怎麼Train model

假设有个model要放production了但是发现online one important feature missing不能重新train model 你怎么办

7. NLP/RNN相关
LSTM的公式是什么

why use RNN/LSTM

LSTM比RNN好在哪

limitation of RNN

How to solve gradient vanishing in RNN

What is attention, why attention

Language Model的原理，N-Gram Model

What’s CBOW and skip-gram?

什么是Word2Vec， loss function是什么， negative sampling是什么

8. CNN/CV相关
maxpooling， conv layer是什么, 为什么做pooling，为什么用conv lay，什么是equivariant to-translationa, invariant to translation

1x1 filter

什么是skip connection
（楼主没有面任何CV的岗位之前所以基本没收集到什么CV相关的问题）

9. 关于准备考ML 概念的面试的一些建议

1. 如果你简历上提到了一个模型，请确保你对这个模型有着深入全面的了解 （比如很多人可能简历里都提到了XgBoost，但是可能了解并不全面）

举个例子，我简历上提到了Graph Convolutional NN， 我面试的时候就被要求不用包手写一个简单的GCN。

2. 如果job description上提到了某些模型，最好对这些模型也比较熟悉。

3. 对你这个组的domain的相关模型要熟悉。

比如，你面一个明确做NLP的组，那么上述面经就过于基础了。
你或许还要知道 What is BERT， explain the model architecture；what is Transformer model， explain the model architecture；Transformer/BERT 比LSTM好在哪；difference between self attention and traditional attention mechanism；或许你还要知道一些简单的做distill的方法..或许根据组的方向你还要知道ASR, 或者Chat bot等等的方向的一些widely used的模型或者方法。
比如你面一个CTR的组，或许可能你大概至少要稍微了解下wide-and-deep
比如你面一个CV-segment的组，你或许可能要了解DeepMask，U-Net...等等..

你应该不一定需要知道最SOTA的模型，但是知道那些最广为运用的模型或许可能是必要的。这是我的想法，不一定正确。

