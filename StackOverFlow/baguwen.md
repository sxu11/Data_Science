https://yanjin.space/blog/2020/2020305.html

1. What are overfitting/underfiting?
  - Data has 2 parts: signal + noise; goal is to fit signal, not noise
  - Overfitting is fitting the noise, usually by imposing more complicated structure than in data
  - Underfiting is no fitting signal enough, usually by imposing too simplistic
  
What are bias/variance trade off? 
  - bias is deviation from signal, variance is ... 
  - From Andrew's DL cs 2 wk1 (basic recipe): 
    - Traditional ML: tradeoff
      - High bias? Need to improve training data performance 
        - Bigger network
        - Train longer
      - High variance? Need to improve validation data performance
        - More data
        - Regularization
    - Deep Learning: often not talk about tradeoff much by NN architecture search
      - With more data and more layers today, DL can achieve both (or least doesn't hurt the other) at same time
      - Just need to add more training time or computational power

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
- In andrew's DL course e.g.:
  - big lambda term suppresses w, which makes less neuron
  - in case of tanh activation function, when w is small, activation near linear, overall closer to Logistic regression
- Other ways:
  - Dropouts (randomly remove all links of certain nodes)
    - Intuition: cannot rely on any 1 feature, need to spreat out weights
  - Data augmentation
  - Early stopping: 
    - How: 
      - Plot BOTH Training and Validation errors
      - stop at latter's valley pt
    - 1 downside: 
      - ML usually has 2 orthogonal tasks: do Optimize vs don't Overfit
      - Early stopping tries to do both at same time

Why regularization uses L1 L2, not L3, L4..  
- l3 l4 no foundamental diff w/ l2
- L2 term: lambda / 2m  ||w||_2^2, mostly used
- L1 term: lambda / 2m  ||w||_1, said to bring in sparsity, less used in practice

1.2 Metric:
precision and recall, trade-off

label imbalance, use what metric

classification metric? and why
- precision, recall (population sensitive?)
- for imbalanced data
  - TPR = sensitivity = recall = prob that an actual positive will test positive 
  - ROC curve: TPR vs FPR (= 1 - specificity)
  - precision: when I say you are 1 (conditioned on prediction), you are actually 1; this is population-dependent metric, because in different populations, I (a specific classifier) have different predictions
  - specificity: when you are 0 (conditioned on truth), I actually say 0; population-independent, because truth population are fixed. 

confusion matrix

AUC explain 
- ()the probability of ranking a randomly selected positive sample higher blablabla....)

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
- MLE: assume come from a certain distribution; observed data most probab, by optimizing parameters
- MSE: same when assume normal distribution

什么是relative entropy/crossentropy,  以及K-L divergence 他们intuition
- kl = relative entropy, 1 distr differs from reference entropy
- weighted sum of log ratio, across X?

Logistic Regression's loss is what, derive?
- cross entropy
- TODO

SVM's loss is what
- hinge loss, max(0, 1-p*y)
- SVM gives some punishment to both incorrect predictions and those close to decision boundary

Multiclass Logistic Regression, why use cross entropy as cost function

Decision Tree split node, what is the optimization goal?
- gini index is one of impurity: \sum f_i (1-f_i)

2. DL基础概念类
DNN为什么要有bias term, bias term的intuition是什么
- to shift position of curve to delay/accelerate activation of node

什么是Back Propagation
- use diff as feedback to adjust weights layer by layer, backward

梯度消失和梯度爆炸是什么，怎么解决
- what:
  - n layers, n derivatives were multiplied together
  - if derivatives are small (stop)/large (unstable), we have either
- how TODO:
  - reduce num of layers
  - Gradient Clipping for explode
  - weight initialization, partial solve [article](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)

神经网络初始化能不能把weights都initialize成0
- TODO

DNN和Logistic Regression的区别
- stacked?

你为什么觉得DNN的拟合能力比Logistic Regression强
- complexity, more parameters/degree of freedom

how to do hyperparameter tuning in DL/ 
- random search, grid search
- more advanced: genetic, bayesian (IDK details)

Deep Learning有哪些预防overfitting的办法
- dropout, batchnorm (?)

什么是Dropout，why it works，dropout的流程是什么 (训练和测试时的区别)
- See DL

什么是Batch Norm, why it works, BN的流程是什么 (训练和测试时的区别)
- See DL

common activation functions （sigmoid, tanh, relu, leaky relu） 是什么以及每个的优缺点
- See DL

为什么需要non-linear activation functions
- introduce non-linearity, otherwise just logistic regression

Different optimizers (SGD, RMSprop, Momentum, Adagrad，Adam) 的区别
- See DL

Batch 和 SGD的优缺点, Batch size的影响
- SGD is Batch size 1
- small: more stable, but slow

learning rate过大过小对于模型的影响
- similar to batch size

Problem of Plateau, saddle point
- search space, gradient tends to 0 locally

When transfer learning makes sense.
  - DNN is able to extract different (meaningful or not) levels of representations
  - Some categories share these more abstract levels (on most fundamental level, lines, edges, angles)


3. ML模型类
3.1 Regression:
Linear Regression的基础假设是什么
- noise normal distributed?

what will happen when we have correlated variables, how to solve
- coefficient doesn't make sense
- what to do?

explain regression coefficient
- weight

what is the relationship between minimizing squared error  and maximizing the likelihood
- for normal distribution , the same

How could you minimize the inter-correlation between variables with Linear Regression?
- PCA?

if the relationship between y and x is no linear, can linear regression solve that
- only locally

why use interaction variables
- ...

3.2 Clustering and EM:
K-means clustering (explain the algorithm in detail; whether it will converge, 收敛到global or local optimums;  how to stop)
- TODO: local or global?

EM算法是什么
[sth](https://www.youtube.com/watch?v=REypj2sy_5U) 
- Mixture (Gaussian/Multinomial, soft clustering) model里面用
  - 相比于k-means (hard clustering)
  - 但mean/variance不知道
    - EM: automatically discover all params for the K "sources"
  - chicken & egg problem
  - Steps (1d e.g.):
    - start w/ mu[0],sigma[0] and mu[1],sigma[1]
    - for each pt: P(0|X[i]) does it look like from 0 or 1?
      - like coloring each pt w/ the 0 and 1 colors
      - Pr(X[i]|0) = 1/sqrt(2pi sigma[0]^2) e^(-(x-mu[0])/2sigma[0]^2)
      - zero[i] = Pr(0|X[i]) = Pr(X[i]|0) Pr(0)/(Pr(X[i]|0) Pr(0) + Pr(X[i]|1) Pr(1))
      - one[i] = 1 - zero[i]
      - 新的mu[0] = \sum zero[i] X[i] / \sum zero[i]
      - 新的sigma[0] = \sum zero[i] (X[i]-mu[0])^2 / \sum zero[i] 
    - Could also estimate the priors: 
      - P(0) = \sum zeros[i] / n
      - P(1) = 1 - P(0)
    - adjust mu[0],sigma[0] and mu[1],sigma[1] to fit pts assigned to them
    - iterate until converge
    
    
[sth](https://stats.stackexchange.com/questions/97324/em-algorithm-practice-problem) interesting:  
  0. Setups
    - X[1],...,X[n] are independent exponential RVs w/ rate T. 
      - P(X) = theta e^(-T X)
    - Not directly observable. Only observe if they fall in intervals: 
      - G1[j] = \1{X[j]<1}, G2[j] = \1{1<X[j]<2}, G3[j] = \1{X[j]>2}
  1. Observed data likelihood? 
    - L(T|G) = \prod Pr{X[j]<1}^G1[j] * Pr{1<X[j]<2}^G2[j] * Pr{X[j]>2}^G3[j]
                 = \prod [1-e^(-T)]^G1[j] * [e^(-T)-e^(-2T)]^G2[j] * [e^(-2T)]^G3[j]
    - 其实这个算的就是 Pr(G|T); 根据贝叶斯二者是\propto的关系，且likelihood本来就是相对值，无所谓scale
  
  2. Complete data likelihood? 
    - L(T|X,G) = \prod T e^(-T X[j]) 
    - 这个是 Pr(X,G|T)
  
  3. Predictive density of the latent variable?
    - f(X[J]|G,T) = f_{X,G}(X[j],g) / f_{G}(g), where 
      - nominator = T e^(-T X[j]) 
      - denominator = [1-e^(-T)]^g1[j] [e^(-T)-e^(-2T)]^g2[j] [e^(-2T)]^g3[j]
    - 这个关键，是第j个data pt的 Pr(X[j]|G,T)的density, 假设其G[j]
      - 分子是 Pr(X[j],G[j] | T)=L(T|X,G), 分母是 Pr(G[j] | T) = L(T|G)
    - 这个其实就是T的 likelihood func?!
  
  4. E-step, give:
    - Q(T, T[i]) = nlogT - T \sum E[X[j]|G,T[i]] = nlogT minus 3 terms: 
      - T(\sum g1[j]/(1-e^(-T[i]))) * (1/T[i] - e^(-T[i])(1+1/T[i]))
      - T(\sum g2[j]/e^(-T[i])(1-e^(-T[i]))) * (e^(-T[i])(1+1/T[i]) - e^(-2T[i])(2+1/T[i]))
      - T(\sum g3[j]/e^(-2T[i])) * e^(-2T[i])(2+1/T[i])
    - which together = nlogT - TN[1]A - TN[2]B - TN[3]C, where
      - N[1] = \sum g1[j]
    - Q(T|T[i])定义为expected val of log likelihood func of T, given ...
    
  5. Let dQ / dT = 0, get:
    - T[i+1] = n / (N[1]A + N[2]B + N[3]C)

probability vs likelihood:
- probability: attaches to possible results
  - Possible results are mutually exclusive and exhaustive.
- likelihood: attaches to hypotheses
  - Hypotheses, unlike results, are neither mutually exclusive nor exhaustive.

GMM是什么，和Kmeans的关系
- Kmeans only considers mean, while Gaussian mixture model considers both mean/variance
- TODO: [read](https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/) that includes EM


3.3 Decision Tree
How regression/classification DT split nodes?
- minimize gini impurity

How to prevent overfitting in DT?
- depth

How to do regularization in DT?
- regularization term?

3.4 Ensemble Learning
difference between bagging and boosting
- ...

gbdt和random forest 区别，pros and cons
- gbdt: more likely to overfit, but faster

explain gbdt/random forest

will random forest help reduce bias or variance/why random forest can help reduce variance
- independent

3.5 Generative Model
和Discrimitive模型比起来，Generative 更容易overfitting还是underfitting
- TODO [andrew paper](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)

Naïve Bayes的原理，基础假设是什么
- independent

LDA/QDA是什么，假设是什么
- TODO

3.6 Logistic Regression
logistic regression和svm的差别 （
- 我想这个主要是想问两者的loss的不同以及输出的不同，一个是概率输出一个是score）
- LR大部分面经集中在logloss和regularization，相关的问题在上个帖子有了这里就不重复了。

3.7 其他模型
Explain SVM, 如何引入非线性
- hinge loss, non-boundary pts don't contribute

Explain PCA
- TODO: [andrew explain](https://www.youtube.com/watch?v=rng04VJxUt4)

Explain kernel methods, why to use
- essentially, different weights in locals

what kernels do you know
- Gauss, TODO? [andrew](https://www.youtube.com/watch?v=mTyT-oHoivA)

怎么把SVM的output按照概率输出
- TODO [andrew](https://www.youtube.com/watch?v=hCOIMkcsm_g)

Explain KNN
- ...

!所有模型的pros and cons （最高频的一个问题）
- TODO

4. 数据处理类
怎么处理imbalanced data
- under/over sampling; TODO: other new methods SMOKE

high-dim classification有什么问题，以及如何处理
- ?

missing data如何处理
- depend on if it is missing at random

how to do feature selection
- I tried recursive selection; could also do L1

how to capture feature interaction
- tree

5. implementation 、推导类
写代码实现两层fully connected网络
- TODO

手写CNN
- TODO

手写KNN
- TODO

手写K-means
- TODO

手写softmax的back propagation
1. error func:
  - cross entropy: E(t,p) = -\sum_j t_j log p_j (t as y_true, p as y_pred here)
2. softmax: 
  - for a total of J classes, it outputs as softmax probability p_j = e^{z_j} / \sum_j e^{z_j}
  - here z_j = \sum_i w_{ij} p_i + b, p_i are all outputs from prev layer
  - ... -> ... -> p_i -> p_j
3. goal: calc how to update w_{ij} in the prev layer
  - chain rule: dE / dw_{ij} = (dE / p_j) (d p_j / d z_j) (d z_j / d w_{ij}) 

    a. (dE / d p_j) = -t_j / p_j

    b. (d p_j / d z_j) = (e^{z_j}*\sum_j e^{z_j} - e^{2z_j})/ [\sum_j e^{z_j}]^2 = p_j - p_j^2    
    
    c. (d z_j / d w_{ij}) = p_i
  - together, we have dE / dw_{ij} = -t_j (1-p_j) p_i
  - [WRONG](https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy)! when t_j=0, this weight will never update?!
  - [WRONG](https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy)!! cannot assume  
  - Reason: Notice p_k all has z_J. So dE / dz_j needs full expansion!!
    - Lesson: Chain rule needs to be step by step. Don't skip!
  - Correct: 
    (dE / dz_j other terms) = sum_{k!=j} (dE / dp_k) (dp_k / dz_j)
    
    a'. dE / dp_k = -t_k / p_k
    
    b'. dp_k / dz_j = (0-e^{z_j+z_k})/(\sum_j e^{z_j})^2 = - p_k p_j
    
    (a+b)'.  -t_j(1-p_j) + \sum_{k!=j} t_k * p_j = -t_j + \sum_{k} t_k * p_j = -t_j + p_j
  - together', we have dE / dw_{ij} = p_i (p_j - t_j)

给一个LSTM network的结构要你计算how many parameters
- TODO

convolution layer的output size怎么算? 写出公式
- TODO
  
6.项目经验类
训练好的模型在现实中不work,问你可能的原因
- data distribution; online offline FE different/missing

Loss趋于Inf或者NaN的可能的原因
- exploding

生产和开发时候data发生了一些shift应该如何detect和补救
- retrain

annotation有限的情況下你要怎麼Train model
- semi-supervised? TODO

假设有个model要放production了但是发现online one important feature missing不能重新train model 你怎么办
- ??? calibration??
- use other features to interpolate the feature?

7. NLP/RNN相关
LSTM的公式是什么
- TODO

why use RNN/LSTM
- TODO

LSTM比RNN好在哪
- TODO

limitation of RNN
- TODO

How to solve gradient vanishing in RNN
- TODO

What is attention, why attention
- TODO

Language Model的原理，N-Gram Model
- TODO

What’s CBOW and skip-gram?
- TODO

什么是Word2Vec， loss function是什么， negative sampling是什么
- TODO

8. CNN/CV相关
maxpooling， conv layer是什么, 为什么做pooling，为什么用conv lay，什么是equivariant to-translationa, invariant to translation
- TODO

1x1 filter
- TODO

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


ML design模版 [post1](https://www.1point3acres.com/bbs/thread-573328-1-1.html)
[post2](https://www.1point3acres.com/bbs/interview/google-machine-learning-585908.html)
- 顺序：
  - 什么问题？
    - classification
    - regression
    - relevance/matching/ranking
  - diagram
    - training/testing data, 
    - input representation
    - model (先)
    - output, 
    - evaluation, 
    - optimization (parameter estimation)
  - model先
      - 2-3个常用的，比较优劣势
            - 不要pick太弱智的！
            - 可以画出model的萌图。。
            - 写一下技术要点，比如：
              - DNN的SGD/ADAM
              - LR的log likelihood
              - regularization: L1 L2
  - Input features: 
    - DNN: one-hot -> embedding
  - (老手1) Data:
    - 根据具体的问题，data的solution可以非常creative。甚至Training data和testing有时候不一致
      - e.g. language model方面的问题：decide一个twitter post是什么语言？
      training 可能就用wikipedia，testing则可以收集user data或者platform-specific的data，
      这时候也需要指明testing如何get ground truth(testing label).
    - training + label,  
      - propose可用的data 来源+data format
      - how to preprocess data -> make training data
      - how to build/create label
    - testing + ground truth
  - (老手2) Eval: 重点在metrics
      - 一个是ROC/AUC curve
      - 第二个是domain specific metrics，比如广告就有CTR
      - 第三个是confusion matrix (重点是从它延申出来precision/recall/accuracy等等对你的solution重要的metrics)
      

- 大局：
  - 结构化表达。超过三点，听的人容易lost。
    - 第一句话先说：这个问题可以从2方面来考虑：用户和系统。系统这方面有3(数字)个solution
    - I’m going to talk about a few components..
  - Stress test:
    - 坚持the principle of problem solving: Break down problem to solvable subtasks
      - e.g. "你deliver了一个ML的产品/系统，用户使用以后，汇报系统的accuracy 远低于你自己test 的accuracy，哪些方面可能出问题了？要求不能看log。"
        - 整个系统有两个element：你的ML系统，用户的application。
        - 把大问题break down成两个element自己的问题+element之间衔接的问题.
          - 产品的问题(overfitting，training data coverage，etc)
          - 用户的application的问题(使用产品的domain和develop 产品的domain不一致， 使用方数据的distribution和training data不一致，etc)
          - 用户的问题(没有按照设计的方式来使用系统，measure的方法不对，使用了和开发方不一样的metrics，etc)
  - 每一步都尽快和面试官确认，move on，不耽误时间

- 细节：
  - 会沟通：rephrase问题 + make concrete example
  - 会接话：听出弦外之音的问题，听出面试官的concern
  - 熟练的讲解参数估计，能显示solid的数学背景。讲估计参数可以用哪些optimization的方法(MSE, loglikelihood+GD, SGD-training data太大量, ADAM-sparse input)，比较优劣.
  
  
- 自己的项目
  - 一是和对方相关，
  - 二是技术越新越厉害，越好；但NDA, 又不能说的太细
    - 怎么解决呢？在报告的结尾，给一个rethink，讲两部分
      - 第一，反思如果现在做同样的问题，哪些地方可以提高
      - 第二，概述一下，你在目前的职位做过的工作，使用了哪些最新的技术，这样显得你的skillset在与时俱进
  - 三是显得你水平高的内容（平时收集）
    
  