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
- TODO

什么是Batch Norm, why it works, BN的流程是什么 (训练和测试时的区别)
- TODO

common activation functions （sigmoid, tanh, relu, leaky relu） 是什么以及每个的优缺点
- TODO

为什么需要non-linear activation functions
- introduce non-linearity, otherwise just logistic regression

Different optimizers (SGD, RMSprop, Momentum, Adagrad，Adam) 的区别
- TODO

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
- TODO

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

