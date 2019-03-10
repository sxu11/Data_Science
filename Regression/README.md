

Regression might be considered "too easy/trivial", but is really important.

It is the most basic way to model global patterns. On the contrary, NN models the local pattern. 
(insights from Elements of Statistical Learning)

Variations of linear regression by introducing bias (a lot of jargon!):
+ L1 Norm (Lasso)
+ L2 Norm (Ridge)
+ both Norms (Elastic Net)

Diagnostic of linear regression:
1. Residuals vs. Fitted: Shows if residuals 
![Alt text](figures/residual_plot.png?raw=true "Optional Title")



**Linear regression setup**

Notations:
\hat means fitted!

Use notation y_i and x_i=(x_i1, x_i2, ..., x_ip), where i=1, 2, ..., n. 
X = (

    x_1.transpose()
    x_2.transpose()
    ...
    x_n.transpose()
    
)

Assumptions on X:
- Linear relation
- Lack of perfect collinearity
Assumptions on e:
- Gaussian
- Constant variance
- Independence

Usually over-determined system (n>p):
- The goal is to find beta = argmin RSS(beta), where RSS=||y-X beta||^2

Ways of optimizing:
- OLS (ordinary least square): y=X beta
So want to find:
 Y = X * beta,
with dimensions (n,1)=(n,p)*(p,1).
By:
\hat beta = (X.transpose() X).inverse() X.transpose()  Y,
with dimensions ((p,n)*(n,p))^{-1} * (p,n) * (n,1) = (p,1).

- Regression coefficients are Gaussianly distributed:
https://stats.stackexchange.com/questions/44838/how-are-the-standard-errors-of-coefficients-calculated-in-a-regression/44841#44841

Coefficient of determination R^2 = 1 - RSS/TSS, where TSS=\sum(y- \bar y)^2

Complexity: n^2*p + p^3. (fastest matrix inversion uses p^2.3 though)



- Maximum likelihood estimation. 
For any distribution of e. 

P(y|x,beta) * P(x,beta) = P(beta|x,y) P(x,y), 
So P(beta|x,y) \propto P(y|x,beta) = -1/sqrt(pi 2sigma^2) * e^[-(y_i - x_i*beta)/2sigma^2]. 

- Ridge or Lasso. Used for predict, but not for inference. 

- Gradient descent (TODO: Newton's method?)
    - Gradient descent maximizes a function using knowledge of its derivative. 
    - Newton's method, a root finding algorithm, maximizes a function using knowledge of its second derivative. 

- Great summary for interpreting linear regression summary() results:
https://stats.stackexchange.com/questions/5135/interpretation-of-rs-lm-output

- Key Limitations of R-squared
    - R-squared cannot determine whether the coefficient estimates and predictions are biased, which is why you must assess the residual plots.
    - R-squared does not indicate whether a regression model is adequate. You can have a low R-squared value for a good model, or a high R-squared value for a model that does not fit the data!
    - The R-squared in your output is a biased estimate of the population R-squared.

**Logistic regression** is given by:
y_i = 1/[1+exp(-x_i*beta)]

By transformation:
logit(y_i)=log[y_i/(1-y_i)] = x_i * beta

Optimization:
- The regression coefficients are usually estimated using MLE.
- Unlike linear regression with normally distributed residuals, it is not possible to find a closed-form expression for beta that maximize the likelihood function 
- An iterative process must be used instead; for example Newton's method.

Metrics, cross entropy (also called log-loss; asymmetric w.r.t. y_i and \hat y_i!):
- H(y_i, \hat y_i) = -ylog(\hat y)-(1-y)log(1-\hat y)
- J = \sum H(y_i, \hat y_i) / N

- Why does it give (calibrated) probability?
    - sklearn: "as it directly optimizes log-loss". 
    - wiki: Denote the empirical prob of outcome i in the training set is q_i, then likelihood:
        \prod (y_i)^(N q_i)
        so the log-likelihood, divided by N is log[\prod (y_i)^(N q_i)]/N = \sum q_i log y_i = - H(q_i, y_i)
    - So minimizing cross entropy HAPPENS to maximize MLE!!!
        

   
- McFadden's pseudo-R squared:
    - pseudo-R squared = 1 - log(L_c)/log(L_null)
    http://thestatsgeek.com/2014/02/08/r-squared-in-logistic-regression/
    - basically, rho-squared can be interpreted like R2, but don't expect it to be as big. 
    And values from 0.2-0.4 indicate (in McFadden's words) excellent model fit.
    https://stats.stackexchange.com/questions/82105/mcfaddens-pseudo-r2-interpretation
    
    
http://ai.stanford.edu/~ronnyk/2009controlledExperimentsOnTheWebSurvey.pdf