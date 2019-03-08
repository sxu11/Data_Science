
Unsupervised! When data is so imbalanced, assume Gaussian 
(and manually pick features, and inference parameters)..

(Also, when there are so many different types of "abnormal labels";
maybe tomorrow there are new ways of "abnormal"..)

- Train-test-split:
    10000 good engines
    20 flawed engines
    
    Training set: 6000 good
    CV: 2000 good, 10 flawed
    Test: 2000 good, 10 flawed
    
- Assume Gaussian, calculate prob:
    - p(x) = \prod p(x_j;mu_j, sigma^2)
    
    - see whether p(x) < eps

    - Note, it is not calculating "surprise!" 
    (given null hypothesis that "come from noise"..
    which requires integrate probs over "more surprise region")
    
    - There is strong signal, which DETERMINES 
    and MEASUREs the level of flaw. 
    There is nothing to test against..
    
- Non-Gaussian features:
    - Transformto Gaussian (take log1p, sqrt, etc.).
        Reason is the implicit power law-like distribution (birth-death)?!
    
- Error analysis
    - Look at several FN (mostly the case) & FP examples
        - Find new feature for FN. 
        - Creating from old features by domain knowledge (CPU load)/(network traffic)

- Multi-variate Gaussian
    - Take into account correlation (like crafting old features in a systematic way)   
    - P(x;mu,Sigma) = 1/[(2pi)^(n/2) |Sigma|^0.5] * exp[-0.5 (x-mu).transpose() * Sigma^-1 * (x-mu)],
        where |y| is matrix determinant. 
    
    - Parameter fitting: analytic solution, when compared to no corr
        - Inverse more computations
        - must have n > p (else the p-by-p Sigma is invertible)
    
---
Andrew Ng's videos of machine learning, 15.1-15.8