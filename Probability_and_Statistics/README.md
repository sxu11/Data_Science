
- Sample Standard Error, by definition, is always equal to 
Sample Standard Deviation / sqrt(n)...

- Poisson Distribution
    - P_pois(k) = lambda^k e^-lambda / k!
    - Can be derived from Binomial P_bino(k) = C^k_n p^k(1-p)^(n-k)
        - It turns out the Poisson distribution is just a special case of 
        the binomial — where the number of trials is large, and 
        the probability of success in any given one is small.

- When measuring the same thing, A has var 1, B has var 2
    - Let C = (A + b B)/(1+b), then C has var (1+2b)/(1+b)^2
    - Take derivation w.r.t. b and let the result be 0

- Beta Distribution:
    - x^(a-1) (1-x)^(b-1) / B(a, b)
    - Beta function: 
        - B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b)
    - Mean: a/(a+b), 
    - Intuition: 
        - Beta distribution can be understood as representing 
            a distribution of probabilities
        - Represents all the possible values of a probability 
            when we don't know what that probability is.
        - An very Bayesian example: Consider guessing a probability p of success.
            - Prior: between 0.21 ~ 0.35, represented by B(81, 219)
            - Reason: mean = 81/(81+219) = 0.27, and pdf plot mostly within (0.2, 0.35)
            - pdf plot: x-axis is the current guess, y-axis is the prob for that guess
            - Update: Beta(a_0+hits, b_0+misses)
            - Me: so a_0 and b_0 are chosen based on initial thought and weight
        - My 2014 PRE's key step:
            - Approximating B(x;a, b) ~ (x^a)/a ~ 1/a
            - Code:  
                - from scipy import stats, special
                - print x**a/a
                - print 1/a
                - print stats.beta.cdf(x, a, b) * special.beta(a,b)

- Negative Binomial Distribution: 
    - Number of successes (probability p) until the r-th failure
    - NB(r,p) = C^k_(k+r-1) * (1-p)^r * p^k, support: k=0,1,2,...
    - Poisson(lambda) = lim_{r->infty} NB(r, lambda/(r+lambda))
    - Overdispersed Poisson: "Contagious" events have positively correlated 
    occurrences causing a larger variance than if the occurrences were independent, 
    due to a positive covariance term.
    - when r=1, becomes the geometric distribution (1-p) p^k, 
    (discrete) waiting time in a Bernoulli process

- Simplest order stat
    - Expected max y of two uniformly sampled variables (x1,x2) in [0,1]
    - For any given Y in [0,1], 
        Prob(y<=Y) = prob(max(x1,x2)<=Y) = prob(x1<=Y) * prob(x2<=Y) = Y^2
        so pdf(y=Y) = 2Y,
    - Desired expectation = \int_0^1 pdf(y=Y) Y dY 
        = \int_0^1 2Y^2 dY = 2/3 Y^3|^1_0 = 2/3
        
        
        
- Bayesian posterior with truncated normal prior
https://math.stackexchange.com/questions/573694/bayesian-posterior-with-truncated-normal-prior

- Conjugate Priors: Beta and Normal.
https://math.mit.edu/~dav/05.dir/class15-slides-all.pdf


- Sum of several exponential distribution is the Gamma distribution

- Gamma function:
    - Gamma(z) = \int_0^inf x^{z-1} e^{-x} dx
    - A smooth curve that connects the points (x, y) given by y = (x − 1)! 
    at the positive integer values for x.
    
- Hypergeometric distribution 
    - A discrete probability distribution that describes the probability of 
    k successes (random draws for which the object drawn has a specified feature) 
    in n draws, without replacement, from a finite population of size 
    N that contains exactly K objects with that feature, wherein each draw is 
    either a success or a failure. In contrast, the binomial distribution 
    describes the probability of k successes in n draws with replacement.
    - pmf: C(K,k) * C(N-k,n-k) / C(N,n)