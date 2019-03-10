

- Sample Standard Error, by definition, is always equal to 
Sample Standard Deviation / sqrt(n)...

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
