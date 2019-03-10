

- Sample Standard Error, by definition, is always equal to 
Sample Standard Deviation / sqrt(n)...

- Negative Binomial: 
    - Number of successes (probability p) until the r-th failure
    - NB(r,p) = C^k_(k+r-1) * (1-p)^r * p^k, support: k=0,1,2,...
    - Poisson(lambda) = lim_{r->infty} NB(r, lambda/(r+lambda))
    - Overdispersed Poisson: "Contagious" events have positively correlated 
    occurrences causing a larger variance than if the occurrences were independent, 
    due to a positive covariance term.
    - when r=1, becomes the geometric distribution (1-p) p^k, 
    (discrete) waiting time in a Bernoulli process
