
Hypothesis Testing in Health

Q1. What type of data do I have?
 - Categorical (Mortality rate: death/survival, Compliance with discharge
 instructions: yes/no)
    - Testing proportions
 
 - Numerical (Temperature, Pain scale, Num of visits)
    - Testing mean
    - Testing correlation/regression
    
Q2. How many samples do I have?
- One sample
- 2 samples
- 2-ish samples
    - One sample, 2 measurements (before and after weights w/ a diet,
    BMI and Systolic Blood Pressure)
    - Two samples with natural pairing (mothers and their babies)
- More than 3 samples


Q3. What is the test supposed to do?
- Compare the data
- Seek a relationship ("risk factor"..)


![Alt text](images/Hypothesis_Testing.png?raw=true "Optional Title")



Resources: 
Chi-square test for goodness-of-fit: https://www.youtube.com/watch?v=b3o_hjWKgQw
Chi-square test for Independence: https://www.youtube.com/watch?v=LE3AIyY_cn8


- t test
    - (Student's) t-test for linear regression coef (df=n-2):
        https://stats.stackexchange.com/questions/286179/why-is-a-t-distribution-used-for-hypothesis-testing-a-linear-regression-coeffici
    - Unequal variance: Welch's t-test
        - Equal or unequal sample sizes, equal variance:
            - Used for A/B, (diff of means)/(pooled SD/sqrt(1/n1+1/n2)), df=n1+n2-2
        - Equal or unequal sample sizes, unequal variances:
            - t_stat = (diff of means)/(s1^2/n1 + s2^2/n2), df=super complicated!
            https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_unequal_variances
        - Dependent t-test for paired samples:
            - t_stat = (mean of diffs)/(SD of diffs/sqrt(n))
        

- F test:
    - The 𝐹 is the ratio of two variances (𝑆𝑆𝑅/𝑆𝑆𝐸), the variance explained by the parameters in the model 
    (sum of squares of regression, SSR) and the residual or unexplained variance (sum of squares of error, SSE).
    
    - The F-test of overall significance indicates whether your linear regression model provides a better fit to the 
    data than a model that contains no independent variables.
    
    - An F-test is a type of statistical test that is very flexible. You can use them in a wide variety of settings. 
    F-tests can evaluate multiple model terms simultaneously, which allows them to compare the fits of different 
    linear models. In contrast, t-tests can evaluate just one term at a time.
    
        - The null hypothesis states that the model with no independent variables fits the data as well as your model.
        - The alternative hypothesis says that your model fits the data better than the intercept-only model.
    
    - To use the F-test to determine whether group means are equal, 
        it’s just a matter of including the correct variances in the ratio. 
        In one-way ANOVA, the F-statistic is this ratio:
        - F = variation between sample means / variation within the samples
        
    - Typically, however, the one-way ANOVA is used to test for differences among at least three groups, 
    since the two-group case can be covered by a t-test (Gosset, 1908). - from wiki
        - When there are only two means to compare, the t-test and the F-test are equivalent; 
        the relation between ANOVA and t is given by F = t2.
    
    - Compared to Likelihood ratio test:
        - "The equivalence between likelihood ratio test and F-test for testing
            variance component in a balanced oneway random effects model"
            
- ANOVA
    - Analysis of Variance (ANOVA) is a statistical method used to test differences 
    between two or more means. It may seem odd that the technique is called "Analysis 
    of Variance" rather than "Analysis of Means." As you will see, the name is 
    appropriate because inferences about means are made by analyzing variance.
    - ANOVA for means assume equal variances...
    
- Levene's Test (for unequal variances)
    - ANOVA (|x_ij - \bar x_j|)

    
    
http://blog.minitab.com/blog/adventures-in-statistics-2/understanding-analysis-of-variance-anova-and-the-f-test
https://statisticsbyjim.com/regression/interpret-f-test-overall-significance-regression/


- Run Test for randomness:
    - https://www.youtube.com/watch?v=J7QmQruh8WA
    
    
- Wald Test: Ancestor for all??
    - https://en.wikipedia.org/wiki/Wald_test
    - "Since the parameter 𝛽𝑗 is estimated using Maxiumum Likelihood Estimation, 
    MLE theory tells us that it is asymptotically normal and hence we can use 
    the large sample Wald confidence interval to get the usual 𝛽𝑗 ± 𝑧 𝑆𝐸(𝛽𝑗)"
    - 


- Cheating Test (Randomized ResponseSurvey (RRS)):
    -  (1) the respondent is faced with a sensitive question; 
    - (2) the respondent is then given some randomization device such as a coin; 
    - (3) the respondent flips the coin without showing it to the interviewer. 
    If the coin lands heads, then the respondent answers “yes” to the question. 
    If the coin lands tails, then the respondent answers truthfully to the question; 
    and, after all data has been collected, 
    - (4) one computes the desired probability of a person’s correct response 
    being “yes”, using a certain probability formula. 
    
    - https://lib.bsu.edu/beneficencepress/mathexchange/02-01/londino.pdf

---
https://www.youtube.com/watch?v=UaptUhOushw



