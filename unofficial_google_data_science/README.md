

There are a total of 26 articles there when I accessed the website on 03/05/2019.

I feel that some of these articles tend to be more wordy, though I would prefer some structured 
way of saying things (STAR):

- Situation:

- Task

- Action:

- Results:

But I do understand more introductory words could be more friendly to non-technical readers. 

**0. Welcome to the unofficial Google data science blog** 
August 26, 2015

**1. An introduction to the Poisson bootstrap**
August 26, 2015
by AMIR NAJMI

The bootstrap is a powerful resampling procedure which makes it easy to compute 
the distribution of any statistical estimator. However, doing the standard bootstrap on big data 
(i.e. which won’t fit in the memory of a single computer) can be computationally prohibitive. In 
this post I describe a simple “statistical fix” to the standard bootstrap procedure allowing us to 
compute bootstrap estimates of standard error in a single pass or in parallel.

**2. On procedural and declarative programming in MapReduce**
September 09, 2015
by SEAN GERRISH and AMIR NAJMI

![Alt text](images/Sawzall.png?raw=true "Optional Title")


**3. Causal attribution in an era of big time-series data**
September 23, 2015
by KAY BRODERSEN

CausalImpact:
Like all state-space models, it rests on two simple equations TODO:
y_t = Z_t^T alpha_t + esp_t
alpha_{t+1} = T_t alpha_t + R_t eta_t

**4. Experiment design and modeling for long-term studies in ads** 
October 07, 2015
by HENNING HOHNHOLD, DEIRDRE O'BRIEN, and DIANE TANG

A/B testing has challenges and blind spots, such as:
- the difficulty of identifying suitable metrics that give "works well" 
a measurable meaning. This is essentially the same as finding a truly useful 
objective to optimize.
- capturing long-term user behavior changes that develop over time periods 
exceeding the typical duration of A/B tests, say, over several months rather 
than a few days.
- accounting for effects "orthogonal" to the randomization used in experimentation. 
For example in ads, experiments using cookies (users) as experimental units are 
not suited to capture the impact of a treatment on advertisers or publishers nor 
their reaction to it.

**5. Data scientist as scientist**
October 21, 2015
by NIALL CARDIN, OMKAR MURALIDHARAN, and AMIR NAJMI

"Our post describes how we arrived at recent changes to design principles for the Google search page, and thus highlights aspects of a data scientist’s role which involve practicing the scientific method."

**6. Using Empirical Bayes to approximate posteriors for large "black box" estimators**
November 04, 2015
by OMKAR MURALIDHARAN

- Situation:
Many machine learning applications have some kind of regression at 
their core, so understanding large-scale regression systems is important. 
But doing this can be hard, for reasons not typically encountered in 
problems with smaller or less critical regression systems. 
    - First, systems can be theoretically intractable. Even systems based on well-understood methods usually have custom tweaks to scale or fit the problem better.
    - Second, important systems evolve quickly, since people are constantly trying to improve them. That means any understanding of the system can become out of date.


- Task:
In this post, 
we describe the challenges posed by one problem — how to get approximate 
posteriors — and an approach that we have found useful.
    - Wanted: approximate posteriors (not just pt estimate!)
    
    "Exact posteriors are hard to get, but we can get approximate ones 
    by extending calibration, a standard way to post-process regression 
    predictions. There are a few different methods for calibration, but 
    all are based on the same idea: instead of using t, estimate and 
    use  E(θ|t) . Calibration fixes aggregate bias, which lets us use 
    methods that are efficient but biased. Calibration also scales easily 
    and doesn’t depend on the details of the system producing t  That 
    means it can handle the large, changing systems we have to deal with."
    
    TODO???

- Action: Empirical Bayes posteriors in four easy steps
    - Bin by t. (TODO: related to my PhD thesis?! )
    - Estimate the prior distribution of  θ|t  in each bin using parametric Empirical Bayes. 
    - Smooth across bins and check fits. 
    - Calculate posterior quantities of interest. 

- Results: 
    - Second order calibration is a nice example of how dealing with large, complex, changing regression systems requires a different approach
    - The resulting method has clear limitations, but is scalable, maintainable, and accurate enough to be useful.



**7. How to get a job at Google — as a data scientist**

November 19, 2015

http://www.unofficialgoogledatascience.com/2015/11/how-to-get-job-at-google-as-data.html

(1) Know your stats. 

At least 3 or 4 courses in probability, statistics, or machine learning
stats.stackexchange.com

(2) Get real-world experience.

Collecting your own data, cleaning it, sanity-checking it, and making use of it.

Write a script to pull data from one of Google’s public APIs and write a blog post about what you’ve found.

Use a web scraper to scrape a few hundred thousand web pages and fit some topic models to create a news recommendation engine. 

Write an app for your phone that tracks your usage and analyze that. 

Be creative!

(3) Spend time coding.

scripting languages like Python and SQL

numerical languages like R, Julia, Matlab, or Mathematica

Bonus points for knowing a compiled language like C++ or Java.

(4) Be passionate.

your talent skews toward the engineering side, you may want to pursue the standard software engineer track and ask for a more analytical role — 

if it skews towards numbers, you may want to pursue the quantitative analyst track.

there are other jobs calling for data scientists in Sales Ops, Marketing and People Ops.




**8. Replacing Sawzall — a case study in domain-specific language migration** 
December 04, 2015
by AARON BECKER

Sawmill execution environment vs logs proxy execution environment


**9. Variance and significance in large-scale online services (LSOS)** 
January 14, 2016
by AMIR NAJMI
http://www.unofficialgoogledatascience.com/2016/01/variance-and-significance-in-large.html

- In this post we explore how and why we can be “data-rich but information-poor”.


**10. LSOS experiments: how I learned to stop worrying and love the variability**
February 29, 2016
by AMIR NAJMI
http://www.unofficialgoogledatascience.com/2016/02/lsos-experiments-how-i-learned-to-stop.html




**11. Using random effects models in prediction problems** 
March 31, 2016
by NICHOLAS A. JOHNSON, ALAN ZHAO, KAI YANG, SHENG WU, FRANK O. KUEHNEL, and ALI NASIRI AMINI

"Random effects can be viewed as an application of empirical Bayes, and if we broaden our view 
to that parent technique, we can find theoretical evidence of their superiority over say fixed 
effects models. For example, the James-Stein estimator has an empirical Bayes interpretation"

TODO: revisit Multi-armed bandit coding

- Random Effect Models

TODO: Gamma-Poisson model?

- A Case Study Click-Through-Rate Prediction

- Scalability studies

- Conclusion
    - provide an interpretable decomposition of variance
    - supply predictive posterior distributions that can be used in stochastic optimization when uncertainty estimates are a critical component (e.g. bandit problems).
    - per-segment click-through-rate models demonstrated that random effects models can deliver superior prediction accuracy. (TODO)

**12. Estimating causal effects using geo experiments** 
May 31, 2016
by JOUNI KERMAN, JON VAVER, and JIM KOEHLER

- Measuring the effectiveness of online ad campaigns
    - return on investment, or Return On Ad Spend (ROAS). 
    - the incremental ROAS, or iROAS.

- Structure of a geo experiment

- Designing a geo experiment: power analysis
TODO

- A model for assessing incremental return on ad spend

- Example
TODO: interesting time series!

- Caveats



**13. To Balance or Not to Balance?**
June 30, 2016
By IVAN DIAZ & JOSEPH KELLY

- The Fundamental Problem of Causal Inference

- The Propensity Score Weighted Estimator

- Estimating the Propensity Score

- Simulation Study

TODO: come back to think about data imbalance.

"To conclude, in the absence of subject-matter knowledge supporting the use of parametric 
functional forms for the propensity score and the balancing conditions, predictive accuracy 
should be used to select an estimator among a collection of candidates. This collection may 
include covariate balanced estimators, and should contain flexible data-adaptive methods
 capable of unveiling complex patterns in the data. In particular, we advocate for the use of 
 model stacking methods such as the Super Learner algorithm implemented in the SuperLearner R 
 package."

**14. Mind Your Units**
July 31, 2016
By JEAN STEINER

In summary, there are many different ways to account for the group structure when the experimental unit differs from the unit of observation.

TODO: A small point. Come back later. 


**15. Next generation tools for data science**
August 31, 2016
By DAVID ADAMS
http://www.unofficialgoogledatascience.com/2016/08/next-generation-tools-for-data-science.html

- The ability to manipulate big data is essential to our notion of data science. 
- While MapReduce remains a fundamental tool, 
    - the well-known Mantel-Haenszel estimator cannot be implemented in a single MapReduce
    1. Expressing complex pipelines requires significant boilerplate, separate programs, and the "interface" between stages to be files
    2. Write intermediate results to disk between stages of pipelines is a serious bottleneck and forces the user to hand-optimize the placement of these divisions
    3. Performing exploratory analysis requires reading and writing to disk, which is slow
    4. Expressing streaming pipelines (low-latency and infinite data sources) is not supported
    5. writing multi-stage pipelines are easily to stumple upon, take e.g. trying to measure
    the effects of a randomized experiment on ratios of metrics using Mantel-Haenszel
    
- Apache Spark and Google Cloud Dataflow represent two alternatives as “next generation” data processing frameworks.
TODO

**16. Statistics for Google Sheets**

September 30, 2016
Editor's note: The Google Sheets add-on described in this blog post is no longer supported externally by Google.

By STEVEN L. SCOTT

TODO: try it when have time?




**17. Practical advice for analysis of large, complex data sets**
October 31, 2016
By PATRICK RILEY

“Good Data Analysis.” 24 advices!
- Technical: Ideas and techniques for how to manipulate and examine your data.
    - Look at your distributions
    (my e.g.: histograms of genes, U-shaped)
    
    - Consider the outliers
        - looking at the queries with the lowest click-through rate (CTR) may reveal clicks on elements in the user interface that you are failing to count
        - Looking at queries with the highest CTR may reveal clicks you should not be counting.
        (my e.g.: Very large clones)
        
    - Report noise/confidence
        - Every estimator that you produce should have a notion of your confidence in this estimate attached to it. 
            - Sometimes formal and precise: confidence intervals or credible intervals for estimators, and p-values or Bayes factors for conclusions
            - Other times you will be more loose: For example if a colleague asks you how many queries about frogs we get on Mondays, you might do a quick analysis looking and a couple of Mondays and report “usually something between 10 and 12 million”
        (my e.g.: Whenever claiming "fitting": regression, "different": AUROC, or "correlative" TODO: pearson)
    
    - Look at examples 
        - Look at stratified sampling, so you are not too focussed on the most common cases.
        (e.g. if you are computing Time to Click, make sure you look at examples throughout your distribution, especially the extremes. If you don’t have the right tools/visualization to look at your data, you need to work on those first.
        ) 
        (my e.g.: Multiple weights per encounter)
    
    - Slice your data
        - In analysis of web traffic, we commonly slice along dimensions like mobile vs. desktop, browser, locale, etc.
        - When comparing compare two groups, be aware of mix shifts. A mix shift is when the amount of data in a slice is different across the groups you are comparing. Simpson’s paradox.
        (my e.g.: Grans vs others; Time frames; )
    
    - Consider practical significance
        - “Even if it is true that value X is 0.1% more than value Y, does it matter?” 
        - “How likely is it that there is still a practically significant change”? 
        (my e.g.: )
        
    - Check for consistency over time  
        (e.g. ICD9/10 pattern over time)     

- Process: Recommendations on how you approach your data, what questions to ask, and what things to check.
    - Separate Validation, Description, and Evaluation
        - Validation or Initial Data Analysis (sanity checking)
        (my e.g.: max/min values, like 99999)
        
        - Description (objective interpretation of this data).
        Users do fewer queries with 7 words in them?”, “The time page load to click (given there was a click) is larger by 1%”
        (my e.g.: )
        
        - Evaluation. 
        “Users find results faster” or “The quality of the clicks is higher.”
        (my e.g.: )

    - Confirm expt/data collection setup
    (my e.g.: Machine-induced homopolymer errors)
        - Communicating precisely between the experimentalist and the analyst is a big challenge. 
        - If you can look at experiment protocols or configurations directly, you should do it. 
        - Otherwise, write down your own understanding of the setup and make sure the people responsible for generating the data agree that it’s correct.
    
    - Check vital signs
        - Did the number of users change? 
        - Did the right number of affected queries show up in all my subgroups? 
        - Did error rates changes? 
        - Just as your doctor always checks your height, weight, and blood pressure when you go in, check your data vital signs to potential catch big problems.
    
    - Standard first, custom second
    Especially when looking at new features and new data, it’s tempting to jump right into the metrics that are novel or special for this new feature.
    (my e.g.: First consider time series methods, then consider my own method also by adding into more domain knowledge!)
    
    - Measure twice, or more
    (my e.g. physicist version: Can we arrive at the same result by two ways? Formula derivation/simulation + empirical evidence.)

    - Check for reproducibility
    
    - Check for consistency with past measurements
    if you are looking at measuring search volume on a special population and you measure a much larger number than the commonly accepted number, then you need to investigate
    (my e.g. AUROC could not be so high!)
    
    - Make hypotheses and look for evidence
        - “What experiments would I run that would validate/invalidate the story I am telling?” (even if can't do)
    
    - Exploratory analysis benefits from end to end iteration
    (my e.g. almost like iterative development of code; things are tree-structure, not linked list; get simplest thing to run through!)
    

- Social: How to work with others and communicate about your data and insights.
    - Data analysis starts with questions, not data or a technique
    (my e.g. doing projects vs. write articles, have to converge... better think of it beforehand! different energy, I would prefer industry..)
    
   - Acknowledge and count your filtering
        - Acknowledge and clearly specify what filtering you are doing
        - Count how much is being filtered at each of your steps
    
   - Ratios should have clear numerator and denominators
   e.g. "click-through rate of a site on search results"
        - “# clicks on site’ / ‘# results for that site’
        - ‘# search result pages with clicks to that site’ / ‘# search result pages with that site shown’
        (my e.g. Yes! What do "sample size" mean for bio/math people! Let's talk in equations!!)
   
   - Educate your consumers:
        - especially important when your data has a high risk of being misinterpreted or selectively cited
        - responsible for providing the context and a full picture of the data and not just the number a consumer asked for
     
     Be both skeptic and champion:
        - be both the champion of the insights you are gaining as well as a skeptic
        (my e.g. why AUROC is so high/low?!)
        
        - Share with peers first, external consumers second
        
        - Expect and accept ignorance and mistakes
            - It feels even worse when you make a mistake and discover it later
        
   

**18. Causality in machine learning**
January 31, 2017
DHARAN, NIALL CARDIN, TODD PHILLIPS, AMIR NAJMI

Consider the following iterative updating procedure:

- Definitions

Y: binary event of uptake Y ()

- On non-randomized data, use the model

logit(EY) = beta_1 X_1 + offset(\hat{beta}_pi X_pi)

and only update the quality score coefficients (here just beta_1)

- On randomized data, use the model

logit(EY) = offset(\hat{beta}_1 X_1) + beta_pi X_pi + beta_e X_e

and only update the prominence coefficient (here beta_pi and beta_e)

**19. Attributing a deep network’s prediction to its input features** 
March 13, 2017
By MUKUND SUNDARARAJAN, ANKUR TALY, QIQI YAN

http://www.unofficialgoogledatascience.com/2017/03/attributing-deep-networks-prediction-to.html

"This post discusses the problem of identifying input feature 
importance for a deep network. We present a very simple method, 
called "integrated gradients", to do this. All it involves is a 
few calls to a gradient operator. It yields insightful results for a 
variety of deep networks."

Comment: Not very relevant to QA role?!



**20. Our quest for robust time series forecasting at scale**
April 17, 2017
http://www.unofficialgoogledatascience.com/2017/04/our-quest-for-robust-time-series.html

CausalImpact is powered by bsts (“Bayesian Structural Time Series”), also from Google, 
which is a time series regression framework using dynamic linear models fit using Markov 
chain Monte Carlo techniques. The regression-based bsts framework can handle predictor 
variables, in contrast to our approach. Facebook in a recent blog post unveiled Prophet, 
which is also a regression-based forecasting tool. But like our approach, Prophet aims 
to be an automatic, robust forecasting tool.

We also permit transformations, such as a Box-Cox transformation.

(1) Model diagram
![Alt text](images/workflow.png?raw=true "Optional Title")

(2) Simple argument of simple averaging will be better than individual/more sophiscated models:

X_1 ~ N(0,1), X_2 ~ N(0,2).

Consider variance of X_A = 0.5 X_1 + 0.5 X_2 or generally X_C = k X_1 + (1-k) X_2

(3) Ensembling of predicting methods

Pretty much any reasonable model we can get our hands on! 
Specific models include variants on many well-known approaches, 
such as the Bass Diffusion Model, the Theta Model, Logistic models, 
bsts, STL, Holt-Winters and other Exponential Smoothing models, 
Seasonal and other other ARIMA-based models, Year-over-Year growth models, 
custom models, and more.


**21. Fitting Bayesian structural time series with the bsts R package**
July 11, 2017

Comment: Why python does not have such a package?! (statsmethod?)


**22. Unintentional data** 
October 12, 2017
by ERIC HOLLINGSWORTH

Comment: intuitive, qualitative



**23. Designing A/B tests in a collaboration network** 
January 16, 2018
by SANGHO YOON
http://www.unofficialgoogledatascience.com/2018/01/designing-ab-tests-in-collaboration.html

Question: How to prevent potential contamination (or inconsistent treatment exposure) of samples due to network effects?

![Alt text](images/Sangho3.png?raw=true "Optional Title")
Did:
- describe the unique challenges in designing experiments on developers working on GCP. 

- use simulation to show how proper selection of the randomization unit can avoid estimation bias.

Difference of GCP compared to other social networks:
- A few large connected networks versus many connected components: 
    - Methodologies for experimenting on users in social networks focus on ways to partition the overall graph into subgraphs
    -  In GCP, there are many small connected components because our customers want to manage their own privacy and security in their projects, and do not want to share access with third parties.

- Spillover effects versus contamination
    - Experiments in social networks must care about "spillover" or influence effects from peers. 
    - In our case, avoiding confusion is more important than estimating indirect treatment effects. For example, imagine the confusion resulting from two users who work on a shared project but see two different versions.

- Method
    - Build user graphs
    - Stratify graphs by size and usage: 
        Measure the size of each component by number of users and revenue 
        and stratify graphs in number of users and revenue.
    - Select samples and random assignment
    - Run experiment

- Results
    - TODO: Modeling network effects
    - TODO: Experimental power and unit of randomization
    - TODO: Estimation bias due to unit of randomization
    - TODO: Dynamic evolution of user collaboration network
    


**24. Compliance bias in mobile experiments** 
March 22, 2018
by DANIEL PERCIVAL
http://www.unofficialgoogledatascience.com/2018/03/quicker-decisions-in-imperfect-mobile.html

Question: many users assigned the treatment do not actually experience the treatment for a long time period after the beginning of the experiment.

Answer: propensity based models often provide insightful refinements to the basic ITT or TOT approaches, and would form the basis for methods that would address these complexities.

TODO Rocky: prognostic score thing!

Issue 1: the empirical mismatch between treatment assignment and experience

Issue 2: the users experiencing treatment are not a simple random subsample of the population

Issue 3: the need to make a timely decision

Intent to Treat (ITT) and Treatment on the Treated (TOT) analysis

TODO: propensity matching is more consistent over time than propensity weighting?!

**25. Crawling the internet: data science within a large engineering system**
July 17, 2018
by BILL RICHOUX


- First alg:
    - At a period of time tau since the last time Page j in Host i (value w_ij) is crawled,
        the prob that it will be fresh (not have meaningful change) will be e^(-delta_ij tau)
    - If Page j is recrawled every Delta_ij time units, its prob of being fresh
        at a time chosen uniformly over this period will be 
        1/Delta_ij \int_0^Delta_ij e^(-delta_ij tau)
    - Our objective of choosing recrawl periods to maximize our freshness metric
        maps to the following optimization problem:
        argmax_{Delta_11, Delta_12, ...} \sum_ij w_ij/(delta_ij Delta_ij) (1 - e^(-delta_ij Delta_ij))
        s.t. \sum_j 1/Delta_ij <= k_i (max crawling rate for the i-th host)
             \sum_ij 1/Delta_ij <= k_0 (max global crawling rate)
             Delta_ij > 0 

- The missing link
    - link between our singular intention (improve our freshness metric) 
    and the multitude of decisions to be made in the software layer 
    (how often to crawl each web page)
    - could not reasonably reside in memory on a normal computer
    - employ a solver distributed across multiple machines (no host would be split across multiple machines)

- Pushback
    - black box solution: 
        - None of the infrastructure engineers on the 
        crawl team were convinced that they could easily understand and 
        diagnose when or why such new optimization infrastructure would fail.
        - problems would invariably occur, as was always the case when working
        at the scale of Google search with a continuously evolving external 
        environment (the web), no matter how well designed the system.
    
    - solving the wrong problem formulation: 
        - Our solution approach was to freeze time, solve a static 
        optimization problem, and then to solve again repeatedly.
        
        - However, in the software layer, the decision being made 
        repeatedly was to determine the subset of web pages that should 
        be crawled next, whenever some crawl bandwidth becomes available.
        
        - We would need to take the solution from our solver, and merge 
        it in with information detailing when each URL was last crawled,
        and then determine what URLs should be crawled next.
        It’s arguably a minor extra step, but requires a bridge. 
        
        - By not matching the problem formulation in the infrastructure, 
        we imposed an extra layer of complexity to interpret our solution.
    
    - substantial expansion in complexity:
        - replace an existing heuristic implemented in a few lines of code,
        - with some new software infrastructure, 
        - a distributed solver for a very large convex optimization problem, 
        — likely tens of thousands of lines of new code. 
        - infra engineers insisted that any new, not-battle-tested system 
            of moderate complexity would likely fail in unintended ways. 
        - Without any indication that such proposed infrastructure would 
        become a standard tool to solve additional problems, 
        the crawl team was hesitant to commit to it.
    
    - limitations on responsiveness: 
        - Although we referred to the solution as optimal, in reality it would 
        never be optimal considering that the parameters feeding into such recrawl 
        logic (the maximum crawl rates, the value of a page, the estimated change 
        rate of a page) were all being updated asynchronously. 

- Deconstructing our mistakes (1. context aware, 2. eagerness to innovate)
    - Failing to appreciate the infrastructure engineer’s pressures and responsibilities
    - Knowing too little of the actual, existing software implementation
    - Seduction of familiarity
    
- A revised proposal: still statis
    - Define crawl rate rho_ij = 1/Delta_ij (now as a var instead of a function of Delta_ij!)
    - Contribution of Page i,j to the overall obj: C_ij(rho_ij)=w_ij/delta_ij rho_ij [1 - e^(-delta_ij/rho_ij)]
    - Overall freshness obj: argmax_rho \sum_ij C_ij rho_ij
    - Constraint: \sum_ij rho_ij <= k_i, \sum_ij rho_ij <= k_0
    - Lagrange multiplier: 
    J(rho) = (\sum_ij C_ij(rho_ij)) + (\sum_i lambda \sum_j rho_ij) + (lambda_0 \sum rho_ij)
    - Setting \partial J/\partial rho_ij=0, we get C'_ij(rho*_ij) = lambda_i + lambda_0,
    where C'(rho) = w/delta(1-e^(-delta/rho)) - w/rho e^(-delta/rho), all indexed by i,j
    - C'(rho) monotone decreases
    
- New perspective on dynamical: 
a function that tell us the value of crawling any web page at any given time
    - Consider V(Delta) = C'(1/Delta) = w/delta(1-e^(-delta Delta)) - w/rho e^(-delta Delta)
     monotonically increasing, starting with V(0)=0, 
     at the optimal crawl period, it follows: V_ij(Delta*_ij) = lambda_i + lambda_0
    - Let tau as the time that has elapsed since Page j on Host i was last crawled, consider
    V_ij(tau) as Page i,j’s crawl value function. 
    
    - Non-greedy: a greedy algorithm would devote more recrawl resources towards high 
    value pages, as lower value pages would commonly starve.
    
    - we can evaluate the crawl value function for each web page on a host. 
    We can use this function to sort the web pages, and then determine which web pages 
    should be scheduled for immediate crawl. 

- Summary
    
    - "Type III error — giving the right answer to the wrong problem"

    - Comment: Usually not a global optimization to solve, but more of a k -> k+1 problem.


