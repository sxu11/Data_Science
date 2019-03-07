

There are a total of 26 articles there when I accessed the website on 03/05/2019.

I feel that some of these articles tend to be more wordy, though I would prefer some structured 
way of saying things (STAR):

- Situation:

- Task

- Action:

- Results:

But I do understand more introductory words could be more friendly to non-technical readers. 


**8. How to get a job at Google — as a data scientist**

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


**Practical advice for analysis of large, complex data sets**
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
        
   

**Causality in machine learning**
Consider the following iterative updating procedure:

- Definitions

Y: binary event of uptake Y ()

- On non-randomized data, use the model

logit(EY) = beta_1 X_1 + offset(\hat{beta}_pi X_pi)

and only update the quality score coefficients (here just beta_1)

- On randomized data, use the model

logit(EY) = offset(\hat{beta}_1 X_1) + beta_pi X_pi + beta_e X_e

and only update the prominence coefficient (here beta_pi and beta_e)

**Attributing a deep network’s prediction to its input features** 
March 13, 2017
By MUKUND SUNDARARAJAN, ANKUR TALY, QIQI YAN

http://www.unofficialgoogledatascience.com/2017/03/attributing-deep-networks-prediction-to.html

"This post discusses the problem of identifying input feature 
importance for a deep network. We present a very simple method, 
called "integrated gradients", to do this. All it involves is a 
few calls to a gradient operator. It yields insightful results for a 
variety of deep networks."

Comment: Not very relevant to QA role?!



**Our quest for robust time series forecasting at scale**
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


**Fitting Bayesian structural time series with the bsts R package**
July 11, 2017

Comment: Why python does not have such a package?! (statsmethod?)


Unintentional data 
October 12, 2017
by ERIC HOLLINGSWORTH

Comment: intuitive, qualitative

**24. Designing A/B tests in a collaboration network** 
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
    


**25. Compliance bias in mobile experiments** 
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

**26. Crawling the internet: data science within a large engineering system**
July 17, 2018
by BILL RICHOUX

"Type III error — giving the right answer to the wrong problem"

Comment: Usually not a global optimization to solve, but more of a k -> k+1 problem.

TODO: look back and more summary!


