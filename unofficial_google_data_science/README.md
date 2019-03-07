

There are a total of 26 articles there when I accessed the website on 03/05/2019.

8. How to get a job at Google — as a data scientist

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
 


Our quest for robust time series forecasting at scale 
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


Fitting Bayesian structural time series with the bsts R package 
July 11, 2017

Comment: Why python does not have such a package?! (statsmethod?)


Unintentional data 
October 12, 2017
by ERIC HOLLINGSWORTH

Comment: intuitive, qualitative

24. Designing A/B tests in a collaboration network 
January 16, 2018
by SANGHO YOON
http://www.unofficialgoogledatascience.com/2018/01/designing-ab-tests-in-collaboration.html

Question: How to prevent potential contamination (or inconsistent treatment exposure) of samples due to network effects?

Did:
(1) describe the unique challenges in designing experiments on developers working on GCP. 
(2) use simulation to show how proper selection of the randomization unit can avoid estimation bias.



25. Compliance bias in mobile experiments 
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

26. Crawling the internet: data science within a large engineering system
July 17, 2018
by BILL RICHOUX

"Type III error — giving the right answer to the wrong problem"

Comment: Usually not a global optimization to solve, but more of a k -> k+1 problem.

TODO: look back and more summary!