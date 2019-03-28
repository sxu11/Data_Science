
Including traditional and Bayesian A/B testing for Analytics. 

online calculator: http://www.evanmiller.org/ab-testing/sample-size.html


Facebook Ads 
https://karolakarlson.com/facebook-ad-ab-testing-rules/

Rule #1: Test high-impact ad campaign elements
- campaign elements with highest split testing ROI: 
    - Countries
    - Precise interests
    - Mobile OS
    - Age ranges
    - Genders
    - Ad images
    - Titles
    - Relationship status
    - Landing page
    - Interested in

Rule #2: Prioritise your A/B test ideas

Rule #3: Test one element at a time

Rule #4: Test a reasonable number of variables

Rule #5: Test 3-5 highly differentiated variations

Rule #6: Use the right Facebook campaign structure

Rule #7: Make sure your test results are valid

Rule #8: Calculate statistical significance

Rule #9: Know what budget you’ll need

Rule #10: Track the right metrics
    - As you look at you Facebook A/B test results, there will be lots of metrics to consider: 
    ad impressions, cost-per-click, click-through-rate, cost-per-conversion, conversion rate…
    - Always track the cost-per-conversion as your primary goal.
    
    
- Firebase practical A/B testing
    - https://www.youtube.com/watch?v=UMz9dSPGzoo

- Workflow for A/B testing:
    - Decide metrics 
        - daily active users (DAU) to measure user engagement
        - click through rate (CTR) to measure a button design on a webpage
        - Four types:
            - Sums and counts
            - Distribution (mean, median, percentiles)
            - Probability and rates (click through probability and click through rate)
            - Ratios: any two numbers divide by each other
        - high sensitivity, robust against others
    - Funnel analysis
        - Cannot use A/B testing: Long term and new experiment. 
    - Decide variants
    - Unit of evaluation measurement, invariant metric
    - Decide sample size and length
        - type-i and ii error, practical significance level
            - Usually the significance level is 0.05 and power is 0.8. 
            practical significance varies depends, is higher than statistical significance level. 
        - sample size
            - The conversion rate value of control variation (baseline value)
            - The minimum difference between control and experiment which is to be identified. 
            The smaller the difference between experiment and control to be identified, 
            the bigger the sample size is required.
            - Chosen confidence/significance level
            - Chosen statistical power
            - Type of the test: one or two tailed test. Sample size for two tailed test is relatively bigger.
            - Evan Miller calculator
        - Cool down time. external events, weekends, holidays, seasonality
    - Run the test
        - Long-term challenge:
            - Separate "Long-term" with "primary impact of applying the A/B 
                treatment" (e.g. due to seasonality, other serving system)
        - Optional for long-term: A/A testing
            - No "primary effects" any more
            - But, have to wait too long
        - Optional for long-term: B/B testing
            - Re-randomize study participants (from control group) daily, 
            so still belong to "control" in the long term
            - In the short term, serve as B/B testing. 
    - Analyze
        - sample mean diff, pooled sd, margin of error (z*sd)
        - diagnostics
            - What do I do if I do not trust the results?
            run the same test again
            - What if I do not have control?
            pick the one that’s the most similar to how you currently design pages
            - When A/B test is not useful, what you can do?
            Analyze the user activity logs
            Conduct retrospective analysis
            Conduct user experience research
            Focus groups and surveys
            Human evaluation
        - Best Refuted Causal Claims from Observations Studies
            - Example: Users who see error messages in Office 365 churn less.
            This does NOT mean we should show more error messages.
            They are just heavier users of Office 365
            See Best Refuted Causal Claims from Observations Studies
            for great examples of this common analysis flaw
    - Additional materials
        - https://towardsdatascience.com/a-summary-of-udacity-a-b-testing-course-9ecc32dedbb1
        - http://napitupulu-jon.appspot.com/posts/subject-experiment-abtesting-udacity.html
        

