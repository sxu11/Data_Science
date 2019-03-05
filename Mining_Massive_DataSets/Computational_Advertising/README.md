

Infinite data, online algorithms

Query-to-advertiser graph
![Alt text](images/Query-to-advertiser.png?raw=true "Optional Title")

Advertiser X wants to show an ad for topic/query Y.
This is an online problem: We have to make decisions as queries/topics show up. 
We do not know what topics will show up in the future.

Online Bipartite matching
e.g. 
Nodes: Boys and Girls; 
Links: Preferences
Goal: Match boys to girls so that the most preferences are satisfied

Problem: Find a maximum matching for a given bipartite graph
   - A perfect one if it exists
- There is a polynomial-time offline algorithm based on augmenting paths (Hopcroft & Karp 1973,
see http://en.wikipedia.org/wiki/Hopcroft-Karp_algorithm)

- But what if we do not know the entire graph upfront?
Initially, we are given the set boys
- In each round, one girl’s choices are revealed (That is, the girl’s edges are revealed)
- At that time, we have to decide to either:
    - Pair the girl with a boy
    - Do not pair the girl with any boy
- Application: e.g. assigning tasks to servers

- Greedy algorithm for the online graph matching problem:
    - Pair the new girl with any eligible boy; If there is none, do not pair the girl
    - How good is the algorithm?
    
- Metric: Competitive Ratio = min_{all possible inputs} (M_greedy / M_opt)
(worst performance of greedy over all possible inputs I)

Setting up theory:
![Alt text](images/greedy.png?raw=true "Optional Title")

(3) Optimal matches all girls in G to (some) boys in B

Combining (2) and (3)
|G| <= |B| <= |M_greedy|

Combining (1) and (4)
Worst case is when |G|=|B|=|M_greedy|
|M_greedy| + |M_greedy| >= |M_opt|
|M_greedy| >=  1/2|M_opt|

"1 greedy edge at most will screw up 1 optimal edge!" -- Some smart student


History of Web Advertising
![Alt text](images/history.png?raw=true "Optional Title")

Performance-based Advertising
- Introduced by Overture around 2000
    - Advertisers bid on search keywords
    - When someone searches for that keyword, the highest bidder’s ad is shown
    - Advertiser is charged only if the ad is clicked on
- Similar model adopted by Google with some changes around 2002
    - Called Adwords
    
Web 2.0
- Performance-based advertising works!
    - Multi-billion-dollar industry
- Interesting problem:
    What ads to show for a given query? 
    (Today’s lecture)
- If I am an advertiser, which search terms should I bid on and how much should I bid? 
    (Not focus of today’s lecture)
    
Adwords Problem
- A stream of queries arrives at the search engine: q1, q2, ...
    - Several advertisers bid on each query
    - When query qi arrives, search engine must pick a subset of advertisers whose ads are shown
- Goal: Maximize search engine’s revenues
    - Simple solution: Instead of raw bids, use the “expected revenue per click” (i.e., Bid*CTR)
- Clearly we need an online algorithm!

Challenges:
- CTR of an ad is unknown
- Advertisers have limited budgets and bid on multiple queries
    - Each ad-query pair has a different likelihood of being clicked
    - Search engine guarantees that the advertiser will not be charged more than their daily budget

Some complications we will not cover:
    - 1) CTR is position dependent 

Some complications we will cover (next lecture):
    - 2) Exploration vs. exploitation
    

The BALANCE algorithm
- Given:
    - 1. A set of bids by advertisers for search queries
    - 2. A click-through rate for each advertiser-query pair
    - 3. A budget for each advertiser (say for 1 month)
    - 4. A limit on the number of ads to be displayed with each search query
- Respond to each search query with a set of advertisers such that:
    - 1. The size of the set is no larger than the limit on the number of ads per query
    - 2. Each advertiser has bid on the search query
    - 3. Each advertiser has enough budget left to pay for the ad if it is clicked upon
    
Greedy:
- Our setting: Simplified environment
    - There is 1 ad shown for each query
    - All advertisers have the same budget B
    - All ads are equally likely to be clicked
    - Bid/value of each ad is the same (=1)
- Simplest algorithm is greedy:
    - For a query pick any advertiser who has bid 1 for that query
    - Competitive ratio of greedy is 1/2
    
BALANCE Algorithm by Mehta, Saberi, Vazirani, and Vazirani
- For each query, pick the advertiser with the
largest unspent budget
    - Break ties arbitrarily (but in a deterministic way)
- In general: For BALANCE on 2 advertisers, competitive ratio = 3/4
- In the general case, worst competitive ratio of BALANCE is 1–1/e = approx. 0.63