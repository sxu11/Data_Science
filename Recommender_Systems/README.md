
I'm interested in Recommender Systems in the clinical context. 

Previous efforts of OrderRex* using co-occurrence of clinical items within 24 hours (like a 'shopping cart'), adjusted by inverse probability weighting. 


- gensim: Best Topic modeling?
    - LDA
    - hierarchical LDA
    - dynamic LDA
    - DeepLearning
    - Word2Vec
    - Doc2Vec
    - POS
    - https://www.youtube.com/watch?v=3mHy4OSyRf0
    
    
    
- Andrew Ng's class:
    - All we want is R!
    
    - So for the R(ating) matrix of shape (m,n), m movies and n users,
    each (i,j) comes from product of two vectors: M_i and N_j, where
    M of size (m,p) is the Movie matrix and N of size (p,n) is the User matrix. 
    p is the number of genre (e.g. when p=2, there are romance and action)
    
    - 16.2 Content-based Recommender (assume we know both P's dimension and 
                                        genre distribution M_i of each movie): 
        - Suppose know M_i, but do not know N_j. 
        To learn missing vals of R, define the optimization problem:
        SE_j = min_{N_j} \sum_{all movie i s.t. j gives a rating} (M_i*N_j - R_{i,j})^2 / (num of movies that are rated by j)
        - Could be solved by a linear regression! Could also add a regularization term 
        \lambda/(num of movies that are rated by j) * L2 norm of N_j! 
        Punishment (regularization) does not include N_j(0), which is the intercept. 
        - To learn all N_j, just do another summation \sum_j SE_j. 
        - Use gradient descent to get updates N_j = N_j - XXX.
        - https://www.youtube.com/watch?v=9siFuMMHNIA&t=322s
    
    - 16.3 User-based Recommender 
    (know P's dimension, but no idea for genre distribution M_i of each movie)
        - Suppose know N_j (user's preference), need to infer genre of each movie M_i
        - Optimization problem:
            Given N_j, learn M_i s.t. SE_i = min_{M_i} \sum_{all user j s.t. gives i a rating} (M_i*N_j - R_{i,j})^2 / 2
        - Again, sum over..
        - https://www.youtube.com/watch?v=9AP-DgFBNP4&t=1s
        
    - Combine both based, get a Collaborative Filtering!
        - Guess N -> M -> N -> M -> ...
        (Underlying: Each user rates multiple movies and each movie is rated by multiple users!)
        - Smarter: minimize both at the same time!
        - J = min_{M,N} \sum_ij (M_i*N_j-R_ij)^2 + \sum_i |M_i|_2 + \sum_j |N_j|_2
        - Got rid of all intercept terms in M_i (and also in N_j)!
        
    - Another way of looking at R, just = M*N. R_ij = M_i*N_j, w/o explicitly showing P. 
        - Low rank matrix factorization
        - How to find movies j related to movie i? (but in clinics, why need find similar items?!)
        - Implementation detail: Mean Normalization 
            - When a user has not yet rated any movie, his only contribution is in the regularization term,
            resulting in N_j=[0,0]
            - Solution: subtract each user j's rating for movie i by its average rating!

---

*OrderRex: clinical order decision support and outcome predictions by data-mining electronic medical records
https://academic.oup.com/jamia/article/23/2/339/2572407