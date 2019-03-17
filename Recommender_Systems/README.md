
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
    - So for the R(ating) matrix of shape (m,n), m movies and n users,
    each (i,j) comes from product of two vectors: M_i and N_j, where
    M of size (m,p) is the Movie matrix and N of size (p,n) is the User matrix. 
    p is the number of genre (e.g. when p=2, there are romance and action)
    
    - 16.2 Content-based CF (assume we know features P of the movie): 
        - To learn N_j, define the optimization problem:
        SE_j = min_{N_j} \sum_{all movie i s.t. j gives a rating} (M_i*N_j - R_{i,j})^2 / (num of movies that are rated by j)
        - Could be solved by a linear regression! Could also add a regularization term 
        \lambda/(num of movies that are rated by j) * L2 norm of N_j! 
        Punishment (regularization) does not include N_j(0), which is the intercept. 
        - To learn all N_j, just do another summation \sum_j SE_j. 
        - Use gradient descent to get updates N_j = N_j - XXX.
        - https://www.youtube.com/watch?v=9siFuMMHNIA&t=322s

---
*OrderRex: clinical order decision support and outcome predictions by data-mining electronic medical records
https://academic.oup.com/jamia/article/23/2/339/2572407