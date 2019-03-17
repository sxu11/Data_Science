

- 8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset

    - 1) Can You Collect More Data?

    - 2) Try Changing Your Performance Metric
    
    - 3) Try Resampling Your Dataset
        - Consider testing under-sampling when you have an a lot data (tens- or hundreds of thousands of instances or more)
        - Consider testing over-sampling when you don’t have a lot of data (tens of thousands of records or less)
        - Consider testing random and non-random (e.g. stratified) sampling schemes.
        - Consider testing different resampled ratios (e.g. you don’t have to target a 1:1 ratio in a binary classification problem, try other ratios)
        
    - 4) Try Generate Synthetic Samples
    
    - 5) Try Different Algorithms
    
    - 6) Try Penalized Models
    
    - 7) Try a Different Perspective
        - Anomaly detection is the detection of rare events.
        This shift in thinking considers the minor class as the outliers 
        class which might help you think of new ways to separate and classify samples.
        - Change detection is similar to anomaly detection except rather 
        than looking for an anomaly it is looking for a change or difference. 
        This might be a change in behavior of a user as observed by usage 
        patterns or bank transactions.
    
    - 8) Try Getting Creative
        - https://www.quora.com/In-classification-how-do-you-handle-an-unbalanced-training-set
        - Decompose your larger class into smaller number of other classes…
        - resampling the unbalanced training set into not one balanced set, 
        but several. Running an ensemble of classifiers on these sets could 
        produce a much better result than one classifier alone. 
        
        
        
- SVM Hyperparameters
    - Kernel selects the type of hyperplane used to separate the data. 
    - The higher the Gamma value (non linear hyperplanes) it tries to exactly fit the training data set 
    - Degree used when kernel is set to ‘poly’. 
    - C is the penalty parameter of the error term. It controls the trade off between 
    smooth decision boundary and classifying the training points correctly.
    - https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
        
    