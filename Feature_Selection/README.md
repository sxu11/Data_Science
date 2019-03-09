

- RFECV:
    - Sample code:
        estimator = RandomForestClassifier(random_state)
        self._selector = RFECV(estimator)
        self._selector.fit(self._input_X, self._input_y)
        
        n_selected = self._selector.n_features_
        support = self._selector.support_
        ranking = self._selector.ranking_
     
     - Why so fast?
        - Parallel run:
            parallel, func, = Parallel(n_jobs=self.n_jobs), delayed(_rfe_single_fit)
        
        - Data size is relatively small (10000*800?!)
        
        - Shrinking column number after each step
        
        - Use just default parameter, no grid search
        
        - Paper:
            Gene Selection for Cancer Classification using Support Vector Machines on 2002
            - Train the classifier (optimize the weights wi with respect to J ).
            - Compute the ranking criterion for all features (D J (i) or (w_i)^2).
            - Remove the feature with smallest ranking criterion.