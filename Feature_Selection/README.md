
sklearn collection of model selection techniques:

- Removing features with low variance: using VarianceThreshold
    - e.g. Features w/ constant values (0 variance)
    
- Univariate feature selection
    Select the best features based on univariate stat test. 
    - SelectKBest
        - e.g. X_new = SelectKBest(sklearn.feature_selection.chi2, k=2).fit_transform(X, y)
        - Scoring function: 
            - For regression: f_regression, mutual_info_regression
            - For classification: chi2, f_classif, mutual_info_classif
            - The methods based on F-test estimate the degree of linear dependency between two random variables. 
            - Mutual information methods can capture any kind of statistical dependency, 
                but being nonparametric, they require more samples for accurate estimation
    - SelectPercentile
    - SelectFpr (false positive), SelectFdr (false discovery), SelectFwe (family wise error)
    - GenericUnivariateSelect 

- Recursive feature elimination, RFE (CV):
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
            
- Feature selection using SelectFromModel:
    Customable
    - L1-based feature selection:
        -   lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
            model = SelectFromModel(lsvc, prefit=True)
            X_new = model.transform(X)
        -   L1-recovery and compressive sensing
        
    - Tree-based feature selection
        - clf = ExtraTreesClassifier(n_estimators=50)
            clf = clf.fit(X, y)
            model = SelectFromModel(clf, prefit=True)
            X_new = model.transform(X)
    
    - Feature selection as part of a pipeline
        - clf = Pipeline([
                  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
                  ('classification', RandomForestClassifier())
                ])
            clf.fit(X, y)