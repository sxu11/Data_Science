
- AIC/BIC

    - Model M has k parameters, n observations X, MLE \hat theta
    - AIC: Akaike's Information Criterion
        - 2k - 2log(\hat L), where \hat L = likelihood of X given \hat theta
    - BIC: Bayesian Information Criterion
        - klog(n) - 2log(\hat L)
        - also punish having too many data!
        - Me: more data, the harder to explain, so smaller L, bigger -log(\hat L)?!
    - Neither tells how good explain the data, but how balance..
    
    - AIC and BIC are not try to answer the same question. 
        - AIC tries to select the model that most adequately describes an unknown, 
        high dimensional reality. This means that reality is never in the set of candidate 
        models that are being considered.
        - BIC tries to find the TRUE model among the set of candidates. 
        I find it quite odd the assumption that reality is instantiated in one of the 
        model that the researchers built along the way. This is a real issue for BIC.
        - AIC fails to converge in probability to the true model, whereas BIC does. 
        - AIC is best for prediction as it is asymptotically equivalent to cross-validation.
            BIC is best for explanation as it is allows consistent estimation of the 
            underlying data generating process.
        - AIC=LOO (leave-one-out) and BIC=K-fold
        - AIC: predict. BIC: explain. 
        https://robjhyndman.com/hyndsight/to-explain-or-predict/
        
        - Minimum Description length (MDL):
            - MDL is derived directly from the BIC when 𝑁→∞ assuming i.i.d samples.

- Lasso, interesting argument:
    - "If you are only interested in prediction, then model selection doesn't help and 
    usually hurts (as opposed to a quadratic penalty = L2 norm = ridge regression with 
    no variable selection). LASSO pays a price in predictive discrimination for trying 
    to do variable selection."
    - "There is NO reason to do stepwise selection. It's just wrong."
    - A list of potential problems with stepwise selection:
        https://www.stata.com/support/faqs/statistics/stepwise-regression-problems/

LASSO/LAR are the best automatic methods. But they are automatic methods. They let the analyst not think."

sklearn collection of model selection techniques:
https://scikit-learn.org/stable/modules/feature_selection.html

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
            
            
            