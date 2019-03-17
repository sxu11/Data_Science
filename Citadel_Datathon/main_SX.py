

import numpy as np
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.width', 300)

def data_duplicate(some_df, num_repeats):
    num_doubles = np.log2(num_repeats)
    num_doubles = int(np.ceil(num_doubles))

    # copy_df = some_df.copy()
    for i in range(num_doubles):
        print "Processing the %d'th doubling..."%i
        some_df = some_df.append(some_df.copy(), ignore_index=True)
    return some_df

def parse_one_occupation_line(oneline_occupation):
    census_code = oneline_occupation[:3]
    prestige_score = oneline_occupation[4:9]
    occupational_category = oneline_occupation[10:-1]
    return (census_code, prestige_score, occupational_category)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=True)
    print 'after learning curve!'
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

if __name__ == '__main__':
    data_folder_sample = 'sample_data/'

    num_repeats = 882000/24
    print num_repeats

    chemicals_sample_df = pd.read_csv(data_folder_sample + 'chemicals.csv', index_col=False)
    chemicals_enlarged_sample_df = data_duplicate(chemicals_sample_df, num_repeats)
    chemicals_enlarged_sample_df.to_csv("enlarged_sample_data/chemicals.csv")

    num_repeats = 1.35 * 10**6 / 10

    droughts_sample_df = pd.read_csv(data_folder_sample + 'droughts.csv', index_col=False)
    droughts_enlarged_sample_df = data_duplicate(droughts_sample_df, num_repeats)
    droughts_enlarged_sample_df.to_csv("enlarged_sample_data/droughts.csv")

rawdata_folder = 'raw_data/'  # ['Nitrates', 'Trihalomethane', 'Halo-Acetic Acid', 'Arsenic', 'DEHP', 'Uranium']
interdata_path = 'intermediate_data/'

if not os.path.exists(interdata_path+'cleaned_chemicals.csv'):

    # chemicals contaminant_level!
    #Counter({'Nitrates': 329372, 'Trihalomethane': 154258, 'Halo-Acetic Acid': 146132, 'Arsenic': 142001, 'DEHP': 72825, 'Uranium': 37731})
    df = pd.read_csv(rawdata_folder + '/' + 'chemicals.csv')
    df = df[['fips', 'year', 'contaminant_level', 'chemical_species', 'pop_served', 'value', 'unit_measurement']]

    # df = pd.concat([chemicals_pd, pd.get_dummies(chemicals_pd['contaminant_level'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['chemical_species'])], axis=1)
    df = df.drop(columns=['contaminant_level', 'chemical_species'])

    df['pop_served'] = df['pop_served'].map(lambda x: 0.1 if x<=0 else x)


    # Define a lambda function to compute the weighted mean:
    wm = lambda x: np.average(x, weights=df.loc[x.index, "pop_served"])
    # Define a dictionary with the functions to apply for a given column:
    f = {'pop_served': ['sum'], 'value': {'weighted_mean' : wm} }
    # Groupby and aggregate with your dictionary:
    use_columns = [x for x in df.columns if (x!='pop_served' and x!='value')]
    print 'use_columns:', use_columns
    df = df.groupby(use_columns,axis=0, as_index=False).agg(f)
    df.columns = df.columns.droplevel(1)

    df.to_csv(interdata_path+'cleaned_chemicals.csv')
else:
    df = pd.read_csv(interdata_path+'cleaned_chemicals.csv')

if not os.path.exists(interdata_path+'cleaned_chemicals_joined_agriculture.csv'):


    occupations_pd = pd.read_csv(rawdata_folder + '/' + 'industry_occupation.csv')
    occupations_pd = occupations_pd[['fips','year','agriculture']]
    df = df.merge(occupations_pd, how='left', on=['fips','year'])
    df.to_csv(interdata_path + '/' + 'cleaned_chemicals_joined_agriculture.csv')
else:
    df = pd.read_csv(interdata_path + '/' + 'cleaned_chemicals_joined_agriculture.csv')

import sklearn
# print df.shape[0]
# print 'good values:', df.shape[0] - sum(df['agriculture'].isna().values)

print 'df.shape:', df.shape
print 'with whole data:', df.dropna().shape[0]
# all missing values are at y!
df = df.dropna()
df = df.drop(columns={'unit_measurement'}) #TODO!

from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
X = df[df.columns.difference(['agriculture'])]
y = df['agriculture']

X_train,X_test,y_train,y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

# clf = LinearRegression()
# clf.fit(X_train, y_train)
# res_pred = clf.predict(X_test)
#
# print res_pred
# print y_test.values
#
# all_rela_errors = (res_pred - y_test.values)/(y_test.values+0.1)
# print all_rela_errors
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# plt.scatter(res_pred, y_test.values)
# plt.xlim([0,7500])
# plt.ylim([0,7500])
# plt.show()
# quit()


import Utils
from sklearn.svm import SVC


estimator = LinearRegression()
title = "Learning Curves (Linear Regression)"
# print 'Starting ' + title

# plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=1)
# plt.show()

# estimator = GaussianNB()
# title = "Learning Curves (Naive Bayes)"
#
# plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=1)
# plt.show()

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=1)

plt.show()
