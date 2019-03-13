

'''
https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
'''

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/creditcard.csv')
# print df.head()

'''
Summary
missing vals
columns
'''
print df.describe()
print df.isna().sum().max()
print df.columns

'''
Data imbalances
'''
print 'No Frauds, Frauds:', df['Class'].value_counts().values
# sns.countplot('Class', data=df)
# plt.show()

'''
How skewed is each feature
distplot is better-looking than hist...
'''
fig, ax = plt.subplots(1, 2, figsize=(18,4))
amount_val = df['Amount'].values
time_val = df['Time'].values

# sns.distplot(amount_val, ax=ax[0], color='r')
# ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
# ax[0].set_xlim([min(amount_val), max(amount_val)])
#
# sns.distplot(time_val, ax=ax[1], color='b')
# ax[1].set_title('Distribution of Transaction Time', fontsize=14)
# ax[1].set_xlim([min(time_val), max(time_val)])
#
# plt.show()

'''
Scaling the remaining Time and Amount

TODO: this is likely wrong, should split first!
'''
from sklearn import preprocessing
std_scalar = preprocessing.StandardScaler()
rob_scalar = preprocessing.RobustScaler()
df['scaled_amount'] = rob_scalar.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scalar.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

print df.head()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]


original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

'''
Under sampling
'''
df = df.sample(frac=1) # shuffle the data
fraud_df = df.loc[df['Class']==1]
non_fraud_df = df.loc[df['Class']==0][:492]

new_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

'''
Correlation matrices
'''
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for ref)")

sub_sample_err = new_df.corr()
sns.heatmap(sub_sample_err, cmap='coolwarm_r', annot_kws={'size':20},
        ax=ax2)
ax2.set_title("SubSample Correlation Matrix \n (use for ref)")
plt.show()

'''
Simple boxplot distribution
'''
f, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x="Class", y="V17", data=new_df, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=new_df, ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=new_df, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V10", data=new_df, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

plt.show()


f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Class", y="V11", data=new_df, ax=axes[0])
axes[0].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=new_df, ax=axes[1])
axes[1].set_title('V4 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V2", data=new_df, ax=axes[2])
axes[2].set_title('V2 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V19", data=new_df, ax=axes[3])
axes[3].set_title('V19 vs Class Positive Correlation')

plt.show()


'''
Detailed distribution
'''
from scipy.stats import norm
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class']==1].values
sns.distplot(v14_fraud_dist, ax=ax1, fit=norm)
ax1.set_title('V14 Distribution \n (Fraud Transactions)')

v12_fraud_dist = new_df['V12'].loc[new_df['Class']==1].values
sns.distplot(v12_fraud_dist, ax=ax2, fit=norm)
ax2.set_title('V12 Distribution \n (Fraud Transactions)')

v10_fraud_dist = new_df['V10'].loc[new_df['Class']==1].values
sns.distplot(v10_fraud_dist, ax=ax3, fit=norm)
ax3.set_title('V10 Distribution \n (Fraud Transactions)')

plt.show()

'''
Removing outliers x, defined by (if x < v14_lower or x > v14_upper)
'''
def remove_outliers(vNum, new_df):
    vNum_fraud = new_df[vNum].loc[new_df['Class']==1].values
    q25, q75 = np.percentile(vNum_fraud, 25), np.percentile(vNum_fraud, 75)
    print '%s q25, q75:'%vNum, q25, q75

    vNum_iqr = q75 - q25
    print '%s iqrt'%vNum, vNum_iqr

    vNum_cut_off = vNum_iqr * 1.5
    vNum_lower, vNum_upper = q25 - vNum_cut_off, q75 + vNum_cut_off
    print '%s cut off:'%vNum, vNum_cut_off
    print '%s lower and upper:'%vNum, vNum_lower, vNum_upper

    outliers = [x for x in vNum_fraud if x < vNum_lower or x > vNum_upper]
    print '%s outliers:'%vNum, outliers

    return new_df.drop(new_df[(new_df[vNum] > vNum_upper) | (new_df[vNum] < vNum_lower)].index)

new_df = remove_outliers('V14', new_df)
new_df = remove_outliers('V12', new_df)
new_df = remove_outliers('V10', new_df)

f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))

colors = ['#B3F9C5', '#f9c5b3']
# Boxplots with outliers removed
# Feature V14
sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1, palette=colors)
ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)
ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

# Feature 12
sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)
ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)
ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

# Feature V10
sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)
ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)
ax3.annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)


plt.show()

'''
Although the subsample is pretty small, the t-SNE algorithm is able to detect clusters pretty accurately 
in every scenario (I shuffle the dataset before running t-SNE)
'''

X = new_df.drop('Class', axis=1)
y = new_df['Class']

'''
T-SNE implementation
https://www.youtube.com/watch?v=NEaUSP4YerM
'''
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print 'T-SNE took %.2f'%(t1-t0)

# PCA
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
print 'PCA took %.2f'%(t1-t0)

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()
print "Truncated SVD took %.2f"%(t1-t0)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
f.suptitle('Clusters using Dimension Reduction')

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

# t-SNE scatter
def plot_reduced_scatter(ax, data, title):
    ax.scatter(data[:, 0], data[:, 1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidth=2)
    ax.scatter(data[:, 0], data[:, 1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidth=2)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(handles=[blue_patch, red_patch])

plot_reduced_scatter(ax1, X_reduced_tsne, 't-SNE')
plot_reduced_scatter(ax2, X_reduced_pca, 'PCA')
plot_reduced_scatter(ax3, X_reduced_svd, 'Truncated SVD')
plt.show()


'''
Classifications
'''
X = new_df.drop('Class', axis=1)
y = new_df['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

classifiers = {
    "LogisticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print "Classifier %s has a training score of %.2f" % (classifier.__class__.__name__, training_score.mean())


from sklearn.model_selection import GridSearchCV

log_reg_params = {"penalty": ['l1', 'l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
log_reg = grid_log_reg.best_estimator_

knears_params = {"n_neighbors":list(range(2,5,1)),
                 "algorithm":['auto','ball_tree','kd_tree','brute']
                 }
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
knears_neighbors = grid_knears.best_estimator_

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel':['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)
svc = grid_svc.best_estimator_

tree_params = {'criterion': ['gini', 'entropy'],
               'max_depth': list(range(2,4,1)),
               'min_samples_leaf': list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)
tree_clf = grid_tree.best_estimator_


# Overfitting Case
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print 'Logistic regression cross val score: %.2f'%log_reg_score.mean()

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print 'K NN cross val score: %.2f'%knears_score.mean()

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print 'SVC cross val score: %.2f'%svc_score.mean()

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print 'Decision tree cross val score: %.2f'%tree_score.mean()

undersample_X = df.drop('Class', axis=1)
undersample_y = df['Class']

for train_index, test_index in sss.split(undersample_X, undersample_y):
    print("Train:", train_index, "Test:", test_index)
    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]

undersample_Xtrain = undersample_Xtrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values

undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []

# Implementing NearMiss Technique
# Distribution of NearMiss (Just to see how it distributes the labels we won't use these variables)
X_nearmiss, y_nearmiss = NearMiss().fit_sample(undersample_X.values, undersample_y.values)
print 'NearMiss label distribution:', Counter(y_nearmiss)

for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
    '''
    Why is it still called 'undersample'?!
    '''
    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg)
    # TODO: SMOTE happens during Cross Validation not before...

    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])

    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14), sharey=True)
plt.ylim((0.87, 1.01))

def plot_learning_curve(ax, estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1,1.,5)):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='#ff9124')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='#2492ff')
    ax.plot(train_sizes, train_scores_mean, 'o-', color='#ff9124', label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='#2492ff', label='Cross-validation score')
    ax.set_title('Logistic Regression Learning Curve', fontsize=14)
    ax.set_xlabel('Training size (m)')
    ax.set_ylabel('Score')
    ax.grid(True)
    ax.legend(loc='best')


cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(ax1, log_reg, X_train, y_train, cv=cv, n_jobs=4)
plot_learning_curve(ax2, knears_neighbors, X_train, y_train, cv=cv, n_jobs=4)
plot_learning_curve(ax3, svc, X_train, y_train, cv=cv, n_jobs=4)
plot_learning_curve(ax4, tree_clf, X_train, y_train, cv=cv, n_jobs=4)
plt.show()

from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method='decision_function')
knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)
svc_pred = cross_val_predict(svc, X_train, y_train, cv=5, method='decision_function')

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)

from sklearn.metrics import roc_auc_score

print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))

log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)


def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16, 8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr,
             label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr,
             label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr,
             label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                 arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                 )
    plt.legend()


graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)
plt.show()

