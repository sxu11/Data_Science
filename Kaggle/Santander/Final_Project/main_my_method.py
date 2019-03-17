
import pandas as pd
from itertools import product
import numpy as np
from xgboost import XGBClassifier as XGBC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import utils
import xgboost as xgb

data_folder = 'data/'
#

# 1. FE like online
file_train = data_folder + 'sales_train.csv'
file_test = data_folder + 'test.csv'
train_valid_data, test_df = utils.load_FE_data(file_train, file_test)
lag_features = utils.get_lag_feature()

# 3. training
X_train_valid = train_valid_data[lag_features]
y_train_valid = train_valid_data['label']

## TODO: Keras
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
# https://keras.io/getting-started/sequential-model-guide/

ml_model = 'xgboost'

if ml_model == 'keras':
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.utils.np_utils import to_categorical

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score

    y_train_valid_onehot = to_categorical(y_train_valid)

    def create_model():
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=len(lag_features), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(len(lag_features), activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    clf = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #results = cross_val_score(model, X_train_valid, y_train_valid_onehot, cv=kfold)
    #print results

    clf.fit(X_train_valid, y_train_valid)

    X_test = test_df[lag_features]
    y_test_pred_one_hot = clf.predict(X_test)
    y_test_pred = np.argmax(to_categorical(y_test_pred_one_hot,len(lag_features)))

elif ml_model == 'xgboost':

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid,
                                                        y_train_valid,
                                                        test_size=0.33,
                                                        random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    watchlist=[(dvalid, 'eval'), (dtrain, 'train')]
    num_round = 10

    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = len(lag_features)
    param['silent'] = 0

    use_saved_model = False
    if not use_saved_model:
        bst = xgb.train(param, dtrain, num_round, watchlist)
        bst.save_model('bst.model')
    else:
        bst = xgb.Booster(model_file='bst.model', )

    ## Only when there not sufficient data
    ## https://www.programcreek.com/python/example/75187/sklearn.cross_validation.StratifiedKFold
    # from sklearn.cross_validation import StratifiedKFold
    #
    # skf = StratifiedKFold(y_train_valid, n_folds=5,shuffle=True)
    # for train_inds, valid_inds in skf:
    #


    y_valid_pred = bst.predict(dvalid)

    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    print(classification_report(y_valid, y_valid_pred, target_names=lag_features))
    print confusion_matrix(y_valid, y_valid_pred, labels=range(len(lag_features))),


    X_test = test_df[lag_features]
    dtest = xgb.DMatrix(X_test)
    y_test_pred = bst.predict(dtest)


    # clf = XGBC(objective='multi:softmax', nthread=1, )
    #
    # clf.fit(X_train_valid, y_train_valid)

    #
    # y_test_pred = clf.predict(X_test)

elif ml_model == 'linear-regression':
    ''

##

## TODO: build local system
do_local_test = False
if do_local_test:
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid,
                                                        y_train_valid,
                                                        test_size=0.33,
                                                        random_state=42)
    clf.fit(X_train, y_train)

    y_valid_pred = clf.predict(X_valid)

    # http://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py


##





## TODO: test multi-thread
# from multiprocessing import set_start_method
# set_start_method("forkserver")



##

test_df['label_pred'] = pd.Series(y_test_pred, index=test_df.index)
label_to_lagcols = test_df['label_pred'].apply(lambda x: lag_features[int(round(x))])
res_pred = test_df.lookup(label_to_lagcols.index, label_to_lagcols.values)

test_df['item_cnt_month'] = pd.Series(res_pred, index=test_df.index).clip(0,20)
test_df['ID'] = test_df['ID'].astype('int32')

test_df[['ID', 'item_cnt_month']].to_csv('new_submission_DMatrix.csv', index=False)