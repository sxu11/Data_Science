

import pandas as pd
import pickle
import my_funcs as mf

train = pd.read_csv('en_train.csv') # play with it

test = pd.read_csv('en_test.csv')

res_filename = 'submission.csv'

NEED_DICT = False
if NEED_DICT:
    dict_filename = 'naive_dict.pickle'
    NEED_TO_TRAIN = False
    if NEED_TO_TRAIN:
        mf.get_dict(train, dict_filename)

    with open(dict_filename, 'r') as handle:
        my_dict = pickle.load(handle)

    ress = mf.predict_test(test, my_dict, res_filename) # list
#res.to_csv('submission.csv',index=False)

ress = zip(test['sentence_id'], test['token_id'], test['before'])

mf.write_to_file(ress, 'submission_naive.csv')