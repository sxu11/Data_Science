
import pandas as pd
pd.set_option('display.width', 10000)
import numpy as np

from spotlight.interactions import Interactions
from spotlight.sequence.implicit import ImplicitSequenceModel

df_train = pd.read_csv('data/patient_items_50000_sample_train.csv')\
    .sort_values(['patient_id', 'item_date'])
print 'df_train.shape:', df_train.shape

pat_ids_train = df_train['patient_id'].drop_duplicates().values.tolist()
print 'number of train patients: ', len(pat_ids_train)

'''
I'm really amazed by this pandas trick, tho it is not useful:

grouped_pat_items = df.groupby('patient_id')['clinical_item_id'].apply(list).reset_index()
'''


'''
Prepare training data
'''
users_train = df_train['patient_id'].values
items_train = df_train['clinical_item_id'].values
times_train = df_train['item_date'].values


'''
Feed training data into model for training
'''
implicit_interactions = Interactions(users_train,
                                     items_train,
                                     timestamps=times_train)
sequential_interaction = implicit_interactions.to_sequence()
implicit_sequence_model = ImplicitSequenceModel()
implicit_sequence_model.fit(sequential_interaction)



'''
Testing
'''
df_test = pd.read_csv('data/patient_items_50000_sample_test.csv')\
    .sort_values(['patient_id', 'item_date'])
print 'df_test.shape:', df_test.shape

pat_ids_test = df_test['patient_id'].drop_duplicates().values.tolist()
print 'number of test patients: ', len(pat_ids_test)

'''
Check: for each patient, how many orders fall into 0-4 hrs? 
How many fall into 4-12 hrs?
'''

from datetime import datetime, timedelta
def split_data_by_hour4(df_cur):
    df_cur = df_cur.copy().reset_index()

    df_cur['item_date'] = df_cur['item_date'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    )
    first_date = df_cur.ix[0, 'item_date']
        # df_cur[['patient_id','item_date']].groupby('patient_id')\
        #             .first().values[0]
    df_cur['seconds_rela'] = df_cur['item_date'].apply(
        lambda x: (x-first_date).total_seconds()
    )

    within_4_hours = df_cur[df_cur['seconds_rela'] < 4*3600]
    between_4_and_12_hours = df_cur[(4 * 3600 < df_cur['seconds_rela'])
                                        & (df_cur['seconds_rela'] < 12 * 3600)]
    return within_4_hours['clinical_item_id'].values.tolist(), \
           between_4_and_12_hours['clinical_item_id'].values.tolist()


def get_top_k(scores, k=10):
    return [x+1 for x in np.argsort(scores)[-k:]]

def precision_at_k(actuals, predicts):
    num_relevant = 0
    for predict in predicts:
        if predict in actuals:
            num_relevant += 1
    return float(num_relevant)/len(predicts)

Achieved_fractions = []
for pat_id in pat_ids_test:
    df_cur = df_test[df_test['patient_id']==pat_id]
    # cur_orders = cur_rows['clinical_item_id'].values.tolist()

    '''
    Needs optimization!
    '''
    # query_set = [cur_orders[0]]
    # query_set = [item_mapping.get(x,0) for x in query_set]

    # valid_set = cur_orders[1:]

    query_set, valid_set = split_data_by_hour4(df_cur)
    # valid_set = [item_mapping.get(x,0) for x in valid_set]

    if len(valid_set)==0:
        continue

    all_recommends = implicit_sequence_model.predict(query_set)
    top_k_recommends = get_top_k(all_recommends, k=10)


    print 'Best achievable precision_at_k:', len(valid_set)/10.,
    print 'Precision_at_k', precision_at_k(actuals=valid_set, predicts=top_k_recommends),

    Achieved_fraction = precision_at_k(actuals=valid_set, predicts=top_k_recommends)/(len(valid_set)/10.)
    print 'Achieved fraction:', Achieved_fraction

    Achieved_fractions.append(Achieved_fraction)
    # print len()
print 'average achieved_fraction:', sum(Achieved_fractions)/len(Achieved_fractions)