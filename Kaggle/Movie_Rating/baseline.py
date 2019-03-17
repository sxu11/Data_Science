
import pandas as pd
import os

data_folder = 'predict-movie-ratings-v22'
train_df = pd.read_csv(os.path.join(data_folder, 'train.csv'))

'''
Mean in the train set
'''
rating_mean = train_df['rating'].mean()

'''
All test movies
'''
test_df = pd.read_csv(os.path.join(data_folder, 'test.csv'))
test_movies = set(test_df['movie'].values.tolist())

movie2rating = {}
for test_movie in test_movies:
    '''
    Here can be introduced...
    '''
    cur_rating = train_df[train_df['movie']==test_movie]['rating'].mean()
    movie2rating[test_movie] = cur_rating
print movie2rating
quit()
'''
Some movies in the 
'''
test_df['rating'] = test_df['movie'].apply(lambda x: movie2rating.get(x,rating_mean))

'''
There are some test movies that do not exist in the train set
'''
test_df['rating'] = test_df['rating'].fillna(3.51241660044)

test_df[['ID', 'rating']].to_csv('submission.csv', index=False, float_format='%.2f')