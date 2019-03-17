
import numpy as np
import pandas as pd
import Utils
pd.set_option('display.width', 300)


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.plot([0,1,2], [3,4,5])
plt.show()
quit()

data_folder = 'raw_data/'

from collections import Counter

# 'agriculture, construction, transport_utilities, prof_scientific_waste'
# occupations_pd = pd.read_csv(data_folder + '/' + 'industry_occupation.csv')
# print occupations_pd.columns[3:-1].values
# print occupations_pd.head(10)
#
# occupations_pd = occupations_pd['fips','year','agriculture']
#
# quit()


# chemicals_pd = pd.read_csv(data_folder + '/' + 'chemicals.csv')
# print 'chemicals_pd.shape:', chemicals_pd.shape
# print set(chemicals_pd['fips'])

# earnings_pd = pd.read_csv(data_folder + '/' + 'earnings.csv')
# print 'earnings_pd.shape:', earnings_pd.shape
# print 'earnings_pd.columns:', earnings_pd.columns

education_pd = pd.read_csv(data_folder + '/' + 'education_attainment.csv')
print education_pd.head()
# education_pd = education_pd.groupby('fips', axis=0, as_index=False).mean()
# print education_pd.head()
# print 'education_pd.shape:', education_pd.shape
# print education_pd.columns
# print len(list(set(education_pd['fips'])))

# print set(education_pd['fips'])


# pick top chemicals!
chemicals_pd = pd.read_csv(data_folder + '/' + 'chemicals.csv', index_col=False)
print set(chemicals_pd['pop_served'].values)



quit()
chemicals_pd = chemicals_pd[['chemical_species','pop_served', 'value', 'year', 'fips']]
print chemicals_pd
df = chemicals_pd

df.columns = df.columns.droplevel(1)
print df
quit()

chemicals_pd['averaged_value'] = chemicals_pd.map(lambda x: np.average(x['value'], weights=x['pop_served']))

quit()



quit()




X_train,X_test,y_train,y_test = train_test_split(
            df[df.columns.difference(['agriculture'])], df['agriculture'], test_size=0.2, random_state=42)

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()



# join earnings!
# earnings_pd = pd.read_csv(data_folder + '/' + 'earnings.csv')
# # earnings_pd = earnings_pd[['fips','']]
# print 'earnings_pd.shape:', earnings_pd.shape
# df = df.merge(earnings_pd, how='left', on=['fips','year'])
# # print df.head(30)
# print 'total number of rows after mering earnings:', df.shape[0]



# join educations!
education_pd = pd.read_csv(data_folder + '/' + 'education_attainment.csv')
education_pd = education_pd[['fips','less_than_hs','hs_diploma','some_college_or_associates','college_bachelors_or_higher','pct_hs_diploma','pct_college_or_associates','pct_college_bachelors_or_higher']]
education_pd = education_pd.groupby('fips', axis=0, as_index=False).mean()
print 'education_pd.shape:', education_pd.shape
df = df.merge(education_pd, how='left', on=['fips'])
print 'total number of rows after mering education:', df.shape[0]


# join occupations!


print df.columns
print df.head(30)
# What I do have now:
quit()


chemicals_list = list(set(chemicals_pd['chemical_species']))
industry_list = list(set(occupations_pd.columns[3:-1].values))
for chemical in chemicals_list:
    for industry in industry_list:
        print chemical, industry
        print 'corr between %s and %s:'%(chemical, industry), df[[chemical,industry]].corr()
quit()




def mapping_contaminations(some_str):
    my_dict = {'Greater than MCL':3, 'Less than or equal MCL':1, 'Non Detect':0, '':0.5}
    return my_dict[some_str]
print 'total number of rows:', df.shape[0]
print 'with whole data:', df.dropna().shape[0]
print 'df.columns:', df.columns

print 'Now statistics!'
df['mapped_contaminations'] = df['contaminant_level'].map(lambda x: mapping_contaminations(x))
print df[['mapped_contaminations', 'total_med']].corr()
print df[['mapped_contaminations', 'some_college_or_associates']].corr()
print df[['mapped_contaminations', 'total_employed']].corr()