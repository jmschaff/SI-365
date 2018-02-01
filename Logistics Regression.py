# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt

# link to metadata: https://www.kaggle.com/weil41/flights/data 
airlines2015 = pd.read_csv("airlines.csv")
airports2015 = pd.read_csv("airports.csv")
flights2015 = pd.read_csv("flights.csv", dtype={"ORIGIN_AIRPORT":str, "DESTINATION_AIRPORT":str})
flights2016 = pd.read_csv("2016_cleaned.csv")
flights2017 = pd.read_csv("current_flights.csv")
flights2015_original = pd.read_csv("flights.csv", dtype={"ORIGIN_AIRPORT":str, "DESTINATION_AIRPORT":str})

##### Add Lat and Long of airport to dataframe
flights2015_airport = flights2015_original
flights2015_airport['LATITUDE'] = 0
flights2015_airport['LONGITUDE'] = 0

# figure out which flights are delayed
# make a delayed variable
cols = ['DAY_OF_WEEK', 'MONTH', 'DAY', 'AIRLINE', 'DISTANCE', 'SCHEDULED_DEPARTURE', 'SCHEDULED_TIME', 'DELAYED']
col_types = {'DAY_OF_WEEK': float, 'MONTH': float, 'DAY': float, 'AIRLINE': str, 'DISTANCE': float, 'SCHEDULED_DEPARTURE': float, 'SCHEDULED_TIME': float, 'DEPARTURE_DELAY': float}

flights2015['DELAYED'] = 0.0
flights2015.loc[flights2015['DEPARTURE_DELAY'] > 10, 'DELAYED'] = 1.0

flights2016['DELAYED'] = 0
flights2016.loc[flights2016['DEPARTURE_DELAY'] > 10, 'DELAYED'] = 1.0

flights2017['DELAYED'] = 0.0
flights2017.loc[flights2017['DEPARTURE_DELAY'] > 10, 'DELAYED'] = 1.0

flights2015 = flights2015[cols]
flights2016 = flights2016[cols]
flights2017 = flights2017[cols]

flights2015 = flights2015[np.isfinite(flights2015['SCHEDULED_TIME'])]

##### ******** make dataset only delays
flights2016_delay = flights2016[flights2016['DELAYED'] == 1]
flights2015_delay = flights2015[flights2015['DELAYED'] == 1]
flights2015_ontime = flights2015[flights2015['DELAYED'] == 0]

#Preprocessing Datasets
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#LabelEncoder 2015
labelencoder_X = LabelEncoder()
flights2015.iloc[:,3] = labelencoder_X.fit_transform(flights2015.values[:, 3])

#OneHotEncoder 2015
flights2015 = flights2015.astype(float)
flights2015_array = flights2015.values
flights2015_array = flights2015_array[~np.isnan(flights2015_array).any(axis=1)]
categ = [cols.index(x) for x in ['DAY_OF_WEEK', 'MONTH', 'DAY', 'AIRLINE']]
enc = OneHotEncoder(categorical_features = categ)
flights2015_array = enc.fit_transform(flights2015_array).toarray()

#LabelEncoder 2016
labelencoder_X = LabelEncoder()
flights2016.iloc[:,3] = labelencoder_X.fit_transform(flights2016.values[:, 3])

#OneHotEncoder 2016
flights2016 = flights2016.astype(float)
flights2016_array = flights2016.values
flights2016_array = flights2016_array[~np.isnan(flights2016_array).any(axis=1)]
categ = [cols.index(x) for x in ['DAY_OF_WEEK', 'MONTH', 'DAY', 'AIRLINE']]
enc = OneHotEncoder(categorical_features = categ)
flights2016_array = enc.fit_transform(flights2016_array).toarray()

#Preprocessing for 2016 delay data

#LabelEncoder 2016
labelencoder_X = LabelEncoder()
flights2016_delay.iloc[:,3] = labelencoder_X.fit_transform(flights2016_delay.values[:, 3])

#OneHotEncoder 2016
flights2016_delay = flights2016_delay.astype(float)
flights2016_delay_array = flights2016_delay.values
flights2016_delay_array = flights2016_delay_array[~np.isnan(flights2016_delay_array).any(axis=1)]
categ = [cols.index(x) for x in ['DAY_OF_WEEK', 'MONTH', 'DAY', 'AIRLINE']]
enc = OneHotEncoder(categorical_features = categ)
flights2016_delay_array = enc.fit_transform(flights2016_delay_array).toarray()

# #Preprocessing for 2017 data
# #Imputer 2017
# imputer = Imputer(missing_values ='NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(flights2017.values[:,0:2])
# flights2017.iloc[:, 0:2] = imputer.transform(flights2017.values[:,0:2])
 
# #LabelEncoder 2017
# labelencoder_X = LabelEncoder()
# flights2017.iloc[:,3] = labelencoder_X.fit_transform(flights2017.values[:, 3])

# #OneHotEncoder 2017
# flights2017 = flights2017.astype(float)
# flights2017_array = flights2017.values
# flights2017_array = flights2017_array[~np.isnan(flights2017_array).any(axis=1)]
# categ = [cols.index(x) for x in ['DAY_OF_WEEK', 'MONTH', 'DAY', 'AIRLINE']]
# enc = OneHotEncoder(categorical_features = categ)
# flights2017_array = enc.fit_transform(flights2017_array).toarray()

# Required Python Packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    """
    Logistic Regression classifier main
    :return:
    """
    # Load the datasets for training and testing the logistic regression classifier
    dataset = flights2015_array
    dataset2 = flights2016_array
    
    # Split the 2015 data set into X (Includes all columns of training features) 
    # and Y (Includes the target column) arrays to train the model
    train_x2015 = dataset[:, :67]
    train_y2015 = dataset[:, 67]
    
    # Split the 2016 data to test the model accuracy
    test_x2016 = dataset2[:, :67]
    test_y2016 = dataset2[:, 67]
    
    print("test_x2016 size :: ", test_x2016.shape)
    print("test_y2016 size :: ", test_y2016.shape)

    # Create and Training Logistic regression model with 2015 data
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x2015, train_y2015)

    # Test the Logistic regression model with 20016 data 
    test_accuracy = logistic_regression_model.score(test_x2016, test_y2016)

    # Output Model Accuracy
    print("Test Accuracy :: ", test_accuracy)
    
    
if __name__ == "__main__":
    main()
    
##### Random Forest
from sklearn.ensemble import RandomForestClassifier
# Set random seed
np.random.seed(0)

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(flights2015_array[:,:67], flights2015_array[:,67])

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
clf.score(flights2016_array[:,:67], flights2016_array[:, 67])

##### Flights Distributions
##### Distance Distributions
plt.hist(flights2015['DISTANCE'], 90)
plt.xlim(0, 3500)
plt.ylim(0,390000)
plt.title("Distance Distribution of 2015 Flights")
plt.xlabel("Distance of Flight")
plt.ylabel("Frequency")

fig = plt.gcf()

##### On-Time Flights Distributions
plt.hist(flights2015_ontime['DISTANCE'], 90)

fig = plt.gcf()

##### Delayed Flights Distributions
plt.hist(flights2015_delay['DISTANCE'], 90)

fig = plt.gcf()

##### Scheduled Departure Distributions
plt.hist(flights2015['SCHEDULED_DEPARTURE'], 24)
plt.xlim(0, 2400)
plt.ylim(0,450000)
plt.title("Scheduled Departure Distributions of 2015 Flights")
plt.xlabel("Scheduled Departure of Flight")
plt.ylabel("Frequency")

fig = plt.gcf()

##### On-Time Flights Distributions
plt.hist(flights2015_ontime['SCHEDULED_DEPARTURE'], 24)

fig = plt.gcf()

##### Delayed Flights Distributions
plt.hist(flights2015_delay['SCHEDULED_DEPARTURE'], 24)

fig = plt.gcf()

##### Schedulued Time Distributions
plt.hist(flights2015['SCHEDULED_TIME'], 70)
plt.xlim(0, 400)
plt.ylim(0,510000)
plt.title("Scheduled Time Distributions of 2015 Flights")
plt.xlabel("Scheduled Time of Flight")
plt.ylabel("Frequency")

fig = plt.gcf()

##### On-Time Flights Distributions
plt.hist(flights2015_ontime['SCHEDULED_TIME'].dropna(), 70)

fig = plt.gcf()

##### Delayed Flights Distributions
plt.hist(flights2015_delay['SCHEDULED_TIME'].dropna(), 70)

fig = plt.gcf()

##### Schedulued Time Distributions
plt.hist(flights2015['DAY_OF_WEEK'], 7)
plt.xlim(1, 7)
plt.ylim(0,900000)
plt.title("Day of Week Distributions of 2015 Flights")
plt.xlabel("Day of Week of Flight")
plt.ylabel("Frequency")

fig = plt.gcf()

##### On-Time Flights Distributions
plt.hist(flights2015_ontime['DAY_OF_WEEK'].dropna(), 7)

fig = plt.gcf()

##### Delayed Flights Distributions
plt.hist(flights2015_delay['DAY_OF_WEEK'].dropna(), 7)

fig = plt.gcf()

##### Day Distributions
plt.hist(flights2015['DAY'], 31)
plt.xlim(1, 31)
plt.ylim(0,210000)
plt.title("Day of Month Distributions of 2015 Flights")
plt.xlabel("Day of Month of Flight")
plt.ylabel("Frequency")

fig = plt.gcf()

##### On-Time Flights Distributions
plt.hist(flights2015_ontime['DAY'].dropna(), 31)

fig = plt.gcf()

##### Delayed Flights Distributions
plt.hist(flights2015_delay['DAY'].dropna(), 31)

fig = plt.gcf()

##### Month Distributions
plt.hist(flights2015['MONTH'], 12)
plt.xlim(1, 12)
plt.ylim(0,600000)
plt.title("Month Distributions of 2015 Flights")
plt.xlabel("Month of Flight")
plt.ylabel("Frequency")

fig = plt.gcf()

##### On-Time Flights Distributions
plt.hist(flights2015_ontime['MONTH'].dropna(), 12)

fig = plt.gcf()

##### Delayed Flights Distributions
plt.hist(flights2015_delay['MONTH'].dropna(), 12)

fig = plt.gcf()





















