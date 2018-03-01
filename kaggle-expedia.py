import pandas as pd
import random


destinations = pd.read_csv('destinations.csv')
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

train.shape

destinations.shape

#explore the first few rows of the data
train.head(5)

test.head(5)

#Exploring hotel clusters
train["hotel_cluster"].value_counts()

#Create a set of all the unique test user ids.
#Create a set of all the unique train user ids.
#Figure out how many test user ids are in the train user ids.
#See if the count matches the total number of test user ids.
test_ids = set(test.user_id.unique())
train_ids = set(train.user_id.unique())
intersection_count = len(test_ids & train_ids)
intersection_count == len(test_ids)


#Downsampling our Kaggle data

#--Add in times and dates
train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month


unique_users_id = train.user_id.unique()

sel_users_id = random.sample(list(unique_users_id),10000)
sel_train = train[train.user_id.isin(sel_users_id)]





















