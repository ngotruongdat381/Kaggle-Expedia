import pandas as pd
import random
from sklearn.decomposition import PCA


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


## Pick 10000 users
unique_users_id = train.user_id.unique()

sel_users_id = random.sample(list(unique_users_id),10000)
sel_train = train[train.user_id.isin(sel_users_id)]

## Pick new training and testing sets
t1 = sel_train[((sel_train.year == 2013) | ((sel_train.year == 2014) & (sel_train.month < 8)))]
t2 = sel_train[((sel_train.year == 2014) & (sel_train.month >= 8))]

## Remove click events
t2 = t2[t2.is_booking == True]

most_common_clusters = list(train.hotel_cluster.value_counts().head().index)

predictions = [most_common_clusters for i in range(t2.shape[0])]


#[destinations for i in range(10)]



#PCA 
pca = PCA(n_components=3)
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small)
dest_small["srch_destination_id"] = destinations["srch_destination_id"]






def calc_fast_features(df):
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")
    
    props = {}
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)
    
    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = df[prop]
    
    date_props = ["month", "day", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')
        
    ret = pd.DataFrame(props)
    
    ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret

df = calc_fast_features(t1)
df.fillna(-1, inplace=True)


# Machine learning
predictors = [c for c in df.columns if c not in ["hotel_cluster"]]
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
scores = cross_validation.cross_val_score(clf, df[predictors], df['hotel_cluster'], cv=3)
scores







































