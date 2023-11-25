import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gc
import re
import os
import time
import sys
from catboost import CatBoostClassifier, Pool, cv
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score,average_precision_score, confusion_matrix
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve


from Preprocess.load import df_train
from Preprocess.load import df_public

print("preprocess.py start")

df_combined = pd.concat([df_train, df_public], axis=0)
df_combined.sort_values(['cano', 'locdt'], inplace=True)

# free up memory
del df_train
del df_public
gc.collect()
df_train = pd.DataFrame()
df_public = pd.DataFrame()

################################################################################################
#
# Feature Engineering: Time-of-Day
#
################################################################################################

print("Feature Engineering: Time-of-Day")
# Assuming df_train is your training data DataFrame and df_public is your testing data DataFrame

# # Combine df_train and df_public into one DataFrame for feature engineering
# df_combined = pd.concat([df_train, df_public], axis=0)

# Extract hour from the "loctm" column
df_combined['loctm'] = df_combined['loctm'].astype(str).str.zfill(6)  # Ensure the "loctm" column is in HHMMSS format
df_combined['hour'] = df_combined['loctm'].str[:2].astype(int)

# Create a more detailed time-of-day feature
def time_of_day(hour):
    if 1 <= hour < 5:
        return 'Late Night'
    elif 5 <= hour < 8:
        return 'Early Morning'
    elif 8 <= hour < 11:
        return 'Morning'
    elif 11 <= hour < 13:
        return 'Noon'
    elif 13 <= hour < 17:
        return 'AfterNoon'
    elif 17 <= hour < 19:
        return 'Sunset'
    elif 19 <= hour < 22:
        return 'Night'
    else:
        return 'Midnight'

df_combined['loctm'] = df_combined['loctm'].astype(int)
df_combined['time_of_day'] = df_combined['hour'].apply(time_of_day)

df_combined['time_of_day'] = df_combined['time_of_day'].astype('category')

# Calculate the most frequent detailed time-of-day category for each user

user_most_frequent_time_of_day = df_combined.groupby('cano')['time_of_day'].apply(lambda x: x.mode().iloc[0]).reset_index()
user_most_frequent_time_of_day.rename(columns={'time_of_day': 'user_most_frequent_time_of_day'}, inplace=True)
df_combined = df_combined.merge(user_most_frequent_time_of_day, on='cano', how='left')
del user_most_frequent_time_of_day

df_combined.drop(columns=['hour'], inplace=True)

################################################################################################
#
# Feature Engineering: Transaction Statistics
#
################################################################################################
print("Feature Engineering: Transaction Statistics")
# Transaction Amount Percentiles
percentiles = [10, 25, 75]
percentile_values = df_combined.groupby('cano')['conam'].quantile([0.1, 0.25, 0.75]).unstack(level=1).reset_index()
percentile_values.columns = ['cano'] + [f'card_conam_percentile_{p}' for p in percentiles]
df_combined = df_combined.merge(percentile_values, on='cano', how='left')
del percentile_values
gc.collect()
percentile_values = pd.DataFrame()

# User's Transaction Amount History Statistics
user_stats = df_combined.groupby('cano')['conam'].agg(['mean', 'median', 'std','max','sum']).reset_index()
user_stats.columns = ['cano', 'card_mean_transaction_amount', 'card_median_transaction_amount', 'card_std_transaction_amount','card_max_conam','card_sum_conam']
df_combined = df_combined.merge(user_stats, on='cano', how='left')
del user_stats
gc.collect()
user_stats = pd.DataFrame()

################################################################################################
#
# Feature Engineering: Transaction Frequency
#
################################################################################################

print("Feature Engineering: Transaction Frequency")
# Number of Transactions by User 使用者總消費次數
user_transaction_counts = df_combined.groupby('cano')['chid'].count().reset_index()
user_transaction_counts.rename(columns={'chid': 'user_transaction_count'}, inplace=True)
df_combined = df_combined.merge(user_transaction_counts, on='cano', how='left')
del user_transaction_counts

#Calculate historical transaction frequencies for each user 使用者每一天消費多少次
historical_user_transaction_frequency = df_combined.groupby(['cano', 'locdt'])['chid'].count().reset_index()
historical_user_transaction_frequency.rename(columns={'chid': 'historical_transaction_frequency'}, inplace=True)
df_combined = df_combined.merge(historical_user_transaction_frequency, on=['cano', 'locdt'], how='left')

# # Calculate user transaction frequency based on date 使用者總交易天數
user_transaction_frequency = df_combined.groupby('cano')['locdt'].nunique().reset_index()
user_transaction_frequency.rename(columns={'locdt': 'user_transaction_frequency'}, inplace=True)
df_combined = df_combined.merge(user_transaction_frequency, on='cano', how='left')
del user_transaction_frequency
# # Calculate the standard deviation of historical transaction frequencies for each user 使用者每一天消費多少次的標準差
user_transaction_frequency_std = historical_user_transaction_frequency.groupby('cano')['historical_transaction_frequency'].std().reset_index()
user_transaction_frequency_std.columns = ['cano', 'user_transaction_frequency_std']
df_combined = df_combined.merge(user_transaction_frequency_std, on='cano', how='left')

del user_transaction_frequency_std
del historical_user_transaction_frequency

user_transaction_velocity = df_combined.groupby('cano')['locdt'].nunique() / df_combined.groupby('cano')['locdt'].count()
user_transaction_velocity = user_transaction_velocity.reset_index()
user_transaction_velocity.columns = ['cano', 'user_transaction_velocity']
df_combined = df_combined.merge(user_transaction_velocity, on='cano', how='left')
del user_transaction_velocity

################################################################################################
#
# Feature Engineering: Last & Unique Transaction
#
################################################################################################
print("Feature Engineering: Last & Unique Transaction")
df_combined['card_last_transaction_locdt'] = df_combined['locdt'] - df_combined.groupby('cano')['locdt'].shift()
df_combined['card_last_same_stocn_transaction_locdt'] = df_combined['locdt'] - df_combined.groupby(['cano','stocn'])['locdt'].shift()
df_combined['card_last_same_mcc_transaction_locdt'] = df_combined['locdt'] - df_combined.groupby(['cano','mcc'])['locdt'].shift()
df_combined['card_last_same_mchno_transaction_locdt'] = df_combined['locdt'] - df_combined.groupby(['cano','mchno'])['locdt'].shift()
df_combined['card_last_same_ecfg_transaction_locdt'] = df_combined['locdt'] - df_combined.groupby(['cano','ecfg'])['locdt'].shift()
df_combined['card_last_same_etymd_transaction_locdt'] = df_combined['locdt'] - df_combined.groupby(['cano','etymd'])['locdt'].shift()
df_combined['stocn_nunique_of_locdt'] = df_combined.groupby(['cano', 'locdt'])['stocn'].transform('nunique')
df_combined['scity_nunique_of_locdt'] = df_combined.groupby(['cano', 'locdt'])['scity'].transform('nunique')
df_combined['stscd_nunique_of_locdt'] = df_combined.groupby(['cano', 'locdt'])['stscd'].transform('nunique')
df_combined['time_nunique_of_locdt'] = df_combined.groupby(['cano', 'locdt'])['time_of_day'].transform('nunique')
df_combined['stocn_nunique_of_ecfg'] = df_combined.groupby(['cano', 'ecfg'])['stocn'].transform('nunique')
df_combined['stscd_nunique_of_ecfg'] = df_combined.groupby(['cano', 'ecfg'])['stscd'].transform('nunique')
df_combined['mcc_nunique_of_ecfg'] = df_combined.groupby(['cano', 'ecfg'])['mcc'].transform('nunique')
df_combined['mchno_nunique_of_ecfg'] = df_combined.groupby(['cano', 'ecfg'])['mchno'].transform('nunique')
df_combined['time_nunique_of_ecfg'] = df_combined.groupby(['cano', 'ecfg'])['time_of_day'].transform('nunique')
################################################################################################
#
# Feature Engineering: Percentage of stocn/ecfg, most frequent X
#
################################################################################################
print("Feature Engineering: Percentage of stocn/ecfg, most frequent X")
# Percentage of Taiwan Transactions by User
user_taiwan_transaction_percentage = df_combined.groupby('cano')['stocn'].apply(lambda x: (x == 0).mean()).reset_index()
user_taiwan_transaction_percentage.rename(columns={'stocn': 'user_taiwan_transaction_percentage'}, inplace=True)
df_combined = df_combined.merge(user_taiwan_transaction_percentage, on='cano', how='left')
del user_taiwan_transaction_percentage


# Percentage of Internet Transactions by User
user_internet_transaction_percentage = df_combined.groupby('cano')['ecfg'].apply(lambda x: (x == '0').mean()).reset_index()
user_internet_transaction_percentage.rename(columns={'ecfg': 'user_internet_transaction_percentage'}, inplace=True)
df_combined = df_combined.merge(user_internet_transaction_percentage, on='cano', how='left')
del user_internet_transaction_percentage


# These 2 Can Remove because require too much computation:
# # User's Most Frequent Merchant
user_most_frequent_merchant = df_combined.groupby('cano')['mchno'].apply(lambda x: x.mode().iloc[0]).reset_index()
user_most_frequent_merchant.rename(columns={'mchno': 'user_most_frequent_merchant'}, inplace=True)
df_combined = df_combined.merge(user_most_frequent_merchant, on='cano', how='left')
del user_most_frequent_merchant

# # User's Most Frequent Merchant Type
user_most_frequent_merchant_type = df_combined.groupby('cano')['mcc'].apply(lambda x: x.mode().iloc[0]).reset_index()
user_most_frequent_merchant_type.rename(columns={'mcc': 'user_most_frequent_merchant_type'}, inplace=True)
df_combined = df_combined.merge(user_most_frequent_merchant_type, on='cano', how='left')
del user_most_frequent_merchant_type

# # Handle NaN values and convert them to -1 in 'user_most_frequent_merchant' and 'user_most_frequent_merchant_type'
df_combined['user_most_frequent_merchant'].fillna(-1, inplace=True)
df_combined['user_most_frequent_merchant_type'].fillna(-1, inplace=True)
df_combined['user_most_frequent_merchant'] = df_combined['user_most_frequent_merchant'].astype('category')
df_combined

################################################################################################
#
# Feature Engineering: Merchant (Transaction Statistics)
#
################################################################################################
print("Feature Engineering: Merchant (Transaction Statistics)")
# Merchant conam History Statistics
merchant_aggregates = df_combined.groupby('mchno')['conam'].agg(['mean', 'median', 'std']).reset_index()
merchant_aggregates.columns = ['mchno', 'merchant_mean_transaction_amount', 'merchant_median_transaction_amount', 'merchant_std_transaction_amount']
df_combined = df_combined.merge(merchant_aggregates, on='mchno', how='left')
del merchant_aggregates

# Merchant Popularity
merchant_popularity = df_combined.groupby('mchno')['cano'].nunique().reset_index()
merchant_popularity.columns = ['mchno', 'merchant_user_count']
df_combined = df_combined.merge(merchant_popularity, on='mchno', how='left')
del merchant_popularity

# Transaction Amount Percentiles for Merchant
merchant_percentiles = [10,  25, 75]
merchant_percentile_values = df_combined.groupby('mchno')['conam'].quantile([0.1, 0.25, 0.75]).unstack(level=1).reset_index()
merchant_percentile_values.columns = ['mchno'] + [f'merchant_conam_percentile_{p}' for p in merchant_percentiles]
df_combined = df_combined.merge(merchant_percentile_values, on='mchno', how='left')
del merchant_percentile_values
df_combined

################################################################################################
#
# Feature Engineering: Merchant (Transaction Frequency)
#
################################################################################################
print("Feature Engineering: Merchant (Transaction Frequency)")

# Calculate merchant transaction counts
merchant_transaction_counts = df_combined.groupby('mchno')['chid'].count().reset_index()
merchant_transaction_counts.rename(columns={'chid': 'merchant_transaction_count'}, inplace=True)
df_combined = df_combined.merge(merchant_transaction_counts, on='mchno', how='left')
del merchant_transaction_counts


merchant_transaction_velocity = df_combined.groupby('mchno')['locdt'].nunique() / df_combined.groupby('mchno')['locdt'].count()
merchant_transaction_velocity = merchant_transaction_velocity.reset_index()
merchant_transaction_velocity.columns = ['mchno', 'merchant_transaction_velocity']
df_combined = df_combined.merge(merchant_transaction_velocity, on='mchno', how='left')
del merchant_transaction_velocity

################################################################################################
#
# Done!
#
################################################################################################

import gc
gc.collect()


# List of columns to convert to 'category' data type
object_columns_to_category = [
    'cano', 'chid',  'mchno', 'acqic', 'user_most_frequent_time_of_day'
]


df_combined[object_columns_to_category] = df_combined[object_columns_to_category].astype('category')

# Split the combined DataFrame back into df_train and df_public based on locdt
df_public = df_combined[df_combined['locdt'] >= int(sys.argv[3])]
df_train = df_combined[df_combined['locdt'] < int(sys.argv[3])]


# Reset the index of the DataFrames
df_train.reset_index(drop=True, inplace=True)
df_public.reset_index(drop=True, inplace=True)




print(df_public.dtypes)

del df_combined
gc.collect()

print("preprocess.py done")

