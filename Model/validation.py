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


from Preprocess.preprocess import df_train
from Preprocess.preprocess import df_public

columns_to_drop = ['txkey']
# Split the data into feature (X) and label (y) data
# df_train_new = df_train[df_train['locdt'] >= 0]
X_train = df_train[df_train['locdt'] < int(sys.argv[4])].drop(['label']+columns_to_drop, axis=1)
X_test = df_train[df_train['locdt'] >= int(sys.argv[4])].drop(['label']+columns_to_drop, axis=1)
y_train = df_train[df_train['locdt'] < int(sys.argv[4])]['label']
y_test = df_train[df_train['locdt'] >= int(sys.argv[4])]['label']

print("# training data: {:d}\n# test data: {:d}".format(len(X_train), len(X_test)))

label_counts = y_train.value_counts()
print("(Normal data)Label 0 Count:", label_counts[0])
print("(Fraud data) Label 1 Count:", label_counts[1])
print("ratio:",label_counts[0]/label_counts[1])

label_counts = y_test.value_counts()
print("(Normal data)Label 0 Count:", label_counts[0])
print("(Fraud data) Label 1 Count:", label_counts[1])
print("ratio:",label_counts[0]/label_counts[1])



cat_features = ['chid','cano','contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'bnsfg', 'stocn', 'scity','csmcu',  'ovrlt', 'flbmk','hcefg','flg_3dsmk']
#cat_features = ['chid','cano','contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'stocn', 'scity', ]
cat_features = cat_features + ['time_of_day','user_most_frequent_time_of_day','user_most_frequent_merchant','user_most_frequent_merchant_type']

# cat_features = ['chid','cano','contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'stocn', 'scity', ]
# cat_features = cat_features + ['time_of_day','user_most_frequent_time_of_day']

train_pool = Pool(X_train,y_train,cat_features=cat_features,feature_names=list(X_train),timestamp=X_train['locdt'])
valid_pool = Pool(X_test,y_test,cat_features=cat_features,feature_names=list(X_train),timestamp=X_test['locdt'])

catboost_params = {
    'iterations': 10000,
    'eval_metric': 'F1', #MCC, precision, recall f1
    'task_type': 'GPU',
    'early_stopping_rounds': 300,
    #'use_best_model': True,
    'verbose': 20,
    'learning_rate': 0.054,
   # 'model_shrink_rate': 0.01
    #'scale_pos_weight': 5,
    'has_time':True,
    'depth':6,
    #'l2_leaf_reg':8,
    'grow_policy':"Lossguide",
    #'border_count':254,
    'class_weights':{0: 2.0, 1: 1.0},
    'boosting_type': "Plain",

}

model0 = CatBoostClassifier(**catboost_params)
model0.fit(train_pool, eval_set=valid_pool, verbose=10)

# Get model predictions once
y_pred = model0.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



# Assuming y_test contains the true labels (0 or 1)
y_pred = model0.predict_proba(X_test)[:, 1]  # Predicted probabilities for class 1

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# Calculate F1 scores for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall)

# Find the threshold that maximizes the F1 score
best_threshold = thresholds[np.argmax(f1_scores)]
best_precision = precision[np.argmax(f1_scores)]
best_recall = recall[np.argmax(f1_scores)]
best_f1 = np.max(f1_scores)

print('Best threshold:', best_threshold)
print('Best Precision:', best_precision)
print('Best Recall:', best_recall)
print('Default f1:',f1_score(y_test, model0.predict(X_test)))
print('Best F1 Score:', best_f1)
