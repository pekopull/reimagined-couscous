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

#from Preprocess.preprocess import df_train
from Preprocess.preprocess import df_public

print("start loading model")
model_path = sys.argv[6]  # Specify your model path
loaded_model = CatBoostClassifier()
loaded_model.load_model(model_path)
print(loaded_model)
print(loaded_model.get_all_params())
print("load model finished")

columns_to_drop = ['txkey']



# Define the new threshold
new_threshold = 0.5
# print length of df_public
print(len(df_public))
predicted_probabilities = loaded_model.predict_proba(df_public.drop(columns_to_drop+['label'],axis=1))
# Apply the new threshold and convert probabilities to binary predictions
adjusted_predictions = (predicted_probabilities[:, 1] > new_threshold).astype(int)

# Count the occurrences of 0 and 1 in the adjusted predictions
num_zeros = np.sum(adjusted_predictions == 0)
num_ones = np.sum(adjusted_predictions == 1)

# Print the counts
print(f"Number of Predictions (0): {num_zeros}")
print(f"Number of Predictions (1): {num_ones}")

ans = pd.DataFrame(data={'txkey': df_public['txkey'], 'pred': adjusted_predictions})
ans=ans.set_index('txkey')
ans.to_csv(sys.argv[7])
ans.head(5)