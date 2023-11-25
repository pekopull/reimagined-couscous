from Model.train import model0
from Model.train import df_public
from Model.train import columns_to_drop
import numpy as np
import pandas as pd
import sys
from catboost import CatBoostClassifier, Pool, cv


model0.save_model(sys.argv[5])


# Define the new threshold
new_threshold = 0.5
predicted_probabilities = model0.predict_proba(df_public.drop(columns_to_drop+['label'],axis=1))
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
ans.to_csv(sys.argv[6])
ans.head(5)