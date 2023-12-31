import sys
import pandas as pd
convert_dict = {
    'txkey': 'category',
    'chid': 'category',
    'cano': 'category',
    'contp': 'category',
    'etymd': 'Int64',
    'mchno': 'category',
    'acqic': 'category',
    'mcc': 'Int64',
    'ecfg': 'category',
    'insfg': 'category',
    'bnsfg': 'category',
    'stocn': 'Int64',
    'scity': 'Int64',
    'stscd': 'Int64',
    'ovrlt': 'category',
    'flbmk': 'category',
    'hcefg': 'Int64',
    'csmcu': 'Int64',
    'flg_3dsmk': 'category'
}

# Assuming 'columns_to_drop' is a list of columns you want to drop
columns_to_drop = ['insfg','flam1']
print("load.py start")

# Drop the specified columns in place
pd.set_option('display.max_columns', None)

# Load dataset_1st/training.csv
df_train = pd.read_csv(sys.argv[1],dtype=convert_dict, na_values='NaN')
df_train.drop(columns=columns_to_drop, inplace=True)
df_train.fillna(-1, inplace=True)

# Load dataset_2nd/public.csv
df_public2 = pd.read_csv(sys.argv[2],dtype=convert_dict, na_values='NaN')
df_public2.drop(columns=columns_to_drop, inplace=True)
df_public2.fillna(-1, inplace=True)

# Load dataset_2nd/private_1_processed.csv
df_private = pd.read_csv(sys.argv[3],dtype=convert_dict, na_values='NaN')
df_private.drop(columns=columns_to_drop, inplace=True)
df_private.fillna(-1, inplace=True)


print("load.py done")