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
df_train = pd.read_csv(sys.argv[1],dtype=convert_dict)
df_train.drop(columns=columns_to_drop, inplace=True)

#df_train = df_train[(df_train['locdt'] >= 40)] # Filter the DataFrame to get rows where 'locdt' > certain number
df_train.fillna(-1, inplace=True)


df_public = pd.read_csv(sys.argv[2],dtype=convert_dict, na_values='NaN')
df_public.drop(columns=columns_to_drop, inplace=True)

#df_train = df_train.drop(["label"], axis=1)
df_public.fillna(-1, inplace=True)

print("load.py done")