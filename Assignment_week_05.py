import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Save SalePrice and drop it from train for preprocessing
train_labels = train['SalePrice']
train.drop(['SalePrice'], axis=1, inplace=True)

# Combine train and test for consistent preprocessing
combined = pd.concat([train, test], sort=False).reset_index(drop=True)

# Drop columns with too many missing values
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']
combined.drop(columns=drop_cols, inplace=True)

# Fill missing values
for col in combined.columns:
    if combined[col].dtype == 'object':
        combined[col] = combined[col].fillna(combined[col].mode()[0])
    else:
        combined[col] = combined[col].fillna(combined[col].median())

# Feature Engineering
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']
combined['TotalBath'] = (combined['FullBath'] + 0.5 * combined['HalfBath'] +
                         combined['BsmtFullBath'] + 0.5 * combined['BsmtHalfBath'])
combined['HouseAge'] = combined['YrSold'] - combined['YearBuilt']
combined['SinceRemodel'] = combined['YrSold'] - combined['YearRemodAdd']
combined['IsRemodeled'] = (combined['YearBuilt'] != combined['YearRemodAdd']).astype(int)
combined['HasGarage'] = combined['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

# Label Encoding for ordinal variables
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
                'GarageQual', 'GarageCond']
label_enc = LabelEncoder()
for col in ordinal_cols:
    if col in combined.columns:
        combined[col] = label_enc.fit_transform(combined[col])

# One-hot encode remaining categorical variables
combined = pd.get_dummies(combined, drop_first=True)

# Standardize numeric features
numeric_feats = combined.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
combined[numeric_feats] = scaler.fit_transform(combined[numeric_feats])

# Split back to train and test
processed_train = combined[:len(train)]
processed_test = combined[len(train):]

# Add back target to train set
processed_train['SalePrice'] = train_labels.values

# Save processed files
processed_train.to_csv('processed_train.csv', index=False)
processed_test.to_csv('processed_test.csv', index=False)

print("✅ Preprocessing complete. Files saved as 'processed_train.csv' and 'processed_test.csv'.")
