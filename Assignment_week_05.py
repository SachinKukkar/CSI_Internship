import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# 📥 Load dataset from OpenML
print("Loading dataset...")
data = fetch_openml(name='house_prices', as_frame=True)
df = data.frame.copy()

# 🎯 Separate features and target
y = df['SalePrice']
X = df.drop('SalePrice', axis=1)

# ❌ Drop columns with too many missing values
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']
X.drop(columns=[col for col in drop_cols if col in X.columns], inplace=True)

# 🧼 Fill missing values
for col in X.columns:
    if X[col].dtype == 'object':
        X[col].fillna(X[col].mode()[0], inplace=True)
    else:
        X[col].fillna(X[col].median(), inplace=True)

# ⚙️ Feature Engineering
X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
X['TotalBath'] = (X['FullBath'] + 0.5 * X['HalfBath'] +
                  X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath'])
X['HouseAge'] = X['YrSold'] - X['YearBuilt']
X['SinceRemodel'] = X['YrSold'] - X['YearRemodAdd']
X['IsRemodeled'] = (X['YearBuilt'] != X['YearRemodAdd']).astype(int)
X['HasGarage'] = X['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

# 🔠 Label Encode ordinal categorical variables
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']

le = LabelEncoder()
for col in ordinal_cols:
    if col in X.columns:
        X[col] = le.fit_transform(X[col])

# 🧠 One-hot encode other categorical variables
X = pd.get_dummies(X, drop_first=True)

# 📏 Standardize numeric features
numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ✂️ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🤖 Train Models
print("Training models...")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 📊 Evaluation
lr_rmse = sqrt(mean_squared_error(y_test, lr_preds))
rf_rmse = sqrt(mean_squared_error(y_test, rf_preds))

print(f"\n✅ Model Evaluation Results:")
print(f"Linear Regression RMSE: {lr_rmse:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")
