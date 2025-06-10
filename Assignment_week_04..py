# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load Titanic dataset (use your own path if needed)
df = sns.load_dataset('titanic')

# Overview of the dataset
print("Dataset Info:\n", df.info())
print("\nDescriptive Statistics:\n", df.describe(include='all'))
print("\nMissing Values:\n", df.isnull().sum())

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Drop 'deck' due to excessive missing values
df = df.drop(columns=['deck'])

# Fill missing 'age' with median
df['age'].fillna(df['age'].median(), inplace=True)

# Fill missing 'embarked' with mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop any remaining rows with missing values
df.dropna(inplace=True)

# Univariate Analysis - Categorical
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(x='survived', data=df, ax=axes[0])
sns.countplot(x='pclass', data=df, ax=axes[1])
sns.countplot(x='sex', data=df, ax=axes[2])
axes[0].set_title("Survival Counts")
axes[1].set_title("Passenger Class Distribution")
axes[2].set_title("Gender Distribution")
plt.tight_layout()
plt.show()

# Univariate Analysis - Numerical
sns.histplot(df['age'], kde=True)
plt.title("Age Distribution")
plt.show()

sns.histplot(df['fare'], kde=True)
plt.title("Fare Distribution")
plt.show()

# Boxplot for Fare (outliers)
sns.boxplot(x=df['fare'])
plt.title("Fare Boxplot")
plt.show()

# Outlier detection using IQR
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['fare'] < Q1 - 1.5 * IQR) | (df['fare'] > Q3 + 1.5 * IQR)]
print(f"Number of fare outliers: {len(outliers)}")

# Bivariate Analysis
sns.barplot(x='sex', y='survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

sns.boxplot(x='survived', y='fare', data=df)
plt.title("Fare by Survival")
plt.show()

# KDE Plot: Age Distribution by Survival
sns.kdeplot(df[df['survived'] == 0]['age'], label='Did Not Survive', shade=True)
sns.kdeplot(df[df['survived'] == 1]['age'], label='Survived', shade=True)
plt.legend()
plt.title("Age Distribution by Survival")
plt.show()

# Heatmap of categorical survival relationships
ct = pd.crosstab(df['pclass'], df['survived'])
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
plt.title("Survival by Class")
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Feature Engineering
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['has_cabin'] = df['cabin'].notnull().astype(int)

# Final correlation with new features
corr = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, cmap='Spectral')
plt.title("Correlation with Engineered Features")
plt.show()
