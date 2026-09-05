import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Initial exploration
print(df.head())
print("Shape:", df.shape)
print(df.columns)
print(df.dtypes)
df.info()
print(df.describe())

# Missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# Handle Age missing values
df['Age'] = df['Age'].fillna(df['Age'].median())

# Handle Embarked missing values
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Create Cabin availability indicator
df['CabinAvailable'] = df['Cabin'].notnull().astype(int)

# Remove original Cabin column
df.drop('Cabin', axis=1, inplace=True)

# Check duplicates
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicates if present
df = df.drop_duplicates()

# Check categorical values
print(df['Sex'].unique())
print(df['Embarked'].unique())
print(df['Pclass'].unique())

# Outlier visualization - Age
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Age'])
plt.title('Box Plot of Age')
plt.show()

# Outlier visualization - Fare
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fare'])
plt.title('Box Plot of Fare')
plt.show()

# IQR method for Fare
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['Fare'] < lower_bound) |
    (df['Fare'] > upper_bound)
]

print("Number of potential Fare outliers:", len(outliers))

# Remove unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Encode Sex
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Final checks
print("\nFinal dataset:")
print(df.head())

print("\nFinal shape:", df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

df.info()