import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load the dataset
df = pd.read_csv("C:\\Users\\Lenovo\\OneDrive\\Desktop\\train.csv")

# Basic info
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# Filling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Drop columns with too many missing values (optional)
df.drop(['Cabin'], axis=1, inplace=True)

# Survival count
print("\nSurvival Count:\n", df['Survived'].value_counts())

# Gender distribution vs survival
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.legend(labels=['Not Survived', 'Survived'])
plt.show()

# Passenger class vs survival
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.legend(labels=['Not Survived', 'Survived'])
plt.show()

# Embarked vs survival
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title("Survival by Port of Embarkation")
plt.legend(labels=['Not Survived', 'Survived'])
plt.show()

# Age distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Age vs survival boxplot
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()

# Fare distribution
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
