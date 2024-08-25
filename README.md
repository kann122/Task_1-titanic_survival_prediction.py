
"""
Original file is located at
    https://colab.research.google.com/drive/1iN-CscwwUKkVgZ7He2B_okx_TOw-D427
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

"""Data Collection/Loading and Processing."""

titanic_data = pd.read_csv('/content/Titanic-Dataset.csv')

titanic_data.head()

titanic_data.shape

titanic_data.info()

titanic_data.isnull().sum()

#remove missing/null values
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

#replace missing values with mean number
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)

titanic_data.info()

titanic_data.isnull().sum()

#letts fix embarked
print(titanic_data['Embarked'].mode())

print(titanic_data['Embarked'].mode()[0])

#replace the mode value with the missing value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)

titanic_data.isnull().sum()

"""Analysing the data"""

titanic_data.describe()

#how many survived?
titanic_data['Survived'].value_counts()

#visualizing data
sns.set()
sns.countplot(x=titanic_data['Survived'])

titanic_data['Sex'].value_counts()

#let's visualize the count of survivals wrt Gender
sns.countplot(x='Sex', hue='Survived', data=titanic_data)

#let's visualize the count of survivals wrt Pclass
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)

# Look at survival rate by sex
titanic_data.groupby('Sex')[['Survived']].mean()

titanic_data['Sex'].unique()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

titanic_data['Sex']=labelencoder.fit_transform(titanic_data['Sex'])
titanic_data.head()

titanic_data['Sex'], titanic_data['Survived']

sns.countplot(x='Sex', hue='Survived', data=titanic_data)

titanic_data.isnull().sum()

df_final= titanic_data
df_final.head()

"""MODEL TRAINING"""

X=titanic_data[['Pclass','Sex']]
Y=titanic_data['Survived']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

log = LogisticRegression(random_state=0)
log.fit(X_train,Y_train)

"""MODEL PREDICTION"""

pred = print(log.predict(X_test))

print(Y_test)

import warnings
warnings.filterwarnings("ignore")

res = log.predict([[2,0]])
if(res==0):
  print("So Sorry! Not Survived")
else:
  print("Survived")
