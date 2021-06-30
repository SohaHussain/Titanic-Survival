## Titanic Survival

#IMPORTING LIBRARIES


# linear algebra

import numpy as np

# data processing

import pandas as pd

#data visualization

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

#algorithms

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

#getting data

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

#data exploration/analysis

train_df.info()
#the training dataset has 891 examples and 11 features + 1 target variable (survived)
#2 of the features are float 5 are integers and 5 are objects

train_df.describe()
# we can see that 38% out of the training-set survived the Titanic.
# We can also see that the passenger ages range from 0.4 to 80. On top of that we can already detect some features,
# that contain missing values, like the ‘Age’ feature.

train_df.head()

# we need to convert a lot of features into numeric ones later on, so that the machine learning algorithms can process
# them. Furthermore, we can see that the features have widely different ranges, that we will need to convert into
# roughly the same scale. We can also spot some more features, that contain missing values (NaN = not a number),
# that we need to deal with.

#let's take a more detailed look at what data are missing

total = train_df.isnull().sum().sort_values(ascending=False)
percent1 = train_df.isnull().sum()/train_df.isnull().count() * 100
percent = (round(percent1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])
missing_data.head()

#The Embarked feature has only 2 missing values, which can easily be filled. It will be much more tricky, to deal with
# #the ‘Age’ feature, which has 177 missing values. The ‘Cabin’ feature needs further investigation, but it looks like
# #that we might want to drop it from the dataset, since 77 % of it are missing.

train_df.columns.values















