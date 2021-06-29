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











