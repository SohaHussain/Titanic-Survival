# Titanic Survival

# IMPORTING LIBRARIES


# linear algebra

import numpy as np

# data processing

import pandas as pd

# data visualization

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# algorithms

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# getting data

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# data exploration/analysis

train_df.info()
# the training dataset has 891 examples and 11 features + 1 target variable (survived)
# 2 of the features are float 5 are integers and 5 are objects

train_df.describe()
# we can see that 38% out of the training-set survived the Titanic.
# We can also see that the passenger ages range from 0.4 to 80. On top of that we can already detect some features,
# that contain missing values, like the ‘Age’ feature.

train_df.head()

# we need to convert a lot of features into numeric ones later on, so that the machine learning algorithms can process
# them. Furthermore, we can see that the features have widely different ranges, that we will need to convert into
# roughly the same scale. We can also spot some more features, that contain missing values (NaN = not a number),
# that we need to deal with.

# let's take a more detailed look at what data are missing

total = train_df.isnull().sum().sort_values(ascending=False)
percent1 = train_df.isnull().sum()/train_df.isnull().count() * 100
percent = (round(percent1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])
missing_data.head()

# The Embarked feature has only 2 missing values, which can easily be filled. It will be much more tricky, to deal with
# #the ‘Age’ feature, which has 177 missing values. The ‘Cabin’ feature needs further investigation, but it looks like
# #that we might want to drop it from the dataset, since 77 % of it are missing.

train_df.columns.values

# it would make sense if everything except ‘PassengerId’, ‘Ticket’ and ‘Name’ would be correlated with a high
# #survival rate.

# 1. age and sex

survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
women = train_df[train_df['Sex'] == 'female']
men = train_df[train_df['Sex'] == 'male']
ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[0], kde=False)
ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[0], kde=False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False)
ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False)
ax.legend()
_ = ax.set_title('Male')

# 2. embarked , sex , pclass

FacetGrid = sns.FacetGrid(train_df, row='Embarked', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
# Embarked seems to be correlated with survival, depending on the gender.
# Women on port Q and on port S have a higher chance of survival. The inverse is true, if they are at port C.
# Men have a high survival probability if they are on port C, but a low probability if they are on port Q or S.
# Pclass also seems to be correlated with survival. We will generate another plot of it below.

# 3. Pclass

sns.barplot(x='Pclass', y='Survived', data=train_df)
# Here we see clearly, that Pclass is contributing to a persons chance of survival, especially if this person is in
# class 1. We will create another Pclass plot below.

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# The plot above confirms our assumption about pclass 1,
# but we can also spot a high probability that a person in pclass 3 will not survive.

# 4. SibSp and ParCh

# SibSp and Parch would make more sense as a combined feature, that shows the total number of relatives,
# a person has on the Titanic. I will create it below and also a feature that sows if someone is not alone.

data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_df['not_alone'].value_counts()

axes = sns.factorplot(x='relatives', y='Survived', data=train_df, aspect=2.5, )

# Here we can see that you had a high probabilty of survival with 1 to 3 realitves, but a lower one if you had less
# than 1 or more than 3 (except for some cases with 6 relatives).


# Data Processing

# First, I will drop ‘PassengerId’ from the train set, because it does not contribute to a persons survival
# probability. I will not drop it from the test set, since it is required there for the submission.

train_df = train_df.drop(['PassengerId'], axis=1)

# Missing Data:
# we have to deal with Cabin (687), Embarked (2) and Age (177).

# Cabin:
# A cabin number looks like ‘C123’ and the letter refers to the deck. Therefore we’re going to extract these and create
# a new feature, that contains a persons deck. Afterwords we will convert the feature into a numeric variable.
# The missing values will be converted to zero.

import re
deck={"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"U":8}
data=[train_df,test_df]
for dataset in data:
    dataset['Cabin']=dataset['Cabin'].fillna("U0")
    dataset['Deck']= dataset['Cabin'].map(lambda x: re.compile("([a-zA-z]+)").search(x).group())
    dataset['Deck']= dataset['Deck'].map(deck)
    dataset['Deck']= dataset['Deck'].fillna(0)
    dataset['Deck']=dataset['Deck'].astype(int)

#we can now drop the cabin feature

train_df=train_df.drop(['Cabin'],axis=1)
test_df=test_df.drop(['Cabin'],axis=1)

# age
# Now we can tackle the issue with the age features missing values. I will create an array that contains random numbers,
# which are computed based on the mean age value in regards to the standard deviation and is_null.

data = [train_df,test_df]
for dataset in data:
    mean=train_df['Age'].mean()
    std=test_df['Age'].std()
    is_null=dataset['Age'].isnull().sum()

    # compute random numbers between mean , std, is_null
    rand_age=np.random.randint(mean-std, mean+std, size=is_null)

    # filling NaN values in age column with random values generated
    age_slice= dataset['Age'].copy()
    age_slice[np.isnan(age_slice)]=rand_age
    dataset['Age']=age_slice
    dataset['Age']=train_df['Age'].astype(int)

train_df['Age'].isnull().sum()

# embarked
# since embarked feature has only 2 values missing, we will fill it with the common one

train_df['Embarked'].describe()
common_value='S'
data=[train_df,test_df]
for dataset in data:
    dataset['Embarked']=dataset['Embarked'].fillna(common_value)

# CONVERTING DATA

train_df.info()

# ‘Fare’ is a float and we have to deal with 4 categorical features: Name, Sex, Ticket and Embarked. Lets investigate
# and transform one after another.

# fare:
# converting fare from float to int using astype() in pandas

data=[train_df,test_df]

for dataset in data:
    dataset['Fare']=dataset['Fare'].fillna(0)
    dataset['Fare']=dataset['Fare'].astype(int)

# Name:
# We will use the Name feature to extract the Titles from the Name, so that we can build a new feature out of that.

data=[train_df,test_df]
titles={'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}

for dataset in data:
    # extract titles
    dataset['Title']=dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
    # replace titiles as more common titles or as rare
    dataset['Title']=dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major',
                                               'Rev','Sir','Jonkheer','Dona'],'Rare')
    dataset['Title']=dataset['Title'].replace('Mlle','Miss')
    dataset['Title']=dataset['Title'].replace('Ms','Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Miss')
    # convert titles into  numbers
    dataset['Title']=dataset['Title'].map(titles)
    # filling Nan with 0 to be safe
    dataset['Title']=dataset['Title'].fillna(0)

train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

# Sex:
# convert sex feature into numeric values

genders={'male':0,'female':1}
data=[train_df,test_df]

for dataset in data:
    dataset['Sex']=dataset['Sex'].map(genders)

# Ticket:
train_df['Ticket'].describe()

# since ticket feature has 681 different values it will be difficult to convert it , so we will drop it

train_df=train_df.drop(['Ticket'],axis=1)
test_df=test_df.drop(['Ticket'],axis=1)

# Embarked:

ports={'S':0,'C':1,'Q':2}
data=[train_df,test_df]

for dataset in data:
    dataset['Embarked']=dataset['Embarked'].map(ports)

# CREATING CATEGORIES

# Age:
# Now we need to convert the ‘age’ feature. First we will convert it from float into integer. Then we will create the
# new ‘AgeGroup” variable, by categorizing every age into a group.

data=[train_df,test_df]
for dataset in data:
    dataset['Age']=dataset['Age'].astype(int)
    dataset.loc[dataset['Age']<=11,'Age']=0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[dataset['Age'] > 66, 'Age'] = 7
train_df['Age'].value_counts()

# Fare:
# For the ‘Fare’ feature, we need to do the same as with the ‘Age’ feature. But it isn’t that easy, because if we cut
# the range of the fare values into a few equally big categories, 80% of the values would fall into the first category.
# Fortunately, we can use sklearn “qcut()” function, that we can use to see, how we can form the categories.

train_df.head(10)

data=[train_df,test_df]

for dataset in data:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare'] = 4
    dataset.loc[dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

# CREATING NEW FEATURES
# I will add two new features to the dataset, that I compute out of other features.

# 1. Age Time class

data=[train_df,test_df]
for dataset in data:
    dataset['Age_Class']=dataset['Age']*dataset['Pclass']

# 2. Fare per person

for dataset in data:
    dataset['Fare_per_person']=dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_per_person']=dataset['Fare_per_person'].astype(int)

# let's take a look at our training set before we start training models

train_df.head(10)

















