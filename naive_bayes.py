from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data visualization purposes
import seaborn as sns  # for statistical data visualization

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('adult.csv')

# EDA
print("**************************   EDA   *************************")

# 1- view dimensions of dataset
print("**************** les dimensions du dataset ****************")
print(df.shape)

# 2- preview the dataset
print("**************** preview of the dataset ****************")
print(df.head())

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names
print("**************** columns of the dataset ****************")
print(df.columns)

# 3- let's again preview the dataset

print(df.head())

# 4- view summary of dataset
print("**************** summary of the dataset ****************")
print(df.info())

# 5- find categorical variables

categorical = [var for var in df.columns if df[var].dtype == 'O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

# 6-  view the categorical variables

print(df[categorical].head())

# 7-  check missing values in categorical variables
print("**************** check null values in the dataset ****************")
print(df[categorical].isnull().sum())


# 8- view frequency counts of values in categorical variables
print("**************** view frequency counts of values in categorical variables ****************")
for var in categorical:

    print(df[var].value_counts())


# 9- view frequency distribution of categorical variables
print("**************** view frequency distribution of categorical variables ****************")
for var in categorical:

    print(df[var].value_counts()/float(len(df)))


"""
    Now, we can see that there are several variables like workclass, occupation and 
    native_country which contain missing values. Generally, the missing values are 
    coded as NaN and python will detect them with the usual command of df.isnull().sum().

    But, in this case the missing values are coded as ?. Python fail to detect these 
    as missing values because it do not consider ? as missing values. 
    So, I have to replace ? with NaN so that Python can detect these missing values.

I will explore these variables and replace ? with NaN.

"""

# 10 -  check labels in workclass variable
print("**************** check labels in workclass variable ****************")
print(df.workclass.unique())


# 11-  check frequency distribution of values in workclass variable
print("**************** check frequency distribution of values in workclass variable ****************")
print(df.workclass.value_counts())


# 12- replace '?' values in workclass variable with `NaN`

print("**************** replace '?' values in workclass variable with NaN ****************")
df['workclass'].replace('?', np.NaN, inplace=True)

print(df)

# 13-  again check the frequency distribution of values in workclass variable
print("**************** again check the frequency distribution of values in workclass variable ****************")
print(df.workclass.value_counts())


# 14-  check labels in occupation variable
print("**************** check labels in occupation variable ****************")
print(df.occupation.unique())


# 15- check frequency distribution of values in occupation variable
print("**************** check frequency distribution of values in occupation variable ****************")
print(df.occupation.value_counts())

# 16- replace '?' values in occupation variable with `NaN`

df['occupation'].replace('?', np.NaN, inplace=True)

# 17-  again check the frequency distribution of values in occupation variable
print("**************** again check the frequency distribution of values in occupation variable ****************")
print(df.occupation.value_counts())


# 18-  check labels in native_country variable
print("**************** check labels in native_country variable ****************")
print(df.native_country.unique())


# 19-  check frequency distribution of values in native_country variable
print("**************** check frequency distribution of values in native_country variable ****************")
print(df.native_country.value_counts())


# 20 -  replace '?' values in native_country variable with `NaN`

df['native_country'].replace('?', np.NaN, inplace=True)


# 21 - again check the frequency distribution of values in native_country variable

df.native_country.value_counts()

# 22- check missing values in categorical variables again
print("**************** check missing values in categorical variables again ****************")
print(df[categorical].isnull().sum())

# 23-  check for cardinality in categorical variables
print("**************** check for cardinality in categorical variables ****************")
for var in categorical:

    print(var, ' contains ', len(df[var].unique()), ' labels')


# 24 -  find numerical variables

numerical = [var for var in df.columns if df[var].dtype != 'O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# 25 -  view the numerical variables
print("**************** view the numerical variables ****************")
print(df[numerical].head())


# 26 - check missing values in numerical variables
print("**************** check missing values in numerical variables ****************")
print(df[numerical].isnull().sum())


#  Declare feature vector and target variable

X = df.drop(['income'], axis=1)

y = df['income']


# Split data into separate training and test set

# split X and y into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


# check the shape of X_train and X_test
print("**************** check the shape of X_train and X_test ****************")
print(X_train.shape, X_test.shape)

#  Feature Engineering

# 1- check data types in X_train
print("**************** check data types in X_train ****************")
print(X_train.dtypes)

# 2- display categorical variables
print("**************** display categorical variables ****************")
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

print(categorical)

# 3- display numerical variables
print("**************** display numerical variables ****************")
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

print(numerical)


# 4 -  print percentage of missing values in the categorical variables in training set
print("**************** percentage of missing values in the categorical variables in training set ****************")
print(X_train[categorical].isnull().mean())


# 5-  print categorical variables with missing data
print("**************** print categorical variables with missing data ****************")
for col in categorical:
    if X_train[col].isnull().mean() > 0:
        print(col, (X_train[col].isnull().mean()))


# 6-  impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native_country'].fillna(
        X_train['native_country'].mode()[0], inplace=True)

# 7-  check missing values in categorical variables in X_train
print("**************** check missing values in categorical variables in X_train ****************")
X_train[categorical].isnull().sum()


# 8-  check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()

# 9- check missing values in X_train
print("**************** check missing values in X_train ****************")
print(X_train.isnull().sum())

# 10- check missing values in X_test
print("**************** check missing values in X_test ****************")
print(X_test.isnull().sum())

# print categorical variables
print("**************** print categorical variables ****************")
print(categorical)

print(X_train[categorical].head())

# import category encoders

print("  **************** Encode categorical features **************** ")
# 11-  encode remaining variables with one-hot encoding

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship',
                                 'race', 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

print(X_train.head())

print(X_train.shape)

print(X_test.head())

print(X_test.shape)

#  Feature Scaling
print("****************  Feature scaling  ****************")
cols = X_train.columns


scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
print(X_train.head())

# Model training
print("**************** Model training ****************")
# train a Gaussian Naive Bayes classifier on the training set


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)

# Predict the results

y_pred = gnb.predict(X_test)
print("**************** predictions ****************")
print(y_pred)

# Check accuracy score

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Compare the train-set and test-set accuracy
print("****************Compare the train-set and test-set accuracy****************")
y_pred_train = gnb.predict(X_train)

print(y_pred_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


#Check for overfitting and underfitting
print("****************Check for overfitting and underfitting****************")
# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

"""
The training-set accuracy score is 0.8067 while the test-set accuracy to be 0.8083. 
These two values are quite comparable. So, there is no sign of overfitting.
"""

#################################"abs

"""
Compare model accuracy with null accuracy
So, the model accuracy is 0.8083. But, we cannot say that our model 
is very good based on the above accuracy. We must compare it with 
the null accuracy. Null accuracy is the accuracy that could be achieved 
by always predicting the most frequent class.

So, we should first check the class distribution in the test set.
"""

# check class distribution in test set
print("*************check class distribution in test set****************")
print(y_test.value_counts())