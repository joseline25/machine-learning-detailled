from sklearn.metrics import mutual_info_score, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from sklearn.model_selection import train_test_split

from sklearn.tree import export_text
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data visualization purposes
import seaborn as sns  # for statistical data visualization

import warnings

warnings.filterwarnings('ignore')

"""
Predict the income of adult 

"""

# load the dataset
data = pd.read_csv('adult.csv')

# get some general infos on the dataset
print(data.info())

"""
    6 numericals features and 9 categorical features
    
    48842 entries
    
    15 columns
"""

# get the statistics of the dataset (for numerical columns)

print(data.describe())

"""

Before doing anything else, let's lowercase all the column names and make
it consistent, replace - with _

"""

data.columns = data.columns.str.lower()
data.columns = data.columns.str.lower().str.replace('-', '_')

print(data.head())

# get the columns
print(data.columns)

"""
    ['age', 'workclass', 'fnlwgt', 'education', 'educational_num',
       'marital_status', 'occupation', 'relationship', 'race', 'gender',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
       'income']
       
    And we want to predict the income based on the other features. 
    We are going to get the values of the columns we want to predict.
    
"""

print(data['income'].values)  # '<=50K' or '>50K'

# convert to 0 or 1
data.income = (data.income == '>50K').astype(int)


""" 

    When we use data.income == '>50K', we create a Pandas series of type boolean. 
    A position in the series is equal to True if it's '>50K' in the original
    series and False otherwise. Because the only other value it can take is '<=50K' 
    this converts '>50K' to True and '<=50K' to False. 
"""

print(data['income'].values)  # 0 et 1
print(data.info())

# get all categorical columns
categorical = []
numerical = []
variables = ['age', 'workclass', 'fnlwgt', 'education', 'educational_num',
             'marital_status', 'occupation', 'relationship', 'race', 'gender',
             'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
for var in variables:
    if data[var].dtypes == 'int64' or data[var].dtypes == 'int32':
        numerical.append(var)
    else:
        categorical.append(var)

print("The categorical columns are ", categorical)
print("The numerical values are ", numerical)

print(data[categorical].nunique())

"""
We have 
    
workclass          9
education         16
marital_status     7
occupation        15
relationship       6
race               5
gender             2
native_country    42

From this, we cannot transform directly categorical columns such as native_country

Let's check for missing values
"""

# for numerical variables, check for null values


print(data.isnull().sum())  # no missing values

# check for negative values
negative_mask = data.drop(categorical, axis=1).lt(0)
negative_data = data.drop(categorical, axis=1)[negative_mask]
print("Negative values : ", negative_data.empty)  # no negative values


# let's manage Categorical variables

print("The proportion of adults with high income (>50K) is :")

global_mean = data.income.mean()

print("the income mean is :  ", round(global_mean, 3))  # 0.239

"""
Environ 23% de notre dataset a un high income. Donc 87% a un low income.

We can clearly see that:
the income rate in our data is 0.23, which is a strong indicator of class imbalance.


Now we come to another important part of exploratory data analysis: understanding which
features may be important for our model.

IMPORTANT: Knowing how other variables affect the target variable, income, is the key to
understanding the data and building a good model


This process is called feature importance

analysis, and it's often done as a part of exploratory data analysis to figure out which
variables will be useful for the model. It also gives us additional insights about the
dataset and helps answer questions like “What makes a adult have a high income?” and “What
are the characteristics of adults whith high income?”




We can look at all the distinct values of a variable. Then, for each variable, there's a
group of adults: all the adults who have this value. For each such group, we
can compute the income rate, which is the group income rate. When we have it, we can
compare it with the global income rate — the income rate calculated for all the observations
at once


If the difference between the rates is small, the value is not important when predicting
income because this group of adults is not really different from the rest of
the adults. On the other hand, if the difference is not small, something inside that
group sets it apart from the rest. A machine learning algorithm should be able to pick
this up and use it when making predictions.
    
"""

for variable in categorical:

    # Computes the AVG(income)

    # For that, we use the agg function to indicate
    # that we need to aggregate data into one value per group: the mean value.
    data_group = data.groupby(by=variable).income.agg(['mean'])
    # Calculates the difference between group income and global rate

    #  we create another column, diff, where we will keep the difference between the group mean
    # and the global mean.
    data_group['diff'] = data_group['mean'] - global_mean
    # Calculates the risk of having low income

    # we create the column risk, where we calculate
    # the fraction between the group mean and the global mean
    data_group['risk/possibility of having a high income'] = data_group['mean'] / global_mean
    print(data_group)
    
    
""" 
If the difference between the group rate and the global rate is small, 
the risk od having a high income is
close to 1: this group has the same level of risk as the rest of the adult'spopulation. 
adults in the group are as likely to have high income as anyone else. In other words, 
a group with a risk close to 1 is not risky at all.

If the risk is lower than 1, the group has lower risks: the hing income rate in this group
is smaller than the global income. For example, the value 0.5 means that the clients
in this group are two times less likely to churn than clients in general



we can measure the degree of
dependency between a categorical variable and the target variable. If two variables are
dependent, knowing the value of one variable gives us some information about
another. On the other hand, if a variable is completely independent of the target 
variable, it's not useful and can be safely removed from the dataset.

In our case, knowing that an adult has a Prof-specialty occupation or 
Married-civ-spouse marital status may or Doctorate/Prof-school/Masters education
or in Self-emp-inc/Federal-gov workclass or being in a married relationship indicate
that this adult is more likely to have a high income than not.
"""


"""
This is exactly the kind of
relationship we want to find in our data. Without such relationships in data,
machine learning models will not work — they will not be able to make predictions.
The higher the degree of dependency, the more useful a feature is.



For categorical variables, one such metric is MUTUAL INFORMATION, which tells how
much information we learn about one variable if we learn the value of the other variable. It’s a concept from information theory, and in machine learning, we often use it
to measure the mutual dependency between two variables.

Higher values of mutual information mean a higher degree of dependence

Mutual information is already implemented in Scikit-learn in the mutual_info_
score function from the metrics package, so we can just use it
"""


# Creates a stand-alone function for calculating mutual information


def calculate_mi(series):
    # Uses the mutual_info_score function from Scikit-learn
    return mutual_info_score(series, data.income)

print("The Mutual Importance of each categorical variable to income is \n")
# Applies the function from to each categorical column of the dataset
data_mi = data[categorical].apply(calculate_mi)
# Sorts the values of the result
data_mi = data_mi.sort_values(ascending=False).to_frame(name='MI')
print(data_mi)


# according to the mutual information table, relationship and marital_status
# variable has the more dependence with the target variable income


"""

CORRELATION COEFFICIENT

Mutual information is a way to quantify the degree of dependency between 
two categorical variables, but it doesn't work when one of the features is numerical,
so we cannot apply it to the three numerical variables that we have.
We can, however, measure the dependency between a binary target variable and a
numerical variable. 

We can pretend that the binary variable is numerical (containing
only the numbers zero and one) and then use the classical methods from statistics to
check for any dependency between these variables.

One such method is the correlation coefficient 
(sometimes referred as Pearson's correlation coefficient). It is a value from –1 to 1*
"""

print("the correlation of numerical features to 'income' is : \n")

print(
    data[numerical].corrwith(data.income))


"""
fnlwgt has a low negative correlation: as fnlwgt grows, income rate goes down.
educational_num has positive correlation: the more an adult is educated,
the more likely they are to have a high income.


After doing initial exploratory data analysis, identifying important features,
and getting some insights into the problem, we are ready to do the next step: 
feature engineering and model training.




4- We transform categorical variables into numeric variables so we can use them in
the model.


we need to perform the feature engineering step: 
transforming all categorical variables to numeric features. We'll do
that in the next section, and after that, we'll be ready to train the
Gaussian model


- One Hot Encoding

we cannot just take a categorical variable
and put it into a machine learning model. The models can deal only with numbers
in matrices. So, we need to convert our categorical data into a matrix form, or
encode.

One such encoding technique is one-hot encoding.
    """
    
# Split the data first

data_train_full, data_test = train_test_split(data, test_size=0.2, random_state=11)

data_train, data_val = train_test_split(data_train_full, test_size=0.33, random_state=11)

y_train = data_train.income.values
y_val = data_val.income.values

""" 
Deletes the income columns from both dataframes to
make sure we don't accidentally use the income variable
as a feature during training

"""
del data_train['income']
del data_val['income']


# Now Hot encoding categorical variables

# first method

train_dict = data_train[categorical + numerical].to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)


# We can learn the names of all these columns by using the get_feature_names method

print(dv.get_feature_names_out())

""" 
As we see, for each categorical feature it creates multiple columns for each of 
its distinct values. It keeps numerical features because
they are numerical; therefore, DictVectorizer doesn't change them.
Now our features are encoded as a matrix, so we can move to the next step: using a
model to predict income.


We have learned how to use Scikit-learn to perform one-hot encoding for categorical
variables, and now we can transform them into a set of numerical features and put
everything together into a matrix.
When we have a matrix, we are ready to do the model training part. In this section
we learn how to train the logistic regression model and interpret its results.

"""

model = GaussianNB()
model.fit(X_train, y_train)

val_dict = data_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)

y_test = data_test.income

print(y_train.reshape((-1,1)))
print(y_pred)
# Calculate accuracy
accuracy = accuracy_score(y_val.reshape((-1,1)), y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


