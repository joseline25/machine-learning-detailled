from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mutual_info_score, accuracy_score
import category_encoders as ce
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data visualization purposes
import seaborn as sns  # for statistical data visualization
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')

# load the dataset
df = pd.read_csv('individual_trash_pickup_dataset.csv')

# EDA
print("**************************   EDA   *************************")

# 1- view dimensions of dataset
print("**************** les dimensions du dataset ****************")
print(df.shape)  # (1000000, 8)

# 2- preview the dataset
print("**************** preview of the dataset ****************")
print(df.head())
# Date  Resident_ID  Temperature Weather Day_of_Week  Previous_Requests  Public_Holiday  Resident_Trash_Pickup_Request

print("**************** columns of the dataset ****************")
print(df.columns)

# 4- view summary of dataset
print("**************** summary of the dataset ****************")
print(df.info())


neg_vals = 0
for val in df.Resident_ID:
    if val < 0:
        neg_vals += 1
        val = 0  # je remplace par 0

print(
    f"the number of negative values is {(neg_vals *100) /1000000}% of the toal number of values. ")
print("********************We drop all the rows with negative Resident_ID***************")
# df[df.select_dtypes(include=[np.number]).ge(0).all(1)] # ça ne marche pas on va gérer ça après


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

# convert Date values from datetime to date

df['Date'] = pd.to_datetime(df['Date']).dt.date

#  view again  frequency distribution of categorical variables
print("**************** view frequency distribution of categorical variables ****************")
for var in categorical:

    print(df[var].value_counts()/float(len(df)))


# 10 -  check labels in Date, Weather, Day_of_Week variable
print("**************** check labels in Date variable ****************")
print(df.Date.unique())
print(df.Weather.unique())
print(df.Day_of_Week.unique())

# 11-  check frequency distribution of values in Date variable
print("**************** check frequency distribution of values in workclass variable ****************\n")
print(df.Date.value_counts())

# aggregate with all features
aggregated_data = df.groupby("Date").agg({"Resident_ID": "first", "Weather": "first", "Temperature": "first",
                                          "Day_of_Week": "first", "Previous_Requests": "sum",
                                          "Public_Holiday": "first", "Resident_Trash_Pickup_Request": "sum"}).reset_index()


print(aggregated_data)

print("\n*****************The proportion of request is :*****************\n")

global_mean = df.Resident_ID.mean()
print(global_mean)  # 48.492014% Our dataset is not imbalanced

"""
Now we come to another important part of exploratory data analysis: understanding which
features may be important for our model.

IMPORTANT: Knowing how other variables affect the target variable, income, is the key to
understanding the data and building a good model


This process is called feature importance

analysis, and it's often done as a part of exploratory data analysis to figure out which
variables will be useful for the model. It also gives us additional insights about the
dataset and helps answer questions like “What are the contitions for a pickup request?” 
and “What are the characteristics of a pickup request?”

"""

# get all categorical columns
categorical = []
numerical = []
variables = ['Date', 'Resident_ID', 'Temperature', 'Weather', 'Day_of_Week',
             'Previous_Requests',  'Public_Holiday',  'Resident_Trash_Pickup_Request']
for var in variables:
    if df[var].dtypes == 'int64' or df[var].dtypes == 'int32':
        numerical.append(var)
    else:
        categorical.append(var)

print("The categorical columns are \n", categorical)


print(df[categorical].nunique())

"""
We have 
Date           41667
Weather            2
Day_of_Week        7
dtype: int64


Let's get the the risk for request depending on one of these features
"""

for variable in categorical:

    # Computes the AVG(income)

    # For that, we use the agg function to indicate
    # that we need to aggregate data into one value per group: the mean value.
    data_group = df.groupby(
        by=variable).Resident_Trash_Pickup_Request.agg(['mean'])
    # Calculates the difference between group Resident_Trash_Pickup_Request and global rate

    #  we create another column, diff, where we will keep the difference between the group mean
    # and the global mean.
    data_group['diff'] = data_group['mean'] - global_mean
    # Calculates the risk of not having a request

    # we create the column risk, where we calculate
    # the fraction between the group mean and the global mean
    data_group['risk/possibility of having a request'] = data_group['mean'] / global_mean
    print(data_group)


"""
Let's calculate the MUtual Importance

"""


def calculate_mi(series):
    # Uses the mutual_info_score function from Scikit-learn
    return mutual_info_score(series, df.Resident_Trash_Pickup_Request)


print("The Mutual Importance of each categorical variable to a request in a day is \n")
# Applies the function from to each categorical column of the dataset
data_mi = df[categorical].apply(calculate_mi)
# Sorts the values of the result
data_mi = data_mi.sort_values(ascending=False).to_frame(name='MI')
print(data_mi)

"""
The Mutual Importance of each categorical variable to income is

                       MI
Date         2.133185e-02
Day_of_Week  1.805312e-06
Weather      2.938864e-07
"""


def calculate_mi_agg(series):
    # Uses the mutual_info_score function from Scikit-learn
    return mutual_info_score(series, aggregated_data.Resident_Trash_Pickup_Request)


print("The Mutual Importance of each categorical variable to toal request per dady is \n")
# Applies the function from to each categorical column of the dataset
data_mi = aggregated_data[categorical].apply(calculate_mi_agg)
# Sorts the values of the result
data_mi = data_mi.sort_values(ascending=False).to_frame(name='MI')
print(data_mi)

"""
The Mutual Importance of each categorical variable to income is

                   MI
Date         2.315354
Day_of_Week  0.001608
Weather      0.000219

We can observe that Date is highly correlated to the amount of request per day
"""

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
    aggregated_data[numerical].corrwith(aggregated_data.Resident_Trash_Pickup_Request))

"""

Resident_ID                     -0.008140
Temperature                      0.007208
Previous_Requests               -0.003524
Public_Holiday                   0.000434
Resident_Trash_Pickup_Request    1.000000
dtype: float64

We observe that no numerical value is correlated to the amount of requests per day

"""

print(
    df[numerical].corrwith(df.Resident_Trash_Pickup_Request))

"""
If we work with the original datset without aggregation, we get

Resident_ID                     -0.000623
Temperature                     -0.000334
Previous_Requests                0.000020
Public_Holiday                  -0.001566
Resident_Trash_Pickup_Request    1.000000
dtype: float64



Normalization: scalling 
"""

# Split the data first

data_train_full, data_test = train_test_split(aggregated_data, test_size=0.2, random_state=11)

data_train, data_val = train_test_split(data_train_full, test_size=0.33, random_state=11)

y_train = data_train.Resident_Trash_Pickup_Request.values
y_val = data_val.Resident_Trash_Pickup_Request.values

""" 
Deletes the Resident_Trash_Pickup_Request columns from both dataframes to
make sure we don't accidentally use the Resident_Trash_Pickup_Request variable
as a feature during training

"""
del data_train['Resident_Trash_Pickup_Request']
del data_val['Resident_Trash_Pickup_Request']

# Now Hot encoding categorical variables

# first method

train_dict = data_train[categorical + numerical].to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)



# # 11-  check for cardinality in categorical variables
# print("**************** check for cardinality in categorical variables ****************")
# for var in categorical:

#     print(var, ' contains ', len(df[var].unique()), ' labels')


# # 12 -  find numerical variables

# numerical = [var for var in df.columns if df[var].dtype != 'O']

# print('There are {} numerical variables\n'.format(len(numerical)))

# print('The numerical variables are :', numerical)

# # 13 -  view the numerical variables
# print("**************** view the numerical variables ****************")
# print(df[numerical].head())

# # 14 - check missing values in numerical variables
# print("**************** check missing values in numerical variables ****************")
# print(df[numerical].isnull().sum())

# #  Declare feature vector and target variable

# X = df.drop(['Resident_Trash_Pickup_Request'], axis=1)

# y = df['Resident_Trash_Pickup_Request']

# # Split data into separate training and test set

# # split X and y into training and testing sets


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=0)


# # check the shape of X_train and X_test
# print("**************** check the shape of X_train and X_test ****************")
# print(X_train.shape, X_test.shape)

# #  Feature Engineering

# # 1- check data types in X_train
# print("**************** check data types in X_train ****************")
# print(X_train.dtypes)

# # 2- display categorical variables
# print("**************** display categorical variables ****************")
# categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

# print(categorical)

# # 3- display numerical variables
# print("**************** display numerical variables ****************")
# numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

# print(numerical)

# # 4 -  print percentage of missing values in the categorical variables in training set
# print("**************** percentage of missing values in the categorical variables in training set ****************")
# print(X_train[categorical].isnull().mean())


# # 5-  print categorical variables with missing data
# print("**************** print categorical variables with missing data ****************")
# for col in categorical:
#     if X_train[col].isnull().mean() > 0:
#         print(col, (X_train[col].isnull().mean()))


# # 6-  impute missing categorical variables with most frequent value

# for df2 in [X_train, X_test]:
#     df2['Date'].fillna(X_train['Date'].mode()[0], inplace=True)
#     df2['Weather'].fillna(X_train['Weather'].mode()[0], inplace=True)
#     df2['Day_of_Week'].fillna(
#         X_train['Day_of_Week'].mode()[0], inplace=True)


# # 7-  check missing values in categorical variables in X_train
# print("**************** check missing values in categorical variables in X_train ****************")
# X_train[categorical].isnull().sum()


# # 8-  check missing values in categorical variables in X_test

# X_test[categorical].isnull().sum()

# # 9- check missing values in X_train
# print("**************** check missing values in X_train ****************")
# print(X_train.isnull().sum())

# # 10- check missing values in X_test
# print("**************** check missing values in X_test ****************")
# print(X_test.isnull().sum())


# # print categorical variables
# print("**************** print categorical variables ****************")
# print(categorical)

# print(X_train[categorical].head())

# # import category encoders

# print("  **************** Encode categorical features **************** ")
# # 11-  encode remaining variables with one-hot encoding

# encoder = ce.OneHotEncoder(cols=['Date', 'Weather', 'Day_of_Week'])

# #X_train = np.array(X_train)
# print(X_train)

# #X_train = encoder.fit_transform(X_train)

# X_test = encoder.transform(X_test)

# print(X_train.head())

# print(X_train.shape)

# print(X_test.head())

# print(X_test.shape)

# #  Feature Scaling
# print("****************  Feature scaling  ****************")
# cols = X_train.columns


# scaler = RobustScaler()

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)

# X_train = pd.DataFrame(X_train, columns=[cols])
# X_test = pd.DataFrame(X_test, columns=[cols])
# print(X_train.head())
