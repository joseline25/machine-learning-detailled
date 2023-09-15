from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
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
print("**************** check frequency distribution of values in workclass variable ****************")
print(df.Date.value_counts())


# 11-  check for cardinality in categorical variables
print("**************** check for cardinality in categorical variables ****************")
for var in categorical:

    print(var, ' contains ', len(df[var].unique()), ' labels')
    
    
# 12 -  find numerical variables

numerical = [var for var in df.columns if df[var].dtype != 'O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

# 13 -  view the numerical variables
print("**************** view the numerical variables ****************")
print(df[numerical].head())

# 14 - check missing values in numerical variables
print("**************** check missing values in numerical variables ****************")
print(df[numerical].isnull().sum())

#  Declare feature vector and target variable

X = df.drop(['Resident_Trash_Pickup_Request'], axis=1)

y = df['Resident_Trash_Pickup_Request']

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
    df2['Date'].fillna(X_train['Date'].mode()[0], inplace=True)
    df2['Weather'].fillna(X_train['Weather'].mode()[0], inplace=True)
    df2['Day_of_Week'].fillna(
        X_train['Day_of_Week'].mode()[0], inplace=True)



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

encoder = ce.OneHotEncoder(cols=['Date', 'Weather', 'Day_of_Week'])

#X_train = np.array(X_train)
print(X_train)

#X_train = encoder.fit_transform(X_train)

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

