import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

"""
Creating a car-price prediction project with a linear regression model


The problem we solve here is predicting the price of a car. Suppose that we
have a website where people can sell and buy used cars. When posting an ad on our
website, sellers often struggle to come up with a meaningful price. We want to help
our users with automatic price recommendations.


We ask the sellers to specify the
model, make, year, mileage, and other important characteristics of a car, and based on
that information, we want to suggest the best price.

The plan for the project is the following:
1 First, we download the dataset. (data.csv)
2 Next, we do some preliminary analysis of the data.
3 After that, we set up a validation strategy to make sure our model produces correct predictions.
4 Then we implement a linear regression model in Python and NumPy.
5 Next, we cover feature engineering to extract important features from the data
to improve the model.
6 Finally, we see how to make our model stable with regularization and use it to
predict car prices.

data.csv
"""


"""
2- Exploratory data analysis


Understanding data is an important step in the machine learning process. Before we
can train any model, we need to know what kind of data we have and whether it is useful. We do this with exploratory data analysis (EDA).
We look at the dataset to learn
- The distribution of the target variable
- The features in this dataset
- The distribution of values in these features
- The quality of the data
- The number of missing values
"""

# Reading and preparing data

data = pd.read_csv('data.csv')
# - observe the first five rows

print(data.head())

# Lowercases all the column names, and replaces spaces with underscores

data.columns = data.columns.str.lower().str.replace(' ', '_')

# Selects only columns with string values

string_columns = list(data.dtypes[data.dtypes == 'object'].index)

# Lowercases and replaces spaces with underscores for values in
# all string columns of the DataFrame

for col in string_columns:
    data[col] = data[col].str.lower().str.replace(' ', '_')

print(data.head())

"""
For us, the most interesting column here is the last one: 
MSRP (manufacturer's suggested retail price, or simply the price of a car)

it's our target variable, the
y, which is the value that we want to learn to predict

We will use this column for predicting the prices of a car.
"""

"""
2-1  Obsevations

One of the first steps of exploratory data analysis should always be to look at what
the values of y (the target) look like. We typically do this by checking the distribution of y: a visual
description of what the possible values of y can be and how often they occur. This type
of visualization is called a histogram.
"""

# sns.histplot(data.msrp, bins=40)

plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='msrp', bins=10)
plt.title(f'distribution of msrp')
plt.xlabel('msrp')
plt.show()

"""
There are many cars with low prices on the left side, but the number
quickly drops, and there's a long tail of very few cars with high prices


The long tail makes it quite difficult for us to see the distribution, but it has an even
stronger effect on a model: such distribution can greatly confuse the model, so it
won't learn well enough. One way to solve this problem is log transformation. If we
apply the log function to the prices, it removes the undesired effect

y_new = log(y + 1)


The +1 part is important in cases that have zeros. The logarithm of zero is minus infinity,
but the logarithm of one is zero. If our values are all non-negative, by adding 1, we
make sure that the transformed values do not go below zero.
For our specific case, zero values are not an issue — all the prices we have start at
$1,000 — but it's still a convention that we follow. NumPy has a function that performs
this transformation:
"""

data['log_price'] = np.log1p(data.msrp)

# The effect of the long tail is removed, and we
# can see the entire distribution in one plot

# sns.histplot(log_price)
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='log_price', bins=10)
plt.title(f'distribution of msrp')
plt.xlabel('log_price')
plt.show()

""" 

As we see, this transformation removes the long tail, and now the distribution
resembles a bell-shaped curve. This distribution is not normal, of course,
because of the large peak in lower prices, but the model can deal with 
it more easily

NOTE 

Generally, it's good when the target distribution looks like the normal
distribution. Under this condition, models such as linear regression
perform well.

"""

"""
2.2 Checking for missing values

This step is important because, typically,
machine learning models cannot deal with missing values automatically.
We need to know whether we need to do anything special to handle those values.

"""
print(data.isnull().sum())


"""
make                    0
model                   0
year                    0
engine_fuel_type        3
engine_hp              69
engine_cylinders       30
transmission_type       0
driven_wheels           0
number_of_doors         6
market_category      3742
vehicle_size            0
vehicle_style           0
highway_mpg             0
city_mpg                0
popularity              0
msrp                    0
log_price               0
dtype: int64



The first thing we see is that MSRP — our target variable — doesn't have any missing
values. This result is good, because otherwise, such records won't be useful to us: we
always need to know the target value of an observation to use it for training the model.

Also, a few columns have missing values, especially market_category, in which we have
almost 4,000 rows with missing values.

We need to deal with missing values later when we train the model, so we should
keep this problem in mind. For now, we don't do anything else with these features and
proceed to the next step: setting up the validation framework so that we can train and
test machine learning models.


2.3 Validation framework

 it's important to set up the validation framework as early as
possible to make sure that the models we train are good and can generalize — that
is, that the model can be applied to new, unseen data. To do that, we put aside some
data and train the model only on one part. Then we use the held-out dataset — the
one we didn't use for training — to make sure that the predictions of the model
make sense.



If we have a small training dataset in which all BMW
cars cost only $10,000, for example, the model will think that this is true for all BMW
cars in the world.
To ensure that this doesn't happen, we use validation. Because the validation dataset
is not used for training the model, the optimization method did not see this data.
When we apply the model to this data, it emulates the case of applying the model to
new data that we've never seen. If the validation dataset has BMW cars with prices
higher than $10,000, but our model will predict $10,000 on them, we will notice that
the model doesn't perform well on these examples.

Let's split the DataFrame such that
- 20% of data goes to validation.
- 20% goes to test.
- The remaining 60% goes to train.
"""

# Gets the number of rows in the DataFrame
n = len(data)
# Calculates how many rows should go to train, validation, and test
n_val = int(0.2 * n)  # 20% of data
n_test = int(0.2 * n)  # 20% of data
n_train = n - (n_val + n_test)  # le reste

# Fixes the random seed to make sure that the results are reproducible
np.random.seed(2)

"""
The function np.random.seed takes in any number and uses this number
as the starting seed for all the generated data inside NumPy's random package.

This is good for reproducibility. If we want somebody else to run this code and get the
same results, we need to make sure that everything is fixed, even the “random” component 
of our code.

NOTE 
This makes the results reproducible on the same computer. With a different operating
system and a different version of NumPy, the result may be different.
"""

# Creates a NumPy array with indices from 0 to (n–1), and shuffles it
idx = np.arange(n)
np.random.shuffle(idx)


# Uses the array with indices to get a shuffled DataFrame
data_shuffled = data.iloc[idx]

"""
After we create an array with indices idx, we can use it to get a shuffled 
version of our initial DataFrame. For that purpose, we use iloc, which 
is a way to access the rows of the DataFrame by their numbers

On ait la même chose avec train_test_split de sklearn.model_selection
"""

# Splits the shuffled DataFrame into train, validation, and test
data_train = data_shuffled.iloc[:n_train].copy()
data_val = data_shuffled.iloc[n_train:n_train+n_val].copy()
data_test = data_shuffled.iloc[n_train+n_val:].copy()

"""
Our initial analysis
showed a long tail in the distribution of prices, and to remove its effect,
we need to apply the log transformation. We can do that for each DataFrame separately
"""

y_train = np.log1p(data_train.msrp.values)
y_val = np.log1p(data_val.msrp.values)
y_test = np.log1p(data_test.msrp.values)
print(y_train)

#To avoid accidentally using the target variable later, let’s remove it from the dataframes:
del data_train['msrp']
del data_val['msrp']
del data_test['msrp']

"""
NOTE 

Removing the target variable is an optional step. But it's helpful to make
sure that we don't use it when training a model: if that happens, we'd use price
for predicting the price, and our model would have perfect accuracy.


2.4 Machine learning for regression

The problem
we are solving is a regression problem: the goal is to predict a number — the price of
a car. For this project we will use the simplest regression model: linear regression.

"""
