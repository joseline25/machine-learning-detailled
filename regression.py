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
"""