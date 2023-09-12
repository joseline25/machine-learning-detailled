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

df = pd.read_csv('car_evaluation.csv')

print(df)

print(df.shape)

# preview the dataset

print(df.head())


"""
    Rename column names
    We can see that the dataset does not have proper column names.
    The columns are merely labelled as 0,1,2.... and so on. 
    We should give proper names to the columns. I will do it as follows
"""

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


df.columns = col_names

print(df.columns)

# let's again preview the dataset

print(df.head())

# summary of dataset

print(df.info())

"""
    Frequency distribution of values in variables
Now, we will check the frequency counts of categorical variables
"""


for col in col_names:
    
    print(df[col].value_counts())
    
# Explore "class" variable

print(df['class'].value_counts())