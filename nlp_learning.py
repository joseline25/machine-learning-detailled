import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('reviews.csv')

print(data.info())

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1185 entries, 0 to 1184
Data columns (total 6 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   Unnamed: 0     1185 non-null   int64
 1   User_name      1185 non-null   object
 2   Review title   1185 non-null   object
 3   Review Rating  1185 non-null   object
 4   Review date    1185 non-null   object
 5   Review_body    1185 non-null   object
dtypes: int64(1), object(5)
memory usage: 55.7+ KB

"""

print(data.columns)

"""
['Unnamed: 0', 'User_name', 'Review title', 'Review Rating',
       'Review date', 'Review_body']
"""

# rename the features

data = data.rename(columns={'Unnamed: 0': 'unknown',
                   'User_name': 'user_name', 'Review title': 'review_title',
                            'Review Rating': 'review_rating', 'Review date': 'review_date', 'Review_body': 'review_body'})

print(data.info())

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1185 entries, 0 to 1184
Data columns (total 6 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   unknown        1185 non-null   int64
 1   user_name      1185 non-null   object
 2   review_title   1185 non-null   object
 3   review_rating  1185 non-null   object
 4   review_date    1185 non-null   object
 5   review_body    1185 non-null   object
dtypes: int64(1), object(5)
memory usage: 55.7+ KB

"""

# Preprocessing
# Perform any necessary preprocessing steps on the text data in the "Review_body" column

# Split the data into features and target
X = data['review_body']
y = data['review_rating']
print(data)