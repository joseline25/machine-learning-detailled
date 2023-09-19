from sklearn.svm import SVC, SVR
# we will only use SVC (Support Vector Classifier)
# SVR is Support Vector Regression
import pandas as pd
from sklearn.datasets import load_iris, load_digits
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('spam.csv', encoding='latin-1')

"""
Since the initial encoding generate an error, we can try latin-1, 
cp1252 or ISO-8859-1

"""
# let's get the features of our dataset
print(data.columns)
# ['v1', 'v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
print(data.info())

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   v1          5572 non-null   object
 1   v2          5572 non-null   object
 2   Unnamed: 2  50 non-null     object
 3   Unnamed: 3  12 non-null     object
 4   Unnamed: 4  6 non-null      object
dtypes: object(5)
memory usage: 217.8+ KB

We have 5572 entries and 5 columns. The columns Unnamed: 2, Unnamed: 3 and
Unnamed: 4 have much more NAN values than non-null values. we can drop them 
maybe. let's see what are these non-null values?



The IterativeImputer class in scikit-learn provides a way to estimate missing
values by modeling each feature using the other features in an iterative manner. It fits a regression model to predict the missing values based on the available data.

Here's an example of how you can use IterativeImputer to impute missing values 
in a dataset:

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Assuming your dataset is stored in a DataFrame called 'data' 
# with missing values

imputer = IterativeImputer()
imputed_data = imputer.fit_transform(data)
"""

print(data['Unnamed: 4'].unique())  # these are weird emails.

# I will retains the columns but I will replace the NaN with something else.
mode_value = data['Unnamed: 2'].mode()[0]
data['Unnamed: 2'] = data['Unnamed: 2'].fillna(mode_value)
print(data['Unnamed: 2'])
print(data['Unnamed: 2'].unique())

mode_value = data['Unnamed: 3'].mode()[0]
data['Unnamed: 3'] = data['Unnamed: 3'].fillna(mode_value)


mode_value = data['Unnamed: 4'].mode()[0]
data['Unnamed: 4'] = data['Unnamed: 4'].fillna(mode_value)


# let's rename our columns

data = data.rename(columns={'Unnamed: 2': 'unknown_two',
                   'Unnamed: 3': 'unknown_three', 'Unnamed: 4': 'unknown_four',
                            'v1': 'target', 'v2': 'email'})
print(data.info())

"""
 <class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 5 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   target         5572 non-null   object
 1   email          5572 non-null   object
 2   unknown_two    5572 non-null   object
 3   unknown_three  5572 non-null   object
 4   unknown_four   5572 non-null   object
dtypes: object(5)
memory usage: 217.8+ KB

 """

print(data.columns)

# ['target', 'email', 'unknown_two', 'unknown_three', 'unknown_four']

print(data.head())

#*********Data Preprocessing**********#

# Split the data into features (X) and target (y)
X = data['email']
y = data['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# data_train_full, data_test = train_test_split(
#     data, test_size=0.2, random_state=11)
# data_train, data_val = train_test_split(data_train_full, test_size=0.25,
#                                         random_state=11)

#*********Feature Extraction**********#

"""
Convert text to numerical representation: Transform the text emails into
numerical features using techniques such as TF-IDF (Term Frequency-Inverse
Document Frequency) or CountVectorizer.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_features = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_features = vectorizer.transform(X_test)


#*********Model Training and Evaluation**********#

"""
Initialize and train the SVM model using the training data.
Evaluate the model's performance using appropriate metrics such
as accuracy, precision, recall, and F1-score.

"""

from sklearn.metrics import classification_report

# Initialize the SVM model
svm_model = SVC()

# Train the model
svm_model.fit(X_train_features, y_train)

# Predict on the testing data
y_pred = svm_model.predict(X_test_features)

# Evaluate the model
report = classification_report(y_test, y_pred)
print(report)


"""
Precision: Precision measures the proportion of correctly predicted positive
instances (true positives) out of all instances predicted as positive 
(true positives + false positives). In other words, it shows the accuracy of
the model in labeling examples as a specific class.

Recall: Recall, also known as sensitivity or true positive rate, measures
the proportion of correctly predicted positive instances (true positives) 
out of all actual positive instances (true positives + false negatives).
It indicates the ability of the model to identify positive examples.

F1-score: The F1-score is the harmonic mean of precision and recall. It provides
a balanced measure between precision and recall, combining both metrics into a
single value.

Support: Support represents the number of occurrences of each class in the true
labels.


my results for this are:

            precision    recall  f1-score   support

         ham       0.98      1.00      0.99       965
        spam       1.00      0.87      0.93       150

    accuracy                           0.98      1115
   macro avg       0.99      0.93      0.96      1115
weighted avg       0.98      0.98      0.98      1115


For the 'spam' class:

Precision: 1.00 indicates that 100% of the predicted 'spam' instances are truly
'spam'.

Recall: 0.87 suggests that the model correctly identifies 87% of the actual 
'spam' instances.

F1-score: 0.93 is a balanced measure that considers both precision and recall.

Support: 150 represents the number of instances labeled as 'spam' in the 
true labels.


For the 'ham' class:

Precision: 0.98 indicates that 98% of the predicted 'ham' instances are
truly 'ham'.

Recall: 1.00 suggests that the model correctly identifies 100% of the 
actual 'ham' instances.

F1-score: 0.99 is the balanced measure considering both precision and recall.

Support: 965 represents the number of instances labeled as 'ham' in 
the true labels.



The "micro avg," "macro avg," and "weighted avg" rows provide aggregated
metrics across all classes. Micro-average calculates metrics globally by 
considering the total true positives, false negatives, and false positives.
Macro-average calculates metrics for each class independently and then takes
the average. Weighted average calculates metrics for each class and weighs
them by their support (the number of occurrences).

Interpreting these metrics helps assess the performance of the classification
model in terms of precision, recall, and F1-score for each class. It's
important to consider the context and specific requirements of your problem
when evaluating these metrics.

"""


#*********Model Tuning**********#

"""
We can perform hyperparameter tuning to find the best parameters
for the SVM model. This can be done using techniques such as GridSearchCV
or RandomizedSearchCV.
"""

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'C': [1, 10, 100],
              'kernel': ['linear', 'rbf']}

# Perform grid search
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train_features, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(best_params)
# {'C': 10, 'kernel': 'linear'}


#**********Check Overfitting*********#

"""
To check if the Support Vector Machine (SVM) model is overfitting,
we can evaluate its performance on both the training set and the 
separate test set. Here are a few steps to follow:

Split the data: Split your dataset into a training set and a test set.
The training set will be used to train the model, while the test set 
will serve as an independent dataset to evaluate its performance.

"""

from sklearn.metrics import classification_report

y_train_pred = svm_model.predict(X_train_features)
train_report = classification_report(y_train, y_train_pred)
print("Training Set Report:")
print(train_report)

y_test_pred = svm_model.predict(X_test_features)
test_report = classification_report(y_test, y_test_pred)
print("Test Set Report:")
print(test_report)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy:", accuracy)

"""
The accuracy represents the performance metric that measures the proportion
of correctly classified instances (or samples) by the model out of the total
number of instances in the dataset. In the context of classification models
like Support Vector Machines (SVM) or K-Nearest Neighbors (KNN), accuracy
provides an overall evaluation of how well the model predicts the correct
class labels for the given data.

More specifically, accuracy is calculated as the ratio of the number of correct
predictions to the total number of predictions made by the model. It is 
expressed as a value between 0 and 1, where 1 represents 100% accuracy 
(all predictions are correct), and 0 represents 0% accuracy (no predictions 
are correct).
"""





