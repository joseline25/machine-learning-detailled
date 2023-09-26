# predict usage patterns (customers.csv)

"""

The nearest neighbors method (k-Nearest Neighbors, or k-NN)
is another very popular classification method that is also sometimes
used in regression problems. This, like decision trees, is one of the 
most comprehensible approaches to classification. The underlying 
intuition is that you look like your neighbors. More formally, 
the method follows the compactness hypothesis: if the distance 
between the examples is measured well enough, then similar 
examples are much more likely to belong to the same class.





About dataset

Imagine a telecommunications provider has segmented its customer base
by service usage patterns, categorizing the customers into four groups.
If demographic data can be used to predict group membership, the company
can customize offers for individual prospective customers. It is a 
classification problem. That is, given the dataset, with predefined labels,
we need to build a model to be used to predict class of a new or unknown case.
The example focuses on using demographic data, such as region, age, 
and marital, to predict usage patterns. The target field, called custcat,
has four possible values that correspond to the four customer groups,
as follows: 

1- Basic Service 
2- E-Service 
3- Plus Service
4- Total Service 

Our objective is to build a classifier, to predict the class of unknown cases.
We will use a specific type of classification called K nearest neighbour.
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('customers.csv')

print(data.info())

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 12 columns):
 #   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   region   1000 non-null   int64
 1   tenure   1000 non-null   int64
 2   age      1000 non-null   int64
 3   marital  1000 non-null   int64
 4   address  1000 non-null   int64
 5   income   1000 non-null   float64
 6   ed       1000 non-null   int64
 7   employ   1000 non-null   int64
 8   retire   1000 non-null   float64
 9   gender   1000 non-null   int64
 10  reside   1000 non-null   int64
 11  custcat  1000 non-null   int64
dtypes: float64(2), int64(10)
memory usage: 93.9 KB

"""

# Split the data into features (X) and target (y)
X = data.drop('custcat', axis=1)
y = data['custcat']


# *********Feature Scaling*********#

"""
Since KNN is sensitive to the scale of the features, it's beneficial
to normalize or standardize the data. Here, we'll use StandardScaler
from scikit-learn to standardize the numerical features.

"""

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# *********Split the Data*********#

"""
Split the dataset into training and testing sets to assess
the model's performance.
"""

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)


# *********Train the KNN Model*********#

"""
Initialize and train the KNN model using the training set.
"""

# Initialize the KNN model
knn_model = KNeighborsClassifier(n_neighbors=377) # au départ =5

# Train the model
knn_model.fit(X_train, y_train)

# *********Evaluate the Model*********#

"""

Predict the target variable for the test set and evaluate the model's 
performance.

"""

# Predict on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred)
print(report)

# *********Model Tuning (Optional)*********#

"""

You can tune the hyperparameters of the KNN model, such as the number 
of neighbors (n_neighbors), to achieve better performance. This can be 
done using techniques like GridSearchCV or RandomizedSearchCV.

"""
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 15,17,19, 23,33 ]} # best result is 15

# Perform grid search
grid_search = GridSearchCV(knn_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(best_params)

from sklearn.metrics import accuracy_score
# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

"""
With 'n_neighbors' =  7
             precision    recall  f1-score   support

           1       0.35      0.47      0.40        60
           2       0.24      0.21      0.22        39
           3       0.40      0.42      0.41        55
           4       0.23      0.15      0.18        46

    accuracy                           0.33       200
   macro avg       0.31      0.31      0.30       200
weighted avg       0.32      0.33      0.32       200

{'n_neighbors': 23}
Accuracy: 0.33




With 'n_neighbors' =  15
              precision    recall  f1-score   support

           1       0.40      0.43      0.42        60
           2       0.23      0.21      0.22        39
           3       0.42      0.51      0.46        55
           4       0.30      0.22      0.25        46

    accuracy                           0.36       200
   macro avg       0.34      0.34      0.34       200
weighted avg       0.35      0.36      0.35       200

{'n_neighbors': 15}
Accuracy: 0.36



With n_neighbors = 23

             precision    recall  f1-score   support

           1       0.37      0.42      0.39        60
           2       0.15      0.10      0.12        39
           3       0.39      0.53      0.45        55
           4       0.29      0.20      0.23        46

    accuracy                           0.34       200
   macro avg       0.30      0.31      0.30       200
weighted avg       0.31      0.34      0.32       200

{'n_neighbors': 23}
Accuracy: 0.335


With 'n_neighbors' = 100
              precision    recall  f1-score   support

           1       0.40      0.57      0.47        60
           2       0.19      0.08      0.11        39
           3       0.39      0.47      0.43        55
           4       0.27      0.20      0.23        46

    accuracy                           0.36       200
   macro avg       0.31      0.33      0.31       200
weighted avg       0.33      0.36      0.33       200

{'n_neighbors': 23}
Accuracy: 0.36


With 'n_neighbors': 500
              precision    recall  f1-score   support

           1       0.42      0.57      0.48        60
           2       0.67      0.05      0.10        39
           3       0.42      0.71      0.53        55
           4       0.26      0.13      0.17        46

    accuracy                           0.41       200
   macro avg       0.44      0.36      0.32       200
weighted avg       0.43      0.41      0.35       200


Accuracy: 0.405



When n_neighbors = 1000, we have the error

Expected n_neighbors <= n_samples,  but n_samples = 800, n_neighbors = 1000

donc n_neighbors ne doit pas excéder la taille du dataset (ici le train_test)

de plus lorsque n_neighbors = 800, on a une precision = 0



With 'n_neighbors' = 377

         precision    recall  f1-score   support

           1       0.44      0.55      0.49        60
           2       0.57      0.10      0.17        39
           3       0.42      0.67      0.51        55
           4       0.31      0.20      0.24        46

    accuracy                           0.41       200
   macro avg       0.43      0.38      0.35       200
weighted avg       0.43      0.41      0.38       200


Accuracy: 0.415
"""