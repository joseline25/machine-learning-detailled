from sklearn.metrics import mutual_info_score, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data visualization purposes
import seaborn as sns  # for statistical data visualization

from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings('ignore')


"""
Decision trees, the simplest tree-based model, are nothing but a sequence of 
if-then-else rules put together.
We can combine multiple decision trees into an ensemble to achieve better performance. 
We cover two tree-based ensemble models: random forest and gradient
boosting.




Project:

determine which physiochemical properties make a wine 'good' on the dataset 
winequality-red_decision_tree.csv
"""


# import the dataset for the assignement

data = pd.read_csv('winequality-red_decision_tree.csv')
# quality_values = {
#     1: 'one',
#     2: 'two',
#     3: 'three',
#     4: 'four',
#     5: 'five',
#     6: 'six',
#     7: 'seven',
#     8: 'eight',
#     9: 'nine',
#     10: 'ten'

# }
# data.quality = data.quality.map(quality_values)

# 1599 entries and 12 columns, all are of type float64 and the target, int64
print(data.info())


# import the data for the book

df = pd.read_csv('CreditScoring.csv')
print(df.info())  # 4455 entries and 14 columns , all are of types int64


"""

First, we can see that all the column names start with a capital letter in df.
Before doing anything else, let's lowercase all the column names and make
it consistent.

"""

df.columns = df.columns.str.lower()
data.columns = data.columns.str.lower()

# replace space with _
df.columns = df.columns.str.lower().str.replace(' ', '_')
data.columns = data.columns.str.lower().str.replace(' ', '_')
print(data.head(10))

# for a better observation, print the transpose

print(data.head().T)

# get the columns
print(data.columns)

"""
Our columns are: 

    ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'ph', 'sulphates', 'alcohol', 'quality']
       
and we want to predict quality which is an integer

We do not have categorical features in this dataset.  

To confirm that data indeed contains nonnumeric characters, we can now use the
isnull() function

"""

print(data.isnull().sum())
# We do not have any null values

# let's check for negatives values

# Check for negative values
negative_mask = data.lt(0)
negative_data = data[negative_mask]
print("Negative values : ", negative_data.empty)  # False
# there are no negative values in any columns


# let’s check the summary
# statistics for each of the columns: min, mean, max, and others.

print(data.describe())

"""
data.describe().round()

The mean of quality is  6.0
The std of quality is 1.0
The minimum quality is 3.0
The maximum quality is 8.0    
"""

# let transform the quality column
# if quality <= 5, then low quality (false) else good quality (true)

data.quality = data.quality.apply(lambda x: False if x <= 6 else True)

# now transform the true and false to 1 or 0

data.quality = data.quality.astype(int)

print(data)

# check for missing values

"""
Dataset preparation

- Split the dataset into train, validation, and test.
- Handle missing values.
- Use one-hot encoding to encode categorical variables.(we don't need it here)
- Create the feature matrix X and the target variable y.


Let's start by splitting the data. We will split the data into three parts:
- Training data (60% of the original dataset)
- Validation data (20%)
- Test data (20%)
    """

data_train_full, data_test = train_test_split(
    data, test_size=0.2, random_state=11)
data_train, data_val = train_test_split(data_train_full, test_size=0.25,
                                        random_state=11)

# check for lenght

print(len(data_train), len(data_val), len(data_test))

# The outcome we want to predict is quality
# We will use it to train a model, so it’s our y — the target variable

# We convert it to a categorical variable since its values range
# from 1 to 10 (3 to 8 in the dataset)


y_train = (data_train.quality).values
y_val = (data_val.quality).values

"""

    Now we need to remove quality from the DataFrames. If we don't do it,
    we may accidentally use this variable for training. For that, we use the 
    del operator
    
"""

del data_train['quality']
del data_val['quality']

# Next, we’ll take care of X — the feature matrix by scalling (normalize)

# - We do not have missing values in the dataset and we do no have categorical variable
# to encode them, let's standardize our data by scalling them (centering and scaling)

scaler = RobustScaler()

X_train = scaler.fit_transform(data_train)
X_val = scaler.transform(data_val)


"""
    Decision trees
A decision tree is a data structure that encodes a series of if-then-else rules. Each node 
in a tree contains a condition. If the condition is satisfied, we go to the right side of
the tree; otherwise, we go to the left. In the end we arrive at the final decision



Decision tree classifier

We'll use Scikit-learn for training a decision tree. Because we're solving
a classification problem, we need to use DecisionTreeClassifier from the 
tree package. 


Training the model is as simple as invoking the fit method:
"""

# Training

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_train)[:, 1]

"""

To check if the result is good, we need to evaluate the predictive performance of the
model on the validation set. Let's use AUC (area under the ROC curve) for that.


determine which physiochemical properties make a wine 'good' is a classification problem

For  binary classification problem, AUC is one of the best evaluation metrics. 
AUC shows how well a model separates positive examples from negative examples.

It has a nice interpretation: it describes the probability that a randomly chosen
positive example (“default”) has a higher score than a randomly chosen negative example
(“OK”) 
"""


print(roc_auc_score(y_train, y_pred))  # 100%


"""

When we execute it, we see that the score is 100% — the perfect score. Does it mean
that we can predict default without errors? Let's check the score on validation before
jumping to conclusions:


"""

y_pred = dt.predict_proba(X_val)[:, 1]
print(roc_auc_score(y_val, y_pred))  # 71%

# After running, we see that AUC on validation is only 74%.

# features importance

# Get feature importances
feature_importances = dt.feature_importances_

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame(
    {'Feature': data_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(" features importance \n")
print(importance_df)

print("The features importance are classified as follows:\n")

tree_text = export_text(dt, feature_names=scaler.feature_names_in_)
print(tree_text)

""" 
We just observed a case of overfitting. The tree learned the training data so well that
it simply memorized the outcome for each customer. However, when we applied it to
the validation set, the model failed. The rules it extracted from the data turned out to
be too specific to the training set.

In such cases, we say that the model cannot generalize.


Overfitting happens when we have a complex model with enough power to
remember all the training data. If we force the model to be simpler, we can make it
less powerful and improve the model's ability to generalize.


We have multiple ways to control the complexity of a tree. One option is to restrict
its size: we can specify the max_depth parameter, which controls the maximum number of 
levels. The more levels a tree has, the more complex rules it can learn



A tree with more levels can learn more complex rules. A tree with two levels
is less complex than a tree with three levels and, thus, less prone to overfitting.

We can try a smaller value and compare the results.
For example, we can change it to 2

"""
print(" Let's fix the depth of the tree to 2 and display the decision tree \n ")

dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, y_train)
y_pred = dt.predict_proba(X_train)[:, 1]
print(" the new roc AUC score is \n ", roc_auc_score(y_train, y_pred))


importance_df = pd.DataFrame(
    {'Feature': data_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("The new table of features importance \n")
print(importance_df)

# To visualize the tree we just learned, we can use the export_text function from the
# tree package:

print("The features importance are classified as follows:\n")

tree_text = export_text(dt, feature_names=scaler.feature_names_in_)
print(tree_text)

"""
Each line in the output corresponds to a node with a condition. If the condition is
true, we go inside and repeat the process until we arrive at the final decision. At the
end, if class is True, then the decision is True ( good quality) and otherwise it's 
False (not good quality)

"""


# Let’s check the score:
print("Let’s check the score:\n")
y_pred = dt.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print('train auc', auc)
y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('validation auc', auc)

"""
the value of train auc is 78%
It means that the model can no longer memorize all 
the outcomes from the training set.
However the score for the validation set is 70% and it was previously
71% 




However, this tree has another problem — it's too simple. To make it better, we
need to tune the model: try different parameters, and see which ones lead to the best
AUC. In addition to max_depth, we can control other parameters. 

To decide if we want to continue splitting the data, we use stopping criteria — criteria
that describe if we should add another split in the tree or stop.
The most common stopping criteria are
- The group is already pure.
- The tree reached the depth limit (controlled by the max_depth parameter).
- The group is too small to continue splitting (controlled by the min_samples_
leaf parameter).


By using these criteria to stop earlier, we force our model to be less complex and,
therefore, reduce the risk of overfitting.


Let's use this information to adjust the training algorithm:
- Find the best split:
- For each feature try all possible threshold values.
- Use the one with the lowest impurity.
- If the maximum allowed depth is reached, stop.
- If the group on the left is sufficiently large and it's not pure yet, repeat on
the left.
- If the group on the right is sufficiently large and it's not pure yet, repeat on
the right.


The process of finding the best set of parameters is called parameter tuning. 

We usually do it by changing the model and checking its score on the validation dataset.
In the end, we use the model with the best validation score.

As we have just learned, we can tune two parameters:
- max_depth
- min_leaf_size
These two are the most important ones, so we will adjust only them.
"""
maxx = 0
depth = 0
for depth in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    if (auc > maxx) and (depth != None):
        maxx = auc
        depthh = depth
    print('%4s -> %.3f' % (depth, auc))


print(
    f"The best auc score is with depth {depthh} with the AUC score of {maxx}")


"""
Next, we tune min_leaf_size. For that, we iterate over the three best parameters of
max_depth, and for each, go over different values of min_leaf_size:
    
    """

print("go over different values of min_leaf_size  ")

for m in [4, 5, 6]:
    print('depth: %s' % m)
    for s in [1, 5, 10, 15, 20, 50, 100, 200]:
        dt = DecisionTreeClassifier(max_depth=m, min_samples_leaf=s)
        dt.fit(X_train, y_train)
        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        print('%s -> %.3f' % (s, auc))
    print()


"""
From this diplay, we can detect the best parameters for our model.
    
    """
    
print("Decision Tree with max depth 6 and 15 leaft")
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)
y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('depth: %s and  %s leaves  -> %.3f' % (m, s, auc))

print("Features importance \n")

feature_importances = dt.feature_importances_
importance_df = pd.DataFrame(
    {'Feature': data_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

print("The corresponding tree is :\n")

tree_text = export_text(dt, feature_names=scaler.feature_names_in_)
print(tree_text)

