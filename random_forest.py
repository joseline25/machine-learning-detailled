from sklearn.metrics import mutual_info_score, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data visualization purposes
import seaborn as sns  # for statistical data visualization

from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')


"""
    One model individually may be wrong,
but if we combine the output of multiple models into one, the chance of an incorrect
answer is smaller. This concept is called ensemble learning, and a combination of models
is called an ensemble.

(Un model peut se tromper mais si plusieurs/la majorité s'acccordent sur un résultat alors
il a plus de chance d'être juste.)


IMPORTANT : 

For this to work, the models need to be different. If we train the same decision tree
model 10 times, they will all predict the same output, so it's not useful at all.

The easiest way to have different models is to train each tree on a different subset of
features. For example, suppose we have three features: assets, debts, and price. We
can train three models:
-  The first will use assets and debts.
-  The second will use debts and price.
-  The last one will use assets and price.

With this approach, we'll have different trees, each making its own decisions

Donc on a pour un même dataset pour une classification, differents arbres qui 
s'entraînnent sur des ensembles de features differents (avec des features communes
souvent)


But when we put their predictions together, their mistakes average out, and
combined, they have more predictive power.
This way of putting together multiple decision trees into an ensemble is called a
random forest. To train a random forest, we can do this


-  Train N independent decision tree models.
-  For each model, select a random subset of features, and use only them for
training.
-  When predicting, combine the output of N models into one.


NOTE 

This is a very simplified version of the algorithm. It's enough to illustrate
the main idea, but in reality, it's more complex.
Scikit-learn contains an implementation of a random forest, 
so we can use it for solving our problem. Let's do it.

from sklearn.ensemble import RandomForestClassifier
"""

# get the data from decision_tree.py before training the model


"""
Project:

determine which physiochemical properties make a wine 'good' on the dataset 
winequality-red_decision_tree.csv
"""


# import the dataset for the assignement

data = pd.read_csv('winequality-red_decision_tree.csv')


"""

First, we can see that all the column names start with a capital letter in df.
Before doing anything else, let's lowercase all the column names and make
it consistent.

"""


data.columns = data.columns.str.lower()

# replace space with _

data.columns = data.columns.str.lower().str.replace(' ', '_')


# for a better observation, print the transpose

print(data.head().T)


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

# print(len(data_train), len(data_val), len(data_test))

# The outcome we want to predict is quality
# We will use it to train a model, so it’s our y — the target variable


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


# 1 - Specify the number of tree of the ensemble

# We do it with the n_estimators parameter

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)

# 2 - After training finishes, we can evaluate the performance of the result

y_pred = rf.predict_proba(X_val)[:, 1]
print(roc_auc_score(y_val, y_pred))  # 87% (it changes everytimes I run)

# Every time we retrain the model, the score changes: it varies from 80% to 88%


"""
    The number of trees in the ensemble is an important parameter, and it influences
the performance of the model. Usually, a model with more trees is better than a model
with fewer trees. On the other hand, adding too many trees is not always helpful.


To see how many trees we need, we can iterate over different values for n_estimators
and see its effect on AUC:
"""

# list of aucs
aucs = []

# Trains progressively more trees in each iteration
for i in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=i, random_state=3)
    rf.fit(X_train, y_train)

    # Evaluates the score
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print('%s -> %.3f' % (i, auc))

    # Adds the score to the list with other scores
    aucs.append(auc)


"""
    In this code, we try different numbers of trees: from 10 to 200, going by steps of 10
(10, 20, 30, …). Each time we train a model, we calculate its AUC on the validation set
and record it.
After we finish, we can plot the results:
"""


plt.plot(range(10, 201, 10), aucs)
plt.show()


"""
    
    The performance grows rapidly for the first 10–25 trees; then the growth slows down 
    a little bit between 25-35 then grow rapidly between 35-60.
    The peek is with 60 trees
After 150, adding more trees is not helpful anymore: the performance stays approximately at
the level of 89%.
The number of trees is not the only parameter we can change to get better performance. Next, we see which other parameters we should also tune to improve the
model.



Parameter tuning for random forest

A random forest ensemble consists of multiple decision trees, so the most important
parameters we need to tune for random forest are the same:
- max_depth
- min_leaf_size
We can change other parameters, but we won't cover them


Let's test a few values for max_depth and see how AUC evolves as the number of
trees grows:


    """

# Creates a dictionary with AUC results
all_aucs = {}

# Iterates over different depth values
for depth in [4, 5, 6, 10, 15, 20]:  # for this model in decision tree the good depth was 6
    print('depth: %s' % depth)
    # Creates a list with AUC results for the current depth level
    aucs = []
    # Iterates over different n_estimator values
    for i in range(10, 201, 10):
        rf = RandomForestClassifier(
            n_estimators=i, max_depth=depth, random_state=1)
        # Evaluates the model
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        print('%s -> %.3f' % (i, auc))
        aucs.append(auc)

    # Save the AUCs for the current depth level in the dictionary
    all_aucs[depth] = aucs
    print()

# Now for each value of max_depth, we have a series of AUC scores. We can plot them now:

num_trees = list(range(10, 201, 10))
plt.plot(num_trees, all_aucs[5], label='depth=5')
# plt.title('AUCs for forest of trees of depth = 5')
# plt.show()
plt.plot(num_trees, all_aucs[10], label='depth=10')
# plt.title('AUCs for forest of trees of depth = 10')
# plt.show()
plt.plot(num_trees, all_aucs[5], label='depth=15')
# plt.title('AUCs for forest of trees of depth = 15')
# plt.show()
plt.plot(num_trees, all_aucs[20], label='depth=20')
# plt.title('AUCs for forest of trees of depth = 20')
plt.show()

"""
With max_depth=10, AUC goes over 91% (30 trees), whereas for other values it performs worse.

Now let's tune min_samples_leaf. We set the value for the max_depth parameter
from the previous step and then follow the same approach as previously for 
determining the best value for min_samples_leaf:
"""

all_aucs = {}
for m in [3, 5, 10]:
    print('min_samples_leaf: %s' % m)
    aucs = []
    for i in range(10, 201, 20):
        rf = RandomForestClassifier(
            n_estimators=i, max_depth=15, min_samples_leaf=m, random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        print('%s -> %.3f' % (i, auc))
        aucs.append(auc)

    all_aucs[m] = aucs
    print()

# Let’s plot it:

num_trees = list(range(10, 201, 20))
plt.plot(num_trees, all_aucs[3], label='min_samples_leaf=3')
plt.plot(num_trees, all_aucs[5], label='min_samples_leaf=5')
plt.plot(num_trees, all_aucs[10], label='min_samples_leaf=10')
plt.show()

# from the observations, choose the right parameters for the model and run it again

# n_estimators = 30 ( 30 trees)
# max_depth = 20
# min_samples_leaf = 3

rf = RandomForestClassifier(
    n_estimators=30, max_depth=20, min_samples_leaf=3, random_state=1)
rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]
print(roc_auc_score(y_val, y_pred))  # 88%



"""
Ensemble methods : Random forests, Gradient boosting

Random forest is not the only way to combine multiple decision trees. 
There's a different approach: gradient boosting. We cover that next.




Gradient boosting
In a random forest, each tree is independent: it's trained on a different set of features.
After individual trees are trained, we combine all their decisions together to get the
final decision.

It's not the only way to combine multiple models together in one ensemble, however.
Alternatively, we can train models sequentially — each next model tries to fix
errors from the previous one:
- Train the first model.
- Look at the errors it makes.
- Train another model that fixes these errors.
- Look at the errors again; repeat sequentially.


This way of combining models is called boosting. Gradient boosting is a particular 
variation of this approach that works especially well with trees

In gradient boosting, we train the models sequentially, and each next
tree fixes the errors of the previous one.

implementations of the gradient boosting model:
GradientBoostingClassifier from Scikit-learn, XGBoost, LightGBM and CatBoost. 

1 - XGBoost : Extreme Gradient Boosting (most popular implementation)

XGBoost doesn't come with scikit-learn, so to use it, we need to install it. 
The easiest way is to install it with pip:

pip install xgboost

Then import it with 

import xgboost as xgb
    """
