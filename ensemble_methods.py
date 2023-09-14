from sklearn.feature_extraction import DictVectorizer
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


dict_train = data_train.to_dict(orient='records')
dict_val = data_val.to_dict(orient='records')

# Next, we’ll take care of X — the feature matrix by scalling (normalize)

# - We do not have missing values in the dataset and we do no have categorical variable
# to encode them, let's standardize our data by scalling them (centering and scaling)

# scaler = RobustScaler()

# X_train = scaler.fit_transform(data_train)
# X_val = scaler.transform(data_val)

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dict_train)
X_val = dv.transform(dict_val)


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

NOTE 
In some cases, importing XGBoost may give you a warning like YMLLoadWarning.
You shouldn't worry about it; the library will work without problems.
    """


import xgboost as xgb

# 1-1 Before we can train an XGBoost model, we need to wrap our data into DMatrix — a
# special data structure for finding splits efficiently.

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dv.feature_names_)

"""
    When creating an instance of DMatrix, we pass three parameters:
- X_train: the feature matrix
- y_train: the target variable
- feature_names: the names of features in X_train
    """

# Let’s do the same for the validation dataset:

dval = xgb.DMatrix(X_val, label=y_val, feature_names=dv.feature_names_)

# The next step is specifying the parameters for training. We’re using only a small subset
# of the default parameters of XGBoost

xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'nthread': 8,
    'seed': 1,
    'silent': 1
}

"""
    For us, the most important parameter now is objective: it specifies the learning task.
We're solving a binary classification problem (is the wine of good quality or not)
— that's why we need to choose binary:logistic. 
    """

# For training an XGBoost model, we use the train function. Let’s start with 10 trees

model = xgb.train(xgb_params, dtrain, num_boost_round=10)

"""
    We provide three arguments to train:
- xgb_params: the parameters for training
- dtrain: the dataset for training (an instance of DMatrix)
- num_boost_round=10: the number of trees to train


To evaluate the model, we need to make a prediction on
the validation dataset. For that, use the predict method with the validation data
wrapped in DMatrix:
    """

y_pred = model.predict(dval)

# Next, we calculate AUC using the same approach as previously:
print("The AUC is: \n", roc_auc_score(y_val, y_pred))
# 89% compare the result with our best random forest model

# Model performance monitoring

"""
    To get an idea of how AUC changes as the number of trees grows,
    we can use a watchlist, a built-in feature in XGBoost for monitoring model performance.
    
A watchlist is a Python list with tuples. Each tuple contains a DMatrix and its name.
This is how we typically do it:
"""

watchlist = [(dtrain, 'train'), (dval, 'val')]

# Additionally, we modify the list of parameters for training:
# we need to specify the metric we use for evaluation.
# In our case, it’s the AUC

xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    # Sets the evaluation metric to the AUC
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'silent': 1
}

"""
    To use the watchlist during training, we need to specify two extra arguments for the
train function:

- evals: the watchlist.
- verbose_eval: how often we print the metric. If we set it to “10,” we see the
result after each 10th step.
"""

model = xgb.train(xgb_params, dtrain, num_boost_round=100,
                  evals=watchlist, verbose_eval=10)

# While training, XGBoost prints the scores to the output

"""
Observations

As the number of trees grows, the score on the training set goes up (second column)

[0]     train-auc:0.94588       val-auc:0.84485
[10]    train-auc:0.99425       val-auc:0.89131
[20]    train-auc:0.99987       val-auc:0.90699
[30]    train-auc:1.00000       val-auc:0.90982
[40]    train-auc:1.00000       val-auc:0.91071
[50]    train-auc:1.00000       val-auc:0.91022
[60]    train-auc:1.00000       val-auc:0.90974
[70]    train-auc:1.00000       val-auc:0.90788
[80]    train-auc:1.00000       val-auc:0.90715
[90]    train-auc:1.00000       val-auc:0.90392
[99]    train-auc:1.00000       val-auc:0.90190

This behavior is expected: in boosting, every next model tries to fix the mistakes from
the previous step, so the score is always improving.


For the validation score, however, this is not the case. It goes up initially but then
starts to decrease. This is the effect of overfitting: our model becomes more and more
complex until it simply memorizes the entire training set. It"s not helpful for predicting 
the outcome for the customers outside of the training set, and the validation score
reflects that.
We get the best AUC on the 00th iteration (91%).

[40]    train-auc:1.00000       val-auc:0.91071


Next, we'll see how to get the best out of XGBoost by tuning its parameters



Parameter tuning for XGBoost
Previously, we used a subset of default parameters for training a model:

xgb_params = {
'eta': 0.3,
'max_depth': 6,
'min_child_weight': 1,

'objective': 'binary:logistic',
'eval_metric': 'auc',
'nthread': 8,
'seed': 1,
'silent': 1
}


We're mostly interested in the first three parameters. These parameters control the
training process:


- eta: Learning rate. Decision trees and random forest don't have this parameter.
In boosting, each tree tries to correct the mistakes from the previous iterations. 
Learning rate determines the weight of this correction. If we have a large value for eta,
the correction overweights the previous predictions significantly.
On the other hand, if the value is small, only a small fraction of this correction is used.
In practice it means
            - If eta is too large, the model starts to overfit quite early without realizing its full
                potential.
            - If it's too small, we need to train too many trees before it can produce good results.
                The default value of 0.3 is reasonably good for large datasets, 
                but for smaller datasets
                like ours, we should try smaller values like 0.1 or even 0.05
    Let's do it and see if it helps to improve the performance (0.1)
    
- max_depth: The maximum allowed depth of each tree; the same as max_depth
in DecisionTreeClassifier from Scikit-learn.

- min_child_weight: The minimal number of observations in each group; the
same as min_leaf_size in DecisionTreeClassifier from Scikit-learn.


Other parameters:
- objective: The type of task we want to solve. For classification, it should be
binary:logistic.
- eval_metric: The metric we use for evaluation. For this project, it's “AUC.”
- nthread: The number of threads we use for training the model. XGBoost is very
good at parallelizing training, so set it to the number of cores your computer has.
- seed: The seed for the random-number generator; we need to set it to make
sure the results are reproducible.
- silent: The verbosity of the output. When we set it to “1,” it outputs only warnings.

"""

xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'silent': 1
}

"""
Because now we can use a watchlist to monitor the performance of our model, we can
train for as many iterations as we want. Previously we used 100 iterations, but this may
be not enough for smaller eta. So let's use 500 rounds for training
"""
print("With eta == 0.1 and number of rounds == 500, we have:\n")
model = xgb.train(xgb_params, dtrain, num_boost_round=500,
                  verbose_eval=10, evals=watchlist)

"""

[0]     train-auc:0.94588       val-auc:0.84485
[10]    train-auc:0.97591       val-auc:0.86954
[20]    train-auc:0.98525       val-auc:0.87398
[30]    train-auc:0.99176       val-auc:0.88307
[40]    train-auc:0.99599       val-auc:0.88675
[50]    train-auc:0.99874       val-auc:0.89382
[60]    train-auc:0.99962       val-auc:0.89560
[70]    train-auc:0.99996       val-auc:0.89891
[80]    train-auc:1.00000       val-auc:0.89770
[90]    train-auc:1.00000       val-auc:0.89705
[100]   train-auc:1.00000       val-auc:0.89867
[110]   train-auc:1.00000       val-auc:0.89899
[120]   train-auc:1.00000       val-auc:0.89875
[130]   train-auc:1.00000       val-auc:0.89713
[140]   train-auc:1.00000       val-auc:0.89584
[150]   train-auc:1.00000       val-auc:0.89487
[160]   train-auc:1.00000       val-auc:0.89430
[170]   train-auc:1.00000       val-auc:0.89317
[180]   train-auc:1.00000       val-auc:0.89261
[190]   train-auc:1.00000       val-auc:0.89236
[200]   train-auc:1.00000       val-auc:0.89139
[210]   train-auc:1.00000       val-auc:0.89156
[220]   train-auc:1.00000       val-auc:0.89075
[230]   train-auc:1.00000       val-auc:0.88986
[240]   train-auc:1.00000       val-auc:0.88945
[250]   train-auc:1.00000       val-auc:0.88897
[260]   train-auc:1.00000       val-auc:0.88752
[270]   train-auc:1.00000       val-auc:0.88638
[280]   train-auc:1.00000       val-auc:0.88622
[290]   train-auc:1.00000       val-auc:0.88582
[300]   train-auc:1.00000       val-auc:0.88517
[310]   train-auc:1.00000       val-auc:0.88436
[320]   train-auc:1.00000       val-auc:0.88420
[330]   train-auc:1.00000       val-auc:0.88501
[340]   train-auc:1.00000       val-auc:0.88453
[350]   train-auc:1.00000       val-auc:0.88444
[360]   train-auc:1.00000       val-auc:0.88412
[370]   train-auc:1.00000       val-auc:0.88372
[380]   train-auc:1.00000       val-auc:0.88364
[390]   train-auc:1.00000       val-auc:0.88364
[400]   train-auc:1.00000       val-auc:0.88388
[410]   train-auc:1.00000       val-auc:0.88331
[420]   train-auc:1.00000       val-auc:0.88331
[430]   train-auc:1.00000       val-auc:0.88267
[440]   train-auc:1.00000       val-auc:0.88242
[450]   train-auc:1.00000       val-auc:0.88202
[460]   train-auc:1.00000       val-auc:0.88194
[470]   train-auc:1.00000       val-auc:0.88154
[480]   train-auc:1.00000       val-auc:0.88113
[490]   train-auc:1.00000       val-auc:0.88089
[499]   train-auc:1.00000       val-auc:0.88057

"""

# [110]   train-auc:1.00000       val-auc:0.89899

# The best validation score is 89.9%



"""
    Feature engineering is the process of creating new features out of existing ones.
For this project, we haven't created any features; we simply used the ones provided 
in the dataset. Adding more features should help improve the performance
of our model. For example, we can add the ratio of residual sugar to the
 quantity of alcool of the item. Experiment with engineering more features.
 
 We can do this with the dataset of ecoocaasitech: a pickup request or not?
 Also, we can predict the amount of requests per day by aggreating date.

"""