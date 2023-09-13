from sklearn.metrics import mutual_info_score
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


"""
We will learn in this python script the techniques to manage
a classifier.

- Performing exploratory data analysis for
identifying important features
- Encoding categorical variables to use them
in machine learning models
- Using logistic regression for classification


We are going to use machine learning to predict churn.
Churn is when customers stop using the services of a company. Thus, churn prediction 
is about identifying customers who are likely to cancel their contracts soon.
If the company can do that, it can offer discounts on these services in an effort to
keep the users.



Naturally, we can use machine learning for that: we can use past data about customers 
who churned and, based on that, create a model for identifying present customers who 
are about to leave. This is a binary classification problem. The target
variable that we want to predict is categorical and has only two possible outcomes:
churn or not churn.


The classification models includes logistic regression, 
decision trees, and neural networks.

We will start with the simplest one: logistic regression

Even though it's indeed the simplest,
it's still 
- powerful and has many advantages over other models:
- it's fast and 
- easy to understand, and 
- its results are easy to interpret. 
- It's a workhorse of machine learning and 
- the most widely used model in the industry.



Imagine that we are working at a telecom company that offers phone and internet
services, and we have a problem: some of our customers are churning. They no longer
are using our services and are going to a different provider. We would like to prevent
that from happening, so we develop a system for identifying these customers and offer
them an incentive to stay. We want to target them with promotional messages and give
them a discount. We also would like to understand why the model thinks our customers
churn, and for that, we need to be able to interpret the model's predictions.


We have collected a dataset where we've recorded some information about our
customers: what type of services they used, how much they paid, and how long they
stayed with us. We also know who canceled their contracts and stopped using our 
services (churned). We will use this information as the target variable in the machine
learning model and predict it using all other available information.


The plan for the project follows:
1 First, we download the dataset and do some initial preparation: rename columns 
and change values inside columns to be consistent throughout the entire
dataset.

2 Then we split the data into train, validation, and test so we can validate our
models.

3 As part of the initial data analysis, we look at feature importance to identify
which features are important in our data.

4 We transform categorical variables into numeric variables so we can use them in
the model.

5 Finally, we train a logistic regression model.
"""

# 1 - download the dataset and do some initial preparation


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df)

# get the number of columns of the dataset

print(len(df))  # 7043 rows and 21 columns

# let's look at the first 10 rows of the dataset

print(df.head(10))

# to see the columns we can display the Transposee of the previous display

print(df.head().T)


"""
    The most interesting one for us is Churn. As the target variable for our model, 
    this is what we want to learn to predict. It takes two values: 
    yes if the customer churned and no if the customer didn't.

"""

# check whether the actual types are correct by Pandas

print(df.dtypes)

"""
    We see (figure 3.3) that most of the types are inferred correctly. Recall that object
means a string value, which is what we expect for most of the columns. However, we
may notice two things. First, SeniorCitizen is detected as int64, so it has a type 
of integer, not object. The reason for this is that instead of the values yes and no,
as we have in other columns, there are 1 and 0 values, so Pandas interprets this as a
column with integers. It’s not really a problem for us, so we don’t need to do any 
additional preprocessing for this column.


The other thing to note is the type for TotalCharges. We would expect this column to
be numeric: it contains the total amount of money the client was charged, so it should
be a number, not a string. Yet Pandas infers the type as “object.” The reason is that in
some cases this column contains a space (“ ”) to represent a missing value. 
When coming across nonnumeric characters, Pandas has no other option but to declare 
the column “object.”



IMPORTANT Watch out for cases when you expect a column to be numeric,
but Pandas says it's not: most likely the column contains special encoding for
missing values that require additional preprocessing.

We can force this column to be numeric by converting it to numbers using a special
function in Pandas: to_numeric. By default, this function raises an exception when it
sees nonnumeric data (such as spaces), but we can make it skip these cases by specifying the errors='coerce' option. This way Pandas will replace all nonnumeric values
with a NaN (not a number):
"""

total_charges = pd.to_numeric(df.TotalCharges, errors='coerce')

"""
    To confirm that data indeed contains nonnumeric characters, we can now use the
isnull() function of total_charges to refer to all the rows where Pandas couldn't
parse the original string:

(on utilise isnull() pour trouver les invalid data qui sont NaN, ça ne marche pas pour
les autres de données non valides.)

check les données invalides des features customerID et TotalCharges
"""

print(df[total_charges.isnull()][['customerID', 'TotalCharges']])


# We see that indeed there are spaces in the TotalCharges column
# ça nous montre les entrées de TotalCharges (et leur customerID correspondant)
# qui sont vides! (isnull())

"""
    Now it’s up to us to decide what to do with these missing values. 
    Although we could do
many things with them, we are going to set the missing values to zero
"""

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
df.TotalCharges = df.TotalCharges.fillna(0)

print(df[total_charges.isnull()][['customerID', 'TotalCharges']])

# Columns naming

"""
In addition, we notice that the column names don't follow the same naming convention.
Some of them start with a lower letter, whereas others start with a capital letter,
and there are also spaces in the values.

Let's make it uniform by lowercasing everything and replacing spaces with underscores.
This way we remove all the inconsistencies in the data.

"""
# replace space with _
df.columns = df.columns.str.lower().str.replace(' ', '_')
# do this particularly for categorical feature
string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# print features to see the modifications
print(df.columns)  # Good


"""
    let's look at our target variable: churn. Currently, it's categorical,
    with two values, “yes” and “no”. For binary classification, 
    all models typically expect
a number: 0 for “no” and 1 for “yes.” Let's convert it to numbers:
"""

print(df['churn'].values)  # yes or no values

df.churn = (df.churn == 'yes').astype(int)

""" 

    When we use df.churn == 'yes', we create a Pandas series of type boolean. 
    A position in the series is equal to True if it's “yes” in the original
    series and False otherwise. Because the only other value it can take is “no,” 
    this converts “yes” to True and “no” to False. 
"""

print(df['churn'].values)  # 0 and 1 now

print(df.info())

""" 
2 Then we split the data into train, validation, and test so we can validate our
models.
"""

# let’s put aside some data for testing (20%)

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

"""
The function train_test_split takes a dataframe df and creates two new dataframes:
df_train_full and df_test. It does this by shuffling the original dataset and
then splitting it in such a way that the test set contains 20% of the data and the train
set contains the remaining 80%

"""

print(df_train_full.info())
print(df_test.info())

"""
Let's take the df_train_full dataframe and split it one more time into train and
validation:
"""

""" 
Sets the random seed when doing the
split to make sure that every time we
run the code, the result is the same
"""

df_train, df_val = train_test_split(
    df_train_full, test_size=0.33, random_state=11)

""" 
    Takes the column with the target variable,
churn, and saves it outside the dataframe
"""

y_train = df_train.churn.values
y_val = df_val.churn.values

""" 
Deletes the churn columns from both dataframes to
make sure we don’t accidentally use the churn variable
as a feature during training

"""
del df_train['churn']
del df_val['churn']


"""
Now the dataframes are prepared, and we are ready to use the training 
dataset for performing initial EDA (exploratory data analysis).
"""


# 3 - As part of the initial data analysis, we look at feature importance to identify
# which features are important in our data.


"""
    Looking at the data before training a model is important. The more we know about
the data and the problems inside, the better the model we can build afterward.



We should always check for any missing values in the dataset because many machine
learning models cannot easily deal with missing data. We have already found a
problem with the TotalCharges column and replaced the missing values with zeros. Now
let's see if we need to perform any additional null handling


"""

print(df_train_full.isnull().sum())

# It prints all zeros, , so we have no missing values in the dataset and don’t
# need to do anything extra

"""Another thing we should do is check the distribution of values in the target variable.
Let's take a look at it using the value_counts() method
"""

print(df_train_full.churn.value_counts())


"""

We know the absolute numbers, but let's also check the proportion of churned
users among all customers. For that, we need to divide the number of customers who
churned by the total number of customers. We know that 1,521 of 5,634 churned, so
the proportion is:

"""

print("The proportion of churned users, or the probability that a customer will churn is :")

print("1521 / 5634 =  ", 1521 / 5634)

"""The proportion of churned users, or the probability of churning, has a special
name: churn rate.
There's another way to calculate the churn rate: the mean() method. It's more convenient
to use than manually calculating the rate
    """

global_mean = df_train_full.churn.mean()

print("the churn mean is :  ", round(global_mean, 3))

"""
Our churn dataset is an example of a so-called imbalanced dataset. There were
three times as many people who didn't churn in our dataset as those who did churn,
and we say that the nonchurn class dominates the churn class.

We can clearly see that:
the churn rate in our data is 0.27, which is a strong indicator of class imbalance.

The opposite of imbalanced is the balanced case, when positive and negative classes are
equally distributed among all observations.
    """


"""
    Both the categorical and numerical variables in our dataset are important,
    but they are
also different and need different treatment. For that, we want to look at them separately.
We will create two lists:

1- categorical, which will contain the names of categorical variables
2- numerical, which, likewise, will have the names of numerical variables
"""

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

"""
First, we can see how many unique values each variable has. We already know we should
have just a few for each column, but let's verify it:

"""

print(df_train_full[categorical].nunique())


"""


Indeed, we see that most of the columns have two or three values and one (paymentmethod)
has four. This is good. We don't need to spend extra time preparing and cleaning the data;
everything is already good to go.


Now we come to another important part of exploratory data analysis: understanding which
features may be important for our model.


IMPORTANT: Knowing how other variables affect the target variable, churn, is the key to
understanding the data and building a good model


This process is called feature importance

analysis, and it's often done as a part of exploratory data analysis to figure out which
variables will be useful for the model. It also gives us additional insights about the
dataset and helps answer questions like “What makes customers churn?” and “What
are the characteristics of people who churn?”


We have two different kinds of features: categorical and numerical. Each kind has
different ways of measuring feature importance, so we will look at each separately


Categorical variables






The first thing we can do is look at the
churn rate for each variable. We know that a categorical variable has a set of values it
can take, and each value defines a group inside the dataset.
We can look at all the distinct values of a variable. Then, for each variable, there's a
group of customers: all the customers who have this value. For each such group, we
can compute the churn rate, which is the group churn rate. When we have it, we can
compare it with the global churn rate — the churn rate calculated for all the observations
at once


If the difference between the rates is small, the value is not important when predicting
churn because this group of customers is not really different from the rest of
the customers. On the other hand, if the difference is not small, something inside that
group sets it apart from the rest. A machine learning algorithm should be able to pick
this up and use it when making predictions.


Let's check first for the gender variable. This gender variable can take two values,
female and male. There are two groups of customers: ones that have gender == 'female'
and ones that have gender == 'male'. To compute the churn rate for all female customers,
we first select only rows that correspond to gender == 'female'
and then compute the churn rate for them

"""


female_mean = df_train_full[df_train_full.gender == 'female'].churn.mean()

male_mean = df_train_full[df_train_full.gender == 'male'].churn.mean()

print("female chunk rate ", round(female_mean, 3))  # 0.27682403433476394
print("male chunk rate ", round(male_mean, 3))  # 0.2632135306553911


"""

The difference between the group rates for both females
and males is quite small, which indicates that knowing the gender of the customer
doesn't help us identify whether they will churn.






Now let's take a look at another variable: partner. It takes values of yes and no, so
there are two groups of customers: the ones for which partner == 'yes' and the ones
for which partner == 'no'.
"""

partner_yes = df_train_full[df_train_full.partner == 'yes'].churn.mean()
print("partner == yes ", round(partner_yes, 3))
partner_no = df_train_full[df_train_full.partner == 'no'].churn.mean()
print("partner == no ", round(partner_no, 3))


"""
The churn rate for people with a partner is significantly less than the rate
for the ones without a partner — 20.5% versus 33% — which indicates that the partner
variable is useful for predicting churn.




RISK RATIO
In addition to looking at the difference between the group rate and the global rate,
it's interesting to look at the ratio between them. In statistics, the ratio between 
probabilities in different groups is called the risk ratio, where risk refers to the
risk of having
the effect. In our case, the effect is churn, so it's the risk of churning:


risk = group rate / global rate

For gender == female, for example, the risk of churning is 1.02:
risk = 27.7% / 27% = 1.02

Risk is a number between zero and infinity. It has a nice interpretation that tells you
how likely the elements of the group are to have the effect (churn) compared with the
entire population.
"""

# for gender == male, the risk of churning is

print("for gender == male the risk of churning is ",
      round(male_mean/global_mean, 3))  # 0.975

# for partner == yes, the risk of churning is
print("for partner == yes the risk of churning is ",
      round(partner_yes/global_mean, 3))  # 0.759

# for partner == no, the risk of churning is
print("for partner == no the risk of churning is ",
      round(partner_no/global_mean, 3))  # 1.222

""" 
If the difference between the group rate and the global rate is small, the risk is
close to 1: this group has the same level of risk as the rest of the population. 
Customers in the group are as likely to churn as anyone else. In other words, 
a group with a risk close to 1 is not risky at all.

If the risk is lower than 1, the group has lower risks: the churn rate in this group
is smaller than the global churn. For example, the value 0.5 means that the clients
in this group are two times less likely to churn than clients in general
"""

# for each categorical variable, let's write a program that would determine the likelihood
# of each group in this variable to chunk


for variable in categorical:
    # get the uniques values in list for each variable
    # print(sorted(set(df_train_full[variable].values)))
    for values_list in sorted(set(df_train_full[variable].values)):
        print(
            f"The churn rate of {variable} == {values_list} is {round(df_train_full[df_train_full[variable]== values_list].churn.mean(), 3) * 100}% and the risk of churn rate is {round(df_train_full[df_train_full[variable]== values_list].churn.mean()/global_mean, 3)*100}%")

        # determine if the group is risky or not
        if round(df_train_full[df_train_full[variable] == values_list].churn.mean()/global_mean, 3) > 1:
            print(f"{variable} == {values_list} is a risky group")
        else:
            print(f"{variable} == {values_list} is not a risky group")


# another mean

global_mean = df_train_full.churn.mean()

# df_group = df_train_full.groupby(by='gender').churn.agg(['mean'])
# df_group['diff'] = df_group['mean'] - global_mean
# df_group['risk'] = df_group['mean'] / global_mean
# print(df_group)


for variable in categorical:

    # Computes the AVG(churn)

    # For that, we use the agg function to indicate
    # that we need to aggregate data into one value per group: the mean value.
    df_group = df_train_full.groupby(by=variable).churn.agg(['mean'])
    # Calculates the difference between group churn and global rate

    #  we create another column, diff, where we will keep the difference between the group mean
    # and the global mean.
    df_group['diff'] = df_group['mean'] - global_mean
    # Calculates the risk of churning

    # we create the column risk, where we calculate
    # the fraction between the group mean and the global mean
    df_group['risk'] = df_group['mean'] / global_mean
    print(df_group)


"""
    This way, just by looking at the differences and the risks, 
    we can identify the most discriminative features: the features 
    that are helpful for detecting churn. Thus, we
expect that these features will be useful for our future models.


The kinds of differences we just explored are useful for our analysis and important for
understanding the data, but it's hard to use them to say what the most important feature
is and whether tech support is more useful than the type of contract.



Luckily, the metrics of importance can help us: we can measure the degree of
dependency between a categorical variable and the target variable. If two variables are
dependent, knowing the value of one variable gives us some information about
another. On the other hand, if a variable is completely independent of the target 
variable, it's not useful and can be safely removed from the dataset.



In our case, knowing that the customer has a month-to-month contract may indicate
that this customer is more likely to churn than not.


IMPORTANT 
Customers with month-to-month contracts tend to churn a lot
more than customers with other kinds of contracts. This is exactly the kind of
relationship we want to find in our data. Without such relationships in data,
machine learning models will not work — they will not be able to make predictions.
The higher the degree of dependency, the more useful a feature is.



For categorical variables, one such metric is MUTUAL INFORMATION, which tells how
much information we learn about one variable if we learn the value of the other variable. It’s a concept from information theory, and in machine learning, we often use it
to measure the mutual dependency between two variables.

Higher values of mutual information mean a higher degree of dependence

Mutual information is already implemented in Scikit-learn in the mutual_info_
score function from the metrics package, so we can just use it

"""


# Creates a stand-alone function for calculating mutual information


def calculate_mi(series):
    # Uses the mutual_info_score function from Scikit-learn
    return mutual_info_score(series, df_train_full.churn)


# Applies the function from to each categorical column of the dataset
df_mi = df_train_full[categorical].apply(calculate_mi)
# Sorts the values of the result
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
print(df_mi)

# according to the mutual information table, contract variable has the more dependence
# with the target variable

"""

CORRELATION COEFFICIENT

Mutual information is a way to quantify the degree of dependency between 
two categorical variables, but it doesn't work when one of the features is numerical,
so we cannot apply it to the three numerical variables that we have.
We can, however, measure the dependency between a binary target variable and a
numerical variable. 

We can pretend that the binary variable is numerical (containing
only the numbers zero and one) and then use the classical methods from statistics to
check for any dependency between these variables.

One such method is the correlation coefficient 
(sometimes referred as Pearson's correlation coefficient). It is a value from –1 to 1*
"""

print("the correlation of numerical features to 'chunk' is : \n")

print(
    df_train_full[numerical].corrwith(df_train_full.churn))


"""
tenure has a high negative correlation: as tenure grows, churn rate goes down.
monthlycharges has positive correlation: the more customers pay,
the more likely they are to churn.


After doing initial exploratory data analysis, identifying important features,
and getting some insights into the problem, we are ready to do the next step: 
feature engineering and model training.

    """

"""
4 We transform categorical variables into numeric variables so we can use them in
the model.





Before we proceed to training, however, we need to perform the feature engineering step: 
transforming all categorical variables to numeric features. We'll do
that in the next section, and after that, we'll be ready to train the logistic regression
model.


- One Hot Encoding

As we already know from the first chapter, we cannot just take a categorical variable
and put it into a machine learning model. The models can deal only with numbers
in matrices. So, we need to convert our categorical data into a matrix form, or
encode.

One such encoding technique is one-hot encoding.

    """
    
# first method

train_dict = df_train[categorical + numerical].to_dict(orient='records')
print(train_dict)
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

print(X_train[0])

# We can learn the names of all these columns by using the get_feature_names method

print(dv.get_feature_names_out())

""" 
As we see, for each categorical feature it creates multiple columns for each of 
its distinct values. For contract, we have contract=month-to-month, contract=one_year,
and contract=two_year, and for dependents, we have dependents=no and dependents
=yes. Features such as tenure and totalcharges keep the original names because
they are numerical; therefore, DictVectorizer doesn't change them.
Now our features are encoded as a matrix, so we can move to the next step: using a
model to predict churn.





We have learned how to use Scikit-learn to perform one-hot encoding for categorical
variables, and now we can transform them into a set of numerical features and put
everything together into a matrix.
When we have a matrix, we are ready to do the model training part. In this section
we learn how to train the logistic regression model and interpret its results.

"""
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', random_state=1)
model.fit(X_train, y_train)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)


