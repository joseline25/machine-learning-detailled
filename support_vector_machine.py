"""
Modèle de classification binaire et multiclasse.
Support Vector Machine est l'une des méthodes les plus simples et élegantes pour
la classification

SVM effectue la classification en dessinant une ligne (hyperplan)entre les points
pour séparer les 2 ou plusieurs catégories. 

SVM se concentre sur la recherche du meilleur hyplan de separation car il peut 
y en avoir plusieurs. 

On a une distance entre les pointsdes categories et l'hyperplan appelée: margin

On ne cherche pas forcément que le margin soit le plus petit possible 
mais plutôt comment avoir l'hyperplan qui fait la meilleure classification.
On cherche à maximiser le margin qui est la distance entre les points
les plus proches de l'hyperplan et l'hyperplan. 

Ces points sont appelés: supporting vectors (d'où le nom du model)

Si on a 2 features, l'hyperplan est une ligne 
si on a 3 features, l'hyperplan est un plan
si on a plus de 3 features, on a tout simplment un hyperplan! qui est
un peu difficile à visualiser.

2 techniques: Gamma et regularisation

SVM est considéré comme un supervised learning algorithm car les categories sont
connues d'avance.

En background, SVM résoud le problème de d'Optimisation Convexe qui maximise le 
margin.

We have a dedicated module for it in sklearn

from sklear import svm

SVM is easy to
    -understand
    -implement
    -use
    -interpret
    
It is effective when the dataset is small!!

Dans les cas où on ne peut pas séparer les données par un hyperplan selon
leur disposition, on peut implementer les techniques suivantes:

- créer des features supplémentaires à partir de celles existant par des
combinaisons (a)

- dans cette dimension supérieure (puisqu'on a de nouvelles features),
trouver l'hyperplan (b) et ensuite projeter sur l'espace précedent (c)

- la technique du Kernel Trick nous permet d'exécuter toutes ces étapes 
de façon efficace.

Uses Cases:
- face recognition
- spam filtration
- text recognition

Machine Learning Tutorial Python - 10 Support Vector Machine (SVM) on Youtube

iris flower : 4 features (petals width and height, sepal width and height)
We will work with petal length and petal width

"""

from sklearn.svm import SVC, SVR
# we will only use SVC (Support Vector Classifier)
# SVR is Support Vector Regression
import pandas as pd
from sklearn.datasets import load_iris, load_digits
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

iris = load_iris()

# get the properties of the dataset

print(dir(iris))
# ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame',
# 'target', 'target_names']

print(iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


# let's create a dataframe out of this because it is easy to visualize
# and work with datframe

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

""" 
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2

"""

# append the target of iris to the newly created dataframe

df['target'] = iris.target
print(df.head())

""" 
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
2                4.7               3.2                1.3               0.2       0
3                4.6               3.1                1.5               0.2       0
4                5.0               3.6                1.4               0.2       0


when we print iris.target_names, we have 
['setosa' 'versicolor' 'virginica']

It means, whenever we have 
- 0 == 'setosa',
- 1 == 'versicolor'
- 2 == 'virginica'
"""
# let's create a new feature (column) named flower_name

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
# it means if target = 0, in flower_name column we have setosa
print(df.head())

"""

 sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target flower_name
0                5.1               3.5                1.4               0.2       0      setosa
1                4.9               3.0                1.4               0.2       0      setosa
2                4.7               3.2                1.3               0.2       0      setosa
3                4.6               3.1                1.5               0.2       0      setosa
4                5.0               3.6                1.4               0.2       0      setosa

"""

#  Visualization


df0 = df[df.target == 0]

df1 = df[df.target == 1]

df2 = df[df.target == 2]

print(df1.head(7))

"""

    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target flower_name
50                7.0               3.2                4.7               1.4       1  versicolor
51                6.4               3.2                4.5               1.5       1  versicolor
52                6.9               3.1                4.9               1.5       1  versicolor
53                5.5               2.3                4.0               1.3       1  versicolor
54                6.5               2.8                4.6               1.5       1  versicolor
55                5.7               2.8                4.5               1.3       1  versicolor
56                6.3               3.3                4.7               1.6       1  versicolor

"""

# let's plot the sepal height and width features of df0 and df1
plt.xlabel('sepal length (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],
            color='green', marker='+')

plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],
            color='blue', marker='.')
plt.show()


# let's plot the petal height and width features of df0 and df1
plt.xlabel('petal length (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],
            color='green', marker='+')

plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],
            color='blue', marker='.')
plt.show()


# split and train the dataset
X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target
print(X.head())
print(y.head())

"""
X
  sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2


y
0    0
1    0
2    0
3    0
4    0
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# création du modèle

model = SVC()

# train the model

model.fit(X_train, y_train)
"""
gamma = 'auto', kernel = 'rbf', ...
"""

# get the accuracy of the model
print(model.score(X_test, y_test))  # 96%

# parameter tunning
# we will work on C the regularization parameter. By default, C=1.0
# model = SVC(C=10) or
# model = SVC(gamma=100) #46,6%
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

"""
We can also tune the model through kernel
By default, kernel='rbf'. We will change it to the value 'linear'
model = SVC(kernel='linear') # 93%
it can be 'linear', 'rbf', 'poly', 'sigmoid', 'precomputed' or a callable


Exercise: 
Train a SVM classifier using sklearn digits dataset
(ie from sklearn.datasets import load_digits ) and then

- measure accuracy of your model using diffrent kernels such as linear and rbf
-  Tune your model further using regularization (C) and gamma parameter
- Use 80% of the dataset for training

"""

digits = load_digits()

# get the properties of the dataset

print(dir(digits))

# ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']

print(digits.feature_names)

""" 

['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5',
'pixel_0_6', 'pixel_0_7', 'pixel_1_0', 'pixel_1_1', 'pixel_1_2', 'pixel_1_3', 
'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7', 'pixel_2_0', 'pixel_2_1', 
'pixel_2_2', 'pixel_2_3', 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7',
'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5',
'pixel_3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3',
'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7', 'pixel_5_0', 'pixel_5_1', 
'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 
'pixel_6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5',
'pixel_6_6', 'pixel_6_7', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3',
'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7']

64 features
"""
# let's create a dataframe out of this because it is easy to visualize
# and work with datframe

data = pd.DataFrame(digits.data, columns=digits.feature_names)
print(data.head())

# append the target of digits to the newly created dataframe

data['target'] = digits.target
print(data.head())
print(digits.target_names)  # [0 1 2 3 4 5 6 7 8 9]

# let's plot the pixel_0_3 and pixel_3_0 features of data
plt.xlabel('pixel_0_3')
plt.scatter(data['pixel_3_0'], data['pixel_0_3'],
            color='green', marker='+')

plt.scatter(data['pixel_0_4'], data['pixel_4_0'],
            color='blue', marker='.')
plt.show()


# split and train the dataset
X = data.drop(['target'], axis='columns')
y = data.target
print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# création du modèle

# model = SVC()

# train the model

# model.fit(X_train, y_train)
# get the accuracy of the model
# print(model.score(X_test, y_test))#99.16%

# Tunning

# on regularization
# model = SVC(C=100)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))# 98,3%

# with gamma parameter
# model = SVC(gamma=10) # 7.7% with gamma = 100 and 8.3% with gamma = 10
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test)) # 7,2%


model = SVC(kernel='poly')
# train the model
model.fit(X_train, y_train)
# get the accuracy of the model
print(model.score(X_test, y_test))  # 97,2% with kernel = 'linear', 
# 87,2% with kernel = 'sigmoid', 98% with kernel = 'poly'
