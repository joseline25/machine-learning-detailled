"""

Binary decision trees operate by subjecting attributes to a series of binary 
(yes/no) decisions. Each decision leads to one of two possibilities. Each decision
leads to another decision or it leads to prediction.

"""

__author__ = 'joseline-youego'


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from math import sqrt
import matplotlib.pyplot as plot


data = pd.read_csv('winequality-red.csv')
print(data)


xList = []

labels = []
names = []
firstLine = True
for line in data:
    if firstLine:
        names = line.strip().split(";")
        firstLine = False
    else:
        #split on semi-colon
        row = line.strip().split(";")
        #put labels in separate array
        labels.append(float(row[-1]))
        #remove label from row
        row.pop()
        #convert row to floats
        floatRow = [float(num) for num in row]
        xList.append(floatRow)
        
# nrows = len(xList)
# ncols = len(xList[0])
wineTree = DecisionTreeRegressor(max_depth=3)
wineTree.fit(xList, labels)


with open("wineTree.dot", 'w') as f:
    f = tree.export_graphviz(wineTree, out_file=f)

# data_train_full, data_test = train_test_split(
#     data, test_size=0.2, random_state=11)
# data_train, data_val = train_test_split(data_train_full, test_size=0.25,
#                                         random_state=11)


# y_train = (data_train.quality).values
# y_val = (data_val.quality).values


# del data_train['quality']
# del data_val['quality']


# scaler = RobustScaler()

# X_train = scaler.fit_transform(data_train)
# X_val = scaler.transform(data_val)


# wineTree = DecisionTreeRegressor(max_depth=3)
# wineTree.fit(X_train, y_train)

# y_pred = dt.predict_proba(X_train)[:, 1]
# print(y_pred)