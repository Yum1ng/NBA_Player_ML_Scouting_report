import scipy.io as sio
import numpy as np
import xlrd
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import tree
import random
import math
import operator

data = xlrd.open_workbook('NBA_15_16_player_basic2.xls')
table = data.sheets()[0]
ncols = table.ncols
nrows = table.nrows
print("column is : ", ncols)
print("row is L ", nrows)
X_temp = np.zeros([477, 25])
for i in range(477):
    for j in range(25):
        if i == 0 or j == 0 or j == 1 or j == 2 or j == 3:
            continue
        else:
            X_temp[i][j] = table.cell(i, j).value
print(X_temp)
X = X_temp[1:477, 4:25]
print(X) # now we get all the data!!

y = np.zeros(len(X))
for i in range(len(X)):
    if X[i][20] == 7 or X[i][20] == 8:
        y[i] = 7
    else:
        if X[i][20] == 5 or X[i][20] == 6:
            y[i] = 5
        else:
            y[i] = X[i][20]
print(y)
print("len(x) ", len(X))
print("len(x)(1): ",  len(X[0]))
X = X[0:len(X), 0:20]
print(X)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.4, random_state=1)
print("len(x_train): ", len(x_train))
print("len(x_test): ", len(x_test))
print("len(y_train): ", len(y_train))
print("len(y_test): ", len(y_test))

depth = []
maxmean = 0
best_depth = 0
for i in range(1, 15):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    # Perform 7-fold cross validation
    scores = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10)
    if scores.mean() > maxmean:
        maxmean = scores.mean()
        best_depth = i
    depth.append((i,scores.mean()))
print("depth is : ", depth)
print("max_depth is : ", best_depth)

clf = tree.DecisionTreeClassifier(max_depth=best_depth)
clf.fit(x_train, y_train)

train_correct = 0
for i in range(len(x_train)):
    if clf.predict((x_train[i].reshape(1, -1))) == y_train[i]:
        train_correct += 1
        #print("correct")
print("training accuracy for 80% training set: ", train_correct/len(x_train))

test_correct = 0
for i in range(len(x_test)):
    if clf.predict((x_test[i].reshape(1, -1))) == y_test[i]:
        test_correct += 1
        #print("correct")
print("testing accuracy for 20% testing set: ", test_correct/len(x_test))




