import scipy.io as sio
import numpy as np
import xlrd
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
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
dimension = 20
y = np.zeros(len(X))
number_of_class = 6
for i in range(len(X)):
    if X[i][dimension] == 7 or X[i][dimension] == 8:
        y[i] = 6
    else:
        if X[i][dimension] == 5 or X[i][dimension] == 6:
            y[i] = 5
        else:
            y[i] = X[i][dimension]
print(y)
print("len(x) ", len(X))
print("len(x)(1): ",  len(X[0]))
X = X[0:len(X), 0:dimension]
print(X)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.6,random_state=1)

depth = []
maxmean = 0
best_depth = 0
for i in range(1, 15):
    clf = RandomForestClassifier(n_estimators=11, max_depth=i, random_state=0)
    # Perform 7-fold cross validation
    scores = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10)
    if scores.mean() > maxmean:
        maxmean = scores.mean()
        best_depth = i
    depth.append((i,scores.mean()))
print("depth is : ", depth)
print("max_depth is : ", best_depth)

RandomForestClassifier(n_estimators=11, max_depth=best_depth, random_state=0)
clf.fit(x_train, y_train)

train_correct = 0
for i in range(len(x_train)):
    if clf.predict((x_train[i].reshape(1, -1))) == y_train[i]:
        train_correct += 1
        #print("correct")
print("training accuracy for 80% training set: ", train_correct/len(x_train))
clf.fit(x_test, y_test)
test_correct = 0
for i in range(len(x_test)):
    if clf.predict((x_test[i].reshape(1, -1))) == y_test[i]:
        test_correct += 1
        #print("correct")
print("testing accuracy for 20% testing set: ", test_correct/len(x_test))
