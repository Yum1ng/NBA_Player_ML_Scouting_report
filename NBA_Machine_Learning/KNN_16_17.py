import scipy.io as sio
import numpy as np
import xlrd
import xlwt
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
import random
import math
import operator

test_data = xlrd.open_workbook('NBA_17_18_player_basic2.xls')
test_table = test_data.sheet_by_index(0)
nrows = test_table.nrows
test_temp = np.zeros([nrows, 24])
nrows = test_table.nrows
for i in range(nrows):
    for j in range(24):
        if i == 0 or j == 0 or j == 1 or j == 2 or j == 3:
            continue
        else:
            test_temp[i][j] = test_table.cell(i, j).value
test = test_temp[1:nrows, 5:24]
print("test:", test)

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
X = X_temp[1:477, 5:25]
print(X)  # now we get all the data!!
dimension = 19
y = np.zeros(len(X))
for i in range(len(X)):
    if X[i][dimension] == 7 or X[i][dimension] == 8:
        # y[i] = 6
        y[i] = 7
    else:
        if X[i][dimension] == 5 or X[i][dimension] == 6:
            # y[i] = 5
            y[i] = X[i][dimension]

        else:
            y[i] = X[i][dimension]
print(y)
print("len(x) ", len(X))
print("len(x)(1): ", len(X[0]))
X = X[0:len(X), 0:dimension]
print(X)

c = 0.1
optimal_c = 0.1
maxa = 0
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X)
clf = KNeighborsRegressor(n_neighbors=5)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv)
print("max scores: ", scores)
clf.fit(X, y)
print("---------------------------------------------")
print("Real Testing Begin")
workbook = xlwt.Workbook()
result_sheet = workbook.add_sheet('result')
for i in range(len(test)):
    label = clf.predict((test[i].reshape(1, -1)))
    print("index : ", i+2, "label is : ", label)
    label = label[0]
    #neighbour = clf.kneighbours(i, 5)
    print("neighbour is : ", neigh.kneighbors(test[i].reshape(1, -1)))

    result_sheet.write(i + 1, 1, label)
workbook.save("17_18__KNN_output.xls")