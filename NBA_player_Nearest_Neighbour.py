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
def calculate_distance(x1, x2, length):
    distance = 0
    i = 0
    for i in range(length):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)
def getNeighbour(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance)
    for i in range (len(trainingSet)):
        dist = calculate_distance(trainingSet[i], testInstance, length)
        distance.append((trainingSet[i], dist, i))
        #print("is this a label? : ", i)
    distance.sort(key = operator.itemgetter(1))
    #print("distance is : ", distance)
    neighbours = []
    for i in range(k):
        neighbours.append(distance[i][2])
    return neighbours

def predict_label(player_mins, player_points, neighbours, train_y, correct_label, even_vote):
    #print("player index is : ", player_mins, player_points)
    label_array = np.zeros(number_of_class+1) #offset 1 more, because we dont use 0
    label_array[correct_label] -= 1
    for i in range (len(neighbours)):
        label_array[train_y[neighbours[i]]] += 1
    most_vote = 0
    player_label = 0
    for i in range (len(label_array)):
        if(label_array[i] > most_vote):
            most_vote = label_array[i]
            player_label = i
        else:
            if label_array[i] == most_vote and most_vote != 0:
                even_vote += 1
                #print("even vote label: ", label_array)
                #print("even vote: ", even_vote)


    #print("The neighbours of this NBA player is : ", label_array)
    return [player_label, even_vote]

k = 5
print("k is : ", k)
correct = 0
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2, random_state=1)
even_vote = 0
for i in range(len(X_train)):
    result_neighbours = getNeighbour(X_train, X_train[i], k+1)
    predict_result = predict_label(X_train[i][1], X_train[i][2], result_neighbours, Y_train, Y_train[i], even_vote)
    #label = predict_label(result_neighbours, Y_train, Y_train[i], even_vote)
    label = predict_result[0]
    even_vote = predict_result[1]
    #print("predict label is : ", label, " and the right label is: ", Y_train[i])
    if label == Y_train[i]:
        correct += 1
        #print("yes")
    else:
        print("incorrect, should be, ", Y_train[i], ", but is: ", label)
        print("incorrect, index : ", X_train[i][1], " and ", X_train[i][2])
        print("label is : ", result_neighbours)
print("correct is : ", correct)
print("total number is : ", len(X_train))
train_accuracy = correct/len(X_train)
print("Training Accuracy is : ", train_accuracy)

correct = 0
even_vote = 0
for i in range(len(X_test)):
    result_neighbours = getNeighbour(X_train, X_test[i], k)
    predict_result = predict_label(X_test[i][1], X_test[i][2], result_neighbours, Y_train, Y_test[i], even_vote)
    # label = predict_label(result_neighbours, Y_train, Y_train[i], even_vote)
    label = predict_result[0]
    even_vote = predict_result[1]
    #print("predict label is : ", label, " and the right label is: ", Y_train[i])
    if label == Y_test[i]:
        correct += 1
        #print("yes")
    else:
        print("incorrect, should be, ", Y_test[i], ", but is: ", label)
        print("incorrect, index : ", X_test[i][1], " and ", X_test[i][2])
print("correct is : ", correct)
print("total number is : ", len(X_test))
test_accuracy = correct/len(X_test)
print("Training Accuracy is : ", train_accuracy)
print("Testing Accuracy is : ", test_accuracy)
