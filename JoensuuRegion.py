# Author: @ Jack Volonte
# Date 11/2/2022
# Description : This program implements a clustering k-means algorithim to determine the center of clusters of data overtime. Producing a graph of the
# start of the scatter of data and initial cluster datapoint choices, and the final output - scatter of data and end clustering datapoint choices.
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


# to print after each iteration
def printThing(Cx, Cy):
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Clustering - Joesnuu Region')
    # plot data (black) and centroids (yellow)
    plt.scatter(X, Y)
    plt.scatter(Cx, Cy, c='r')
    plt.show()

# get distance from a centroid C
def getDistanceFrom(Cx, Cy, X, Y, index):
    arrayOfDist = np.zeros(len(X))
    for count in range(len(arrayOfDist)):
        arrayOfDist[count] = math.sqrt((X[count] - Cx[index]) ** 2 + (Y[count] - Cy[index]) ** 2)  # euclidean distance
    return arrayOfDist

def getSingleDistance(Cx, Cy, X, Y, index, r):
    a = math.sqrt((X[index] - Cx[r])**2 + (Y[index] - Cy[r])**2) # change distance to num_centroid or change Cx to len of X
    return a

def ChangeLabels(LABELS, Distance, lastDistance, centroidNumber):  # changing centroid labels for each datapoint
    goThrough = True
    for r in range(len(lastDistance)):
        for c in range(len(lastDistance)):
            if lastDistance[r][c] != 0:
                goThrough = False
    if goThrough == True:
        for i in range(NUM_CENTROIDS):
            for count in range(len(Distance)):  # for each X value
                if Distance[count] < lastDistance[i][count]:  # check if distance is less than last centroid
                    LABELS[count] = centroidNumber + 1  # giving labels the centroid number + 1
    if goThrough == False:
         for count in range(len(LABELS)):
             LABELS[count] = centroidNumber + 1

def getMeanDistX(LABELS, CentroidNumber):  # mean distance from X
    sum = 0.0
    countt = 0
    for count in range(len(LABELS)):
        if LABELS[count] == CentroidNumber:
            sum = sum + X[count]
            countt = countt + 1

    if countt != 0:
        if sum != 0:
            return sum / countt
    return 0.0

def getMeanDistY(LABELS, CentroidNumber):  # mean distance from Y
    sum = 0.0
    countt = 0
    for count in range(len(LABELS)):
        if LABELS[count] == CentroidNumber:
            sum = sum + Y[count]
            countt = countt + 1
    if countt != 0:
        if sum != 0:
            return sum / countt
    return 0.0

def getEntropy(Cx, Cy, X, Y, i, LABELS):
    count = np.zeros(len(Cx))
    for r in range(len(LABELS)):
        b = LABELS[r] - 1
        b = int(b)
        count[b] = count[b] + 1
    aa = 0.0
    for a in range(len(count)):
        if count[a] != 0:
            aa = aa + (count[a]/len(X))*math.log2(count[a]/len(X))
    return aa


NUM_CENTROIDS = 4
data = pd.read_csv('JoensuuRegion.txt', delimiter=',', header=None,
                   names=['X', 'Y'])

# normalize data
Xx  = data['X']
Yy = data['Y']
X = (Xx-min(Xx))/(max(Xx)-min(Xx))
Y = (Yy-min(Yy))/(max(Yy)-min(Yy)) # normalize data

LABELS = np.zeros(len(X)) # initalize labels

# randomly generate predetermined close to indexs ( RANDOM WAS MAKING IT MESS UP BAD)
C = np.random.choice(np.shape(X)[0], NUM_CENTROIDS)
Cx = np.zeros(len(C))
Cy = np.zeros(len(C))
for i in range(len(C)):
    Cx[i] = X[C[i]]
    Cy[i] = Y[C[i]]
newC = np.zeros(len(C)) # for getting new centroids after assigning and getting mean distance

# get starting centroids using k-means++ algo
n = 0
while n < NUM_CENTROIDS:
    distFrom = np.zeros(len(X))
    maxx = 0
    for i in range(NUM_CENTROIDS):
        if n != 0:
            distFrom = getDistanceFrom(Cx, Cy, X, Y, i)
            maxx = max(distFrom)
            if max(distFrom) > maxx:
                maxx = max(distFrom)
    if n != 0:
        for b in range(len(distFrom)):
            if maxx == distFrom[b]:
                Cx[i] = X[b]
                Cy[i] = Y[b]
    n = n + 1


# set starting label for each centroid
for i in range(len(X)):
    distLast = 100000
    for r in range(NUM_CENTROIDS):
        distEach = getSingleDistance(Cx, Cy, X, Y, i,r)
        if distLast != 100000:
            if distEach <= distLast:
                LABELS[i] = r + 1
        elif distLast == 100000:
            LABELS[i] = 1
        distLast = distEach

# iterations
iterations = 50
printThing(Cx, Cy)

# entropy array
entropy = np.zeros(iterations)

# main loop, iterate iterations number of times
c = 0
while c in range(iterations):

    # - assign all data to closest centroid
    for i in range(len(X)):
        distLast = 0.0
        for r in range(NUM_CENTROIDS):
            distEach = getSingleDistance(Cx, Cy, X, Y, i,r)
            if distEach < distLast:
                LABELS[i] = r + 1
            distLast = distEach


    # for check if mean is moving anymore
    oldCx = Cx
    oldCy = Cy
    # - get new centroids from the mean center of each centroid's data assigned to it
    for b in range(NUM_CENTROIDS):
        # get mean distance
        xMean = getMeanDistX(LABELS, b + 1)
        yMean = getMeanDistY(LABELS, b + 1)


        # set new centroids
        Cx[b] = xMean
        Cy[b] = yMean

    # for entropy graph
    entropy[c] = getEntropy(Cx, Cy, X, Y, i, LABELS)
    c = c + 1



print(Cx)
print(Cy)
printThing(Cx, Cy) # print final graph

plt.title('Mean Entropy - Joesnuu Region')
plt.plot(np.arange(0,iterations,1), entropy, color="orange") # entropy accuracy
plt.show()