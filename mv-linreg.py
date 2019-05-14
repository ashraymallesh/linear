#single variable linear regression in python
#variable prefixes: v = vector, a = array, s = string, i = int, f = float (vectors and arrays are numpy ndarrays)

import numpy as np 
import matplotlib.pyplot as plt

#some global variables
i_n = 0 #number of features
i_m = 0 #number of training examples

def computeCost(aX, v_y, vTheta):
    #calculate cost for a given theta vector
    sum = 0 #summation term in cost function
    for i in range(1,i_m): 
        sum += (h(vTheta, aX[i,:]) - v_y[i])**2
    fCost = (1/(2*i_m))*sum
    return fCost

def h(vTheta, v_x):
    #hypothesis function on a vector x
    return np.dot(vTheta,v_x)

def gradientDescent():
    vTheta = np.zeros((2,1))
    return vTheta

def loadData(sCSVFilename):
    global i_m, i_n
    #load data and plot if 2D
    aData = np.loadtxt(sCSVFilename, delimiter=',')
    aX = aData[:,:-1] #design matrix X with each row as x_i^T
    v_y = aData[:,-1] #vector y with each row as y_i
    i_m = v_y.size #update global m (# training examples)
    i_n = aX.shape[1] #update global n (# features)

    #plot data
    if aData.shape[1] == 2: #only if aData has 2 columns
        plt.scatter(aData[:,0], aData[:,1]) #scatterplot using first,second cols of aData as x,y
        plt.show()

    return aData, aX, v_y

aData, aX, v_y = loadData('ex1data1.txt')
i_initialCost = computeCost(aX, v_y, np.zeros(2,1)) #theta = [0 0]
print('Initial Cost with Theta = [0 0] is', i_initialCost)
