# single variable linear regression in python
# variable prefixes: v = vector, a = array, s = string, i = int, f = float (vectors and arrays are numpy ndarrays)

import numpy as np 
import matplotlib.pyplot as plt

# some global variables
i_n = 0 # number of features
i_m = 0 # number of training examples

def computeCost(aX, v_y, vTheta):
    # calculate cost for a given theta vector
    sum = 0 # summation term in cost function
    for i in range(1,i_m): 
        sum += (h(vTheta, aX[i,:]) - v_y[i])**2
    fCost = (1/(2*i_m))*sum
    return fCost

def h(vTheta, v_x):
    # hypothesis function on a vector x
    return np.dot(vTheta,v_x)

def gradientDescent():
    vTheta = np.zeros((2,1))
    return vTheta

def loadData(sCSVFilename):
    global i_m, i_n
    aData = np.loadtxt(sCSVFilename, delimiter=',')
    v_x = aData[:,0].reshape(-1,1) #reshape(-1,1) is basically hor->ver transpose for a 1D ndarray
    v_y = aData[:,-1].reshape(-1,1) # vector y with each row as y_i
    i_m = v_y.size # (num training examples)
    ones = np.ones((i_m, 1)) # m-dim vector filled with 1s
    aX = np.concatenate((ones, v_x), 1) # design matrix X with each row as x_i^T
    i_n = aX.shape[1] # (num features)

    plt.scatter(v_x, v_y) # scatterplot using first,second cols of aData as x,y
    plt.show() 

    return aX, v_y

aX, v_y = loadData('ex1data1.txt')
