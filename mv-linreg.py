import numpy as np
import matplotlib.pyplot as plt

aData = np.genfromtxt('ex1data2.txt', delimiter=',', skip_header=1)
m,n = aData.shape
X = aData[:,0:-1]
y = aData[:,-1].reshape(-1,1)
mean = np.mean(X, axis=0); std = np.std(X, axis=0) #store mean & std for each feature (col)
X = (X - mean)/std #column wise feature normalization
X = np.c_[np.ones(m), X]
theta = np.zeros((n,1))

def hyp(theta, X):
	return np.dot(X, theta)

def cost(theta, X, y):
	error = hyp(theta, X) - y
	return np.sum(error**2) / (2*m)

def gradientDescent(theta, X, y, alpha, iters):
	for _ in range(iters):
		  