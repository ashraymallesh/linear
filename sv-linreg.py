import numpy as np
import matplotlib.pyplot as plt

aData = np.genfromtxt('dataset.csv', delimiter=',', skip_header=1)
m,n = aData.shape
x = aData[:,0].reshape(-1,1)
y = aData[:,-1].reshape(-1,1)
X = np.c_[np.ones((m,1)), aData[:,0]] # create design matrix X by adding ones as left col (x_0 = 1)

def hyp(theta, X):
	return np.dot(X,theta)

def cost(theta, X, y):
	#compute cost using vectorized numpy computation instead of for loops
	error = hyp(theta, X) - y 
	return np.sum(error**2) / (2*m)

def gradientDescent(theta, X, y, alpha, iters):
    for _ in range(iters):
        error = hyp(theta, X) - y
        theta = theta - ((alpha/m) * np.dot(X.T, error))
    return theta

def plot(theta, X, y):
	plt.scatter(X[:,1],y) #original points scatterplot
	plt.plot(X[:,1], hyp(theta, X), 'r') #plot regression line
	plt.show()

theta = np.zeros((n,1)) # theta vector with 2 zeroes
alpha = 0.01; iters = 1500
theta = gradientDescent(theta, X, y, alpha, iters)
plot(theta, X, y)