import numpy as np
import matplotlib.pyplot as plt

aData = np.genfromtxt('dataset.csv', delimiter=',', skip_header=1)
m,n = aData.shape
x = aData[:,0].reshape(-1,1)
y = aData[:,-1].reshape(-1,1)
X = np.c_[np.ones((m,1)), aData[:,0]] # create design matrix X by adding ones as left col (x_0 = 1)

def hyp(X, theta):
	if X.shape == theta.shape:
		return np.dot(X,theta)
	else:
		return X @ theta # mxn @ nx1 --> mx1

def cost(theta, X, y):
	#compute cost using vectorized numpy computation instead of for loops
	error = hyp(X, theta) - y 
	return np.sum(np.power(error, 2)) / (2*m)

def gradientDescent(X, y, theta, alpha, iters):
    for _ in range(iters):
        error = hyp(X, theta) - y
        theta = theta - ((alpha/m) * np.dot(X.T, error))
    return theta

def plot(theta, X, y):
	plt.scatter(X[:,1],y) #original points scatterplot
	plt.plot(X[:,1], hyp(X, theta), 'r') #plot regression line
	plt.show()

theta = np.zeros((n,1)) # theta vector with 2 zeroes
alpha = 0.01; iters = 1500
theta = gradientDescent(X, y, theta, alpha, iters)
plot(theta, X, y)