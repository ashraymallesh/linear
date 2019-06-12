import numpy as np
import matplotlib.pyplot as plt

aData = np.genfromtxt('dataset.csv', delimiter=',', skip_header=1)
m,n = aData.shape
x = aData[:,0].reshape(-1,1)
y = aData[:,-1].reshape(-1,1)
X = np.c_[np.ones((m,1)), aData[:,0]] # create design matrix X by adding ones as left col (x_0 = 1)

def hyp(x, theta):
	if x.shape == theta.shape: return np.dot(x,theta)
	else: return x @ theta # mxn @ nx1 -> mx1 (matrix multiplication)

def cost(theta, x, y):
	#compute cost using vectorized numpy computation instead of using for-loops
	error = hyp(x, theta) - y 
	return np.sum(error**2) / 2*m

def gradientDescent(x, y, theta, alpha, iters):
    for i in range(iters):
        error = hyp(x, theta) - y
        theta = theta - ((alpha/m) * np.sum(error * x, axis=0))
    return theta[0]

def plot():
	global x,y,theta
	plt.scatter(x,y) #original points scatterplot
	y_pred = (theta[0]*x + theta[1])
	plt.plot(x, y_pred, 'r') #plot regression line
	plt.show()

theta = np.zeros((n,1)) # theta vector with 2 zeroes
theta = gradientDescent(X, y, theta, 0.001, 1000)