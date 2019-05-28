import numpy as np
import matplotlib.pyplot as plt

aData = np.genfromtxt('dataset.csv', delimiter=',')
x = aData[1:,0].reshape(-1,1)
y = aData[1:,-1].reshape(-1,1)
m = x.size # number of training examples
n = 2 # number of features
ones = np.ones((m, 1)) # m-dim vector filled with 1s
X = np.concatenate((ones, x), 1) # create design matrix X by adding ones as left col (x_0 = 1)
theta = np.zeros((n,1)) # theta vector with 2 zeroes

#compute cost using vectorized numpy computation instead of using for-loops
error = (X @ theta) - y # mxn @ nx1 -> mx1 (matrix multiplication)
cost = np.sum(error**2)/(2*m)

def gradientDescent(X, y, theta, alpha, iters):
    for i in range(iters):
        error = (X @ theta) - y # mxn @ nx1 -> mx1 (matrix multiplication)/
        theta = theta - ((alpha/m) * np.sum(error * X, axis=0))
    return theta[0]

print('Initial cost with theta = [0;0] is {0}'.format(cost))

theta = gradientDescent(X, y, theta, 0.001, 1000)
plt.scatter(x,y) #original points scatterplot
y_pred = (theta[0]*x + theta[1])
plt.plot(x, y_pred, 'r') #plot regression line
plt.show()