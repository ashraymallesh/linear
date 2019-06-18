import numpy as np
import matplotlib.pyplot as plt

aData = np.genfromtxt('ex1data2.txt', delimiter=',', skip_header=1)
m,n = aData.shape
X = aData[:,0:2]
y = aData[:,-1].reshape(-1,1)
mean = np.mean(X, axis=0); std = np.std(X, axis=0)
X = (X - mean)/std #column wise feature normalization
X = np.c_[np.ones(m), X]