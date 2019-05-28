import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

aData = np.genfromtxt('dataset.csv', delimiter=',')
X = aData[1:,0].reshape(-1,1)
y = aData[1:,-1].reshape(-1,1)
plt.scatter(X, y)
X_train, X_test, y_train, y_test = train_test_split(X,y)
regressor = LinearRegression() #create an instance of LinearRegression()
regressor.fit(X_train,y_train) #train the model
y_pred = regressor.predict(X_test)
plt.plot(X_test, y_pred, 'r') #plot line of best fit
plt.show()