import numpy as np
import scipy.optimize as opt  
import math
from sklearn import preprocessing

class LogisticRegression:
    def __init__(self):
        """ your code here """
        self.coef_ = []

    def sigmoid(self, M):
        g = np.zeros(M.shape[0])
        for i in range(M.shape[0]):
            g[i] = 1.0 / (1.0 + math.exp(-M[i]))
        return g

    def cost(self, theta, X, y):
        h = self.sigmoid(np.dot(X,theta))
        first = np.multiply(-y, np.log(h))
        second = np.multiply((1 - y), np.log(1 - h))
        return np.sum(first - second) / (X.shape[0])

    def gradient(self, theta, X, y):
        m = X.shape[0] # the num of training samples

        #print(np.dot(X,self.coef_)[:10])

        gradient = np.zeros(X.shape[1])

        h = self.sigmoid(np.dot(X,theta))
        h = np.reshape(h, (len(h), 1))
        error = np.transpose(h-y)

        gradient = (np.dot(error,X)) #/m

        return gradient

    def fit(self, X, y):
        """ your code here """
        X = np.array(X, dtype = float)
        y = np.array(y)
        y = np.reshape(y, (len(y), 1))

        #X = np.insert(X, 0, 1, axis = 1)

        scaler = preprocessing.MinMaxScaler()
        X = scaler.fit_transform(X)

        self.coef_ = np.zeros((X.shape[1],1))

        result = opt.fmin_tnc(func=self.cost, x0=self.coef_, 
            fprime=self.gradient, args=(X, y), messages=0)

        self.coef_ = np.reshape(result[0], (len(result[0]),1))

        return self


    def predict(self, X):
        """ your code here """
        X = np.array(X, dtype = float)
        #X = np.insert(X, 0, 1, axis = 1)

        scaler = preprocessing.MinMaxScaler()
        X = scaler.fit_transform(X)

        probability = self.sigmoid(np.dot(X,self.coef_))
        y = [1 if x >= 0.5 else 0 for x in probability]

        return y

    def score(self, X, y):
        y_pred = self.predict(X)
        correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(y_pred, y)]
        accuracy = (np.sum(correct)) / len(correct)
        return accuracy
