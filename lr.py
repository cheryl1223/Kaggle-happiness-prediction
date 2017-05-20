import numpy as np
import scipy.optimize as opt  
import math
from sklearn import preprocessing

class LogisticRegression:
    def __init__(self):
        """ your code here """
        self.coef_ = []
        self.iter_ = 0

    def sigmoid(self, scores):
        return 1 / (1 + np.exp(-scores))
    def log_likelihood(self, features, target, weights):
        scores = np.dot(features, weights)
        ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
        return ll
    def logistic_regression(self,features, target, num_steps, learning_rate, add_intercept = False):
        if add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
            
        weights = np.zeros(features.shape[1])
        
        for step in range(num_steps):
            self.iter_ = self.iter_+1
            scores = np.dot(features, weights)
            predictions = self.sigmoid(scores)
            previous = self.log_likelihood(features, target, weights)
            # Update weights with gradient
            output_error_signal = target - predictions
            gradient = np.dot(features.T, output_error_signal)
            weights += learning_rate * gradient
            new =  self.log_likelihood(features, target, weights)
            if new - previous < 0.00001:

                break
        print("Iteration number: ", self.iter_)            
        return weights
    def fit(self, X, y):
        weights = self.logistic_regression(X, y,
                     num_steps = 50, learning_rate = 5e-5, add_intercept = True)

        self.coef_ = weights[1:]
        return self


    def predict(self, X):

        probability = self.sigmoid(np.dot(X,self.coef_))
        y = [1 if x >= 0.5 else 0 for x in probability]

        return y

    def score(self, X, y):
        y_pred = self.predict(X)
        correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(y_pred, y)]
        accuracy = (np.sum(correct)) / len(correct)
        return accuracy
