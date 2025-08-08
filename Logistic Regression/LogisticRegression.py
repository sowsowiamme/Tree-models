import numpy as np
import math

class LogisticRegression:


    def __init__(self,learning_rate, n_iters):
        self.learning_rate = learning_rate
        self.weights = None
        self.n_iters = n_iters
    

    def _sigmoid(self,z):
        p = 1/ (1 + np.exp(-z))
        return p

    def fit(self,X,y):
        X_b = np.c_(np.ones((X.shape[0],1)),X)
        self.weights = np.zeros(X_b.shape[1])
        for i in range(self.n_iters):
            z = np.dot(X_b, self.weights)
            p = self._sigmoid(-z)
            gradient = X_b.T.dot(p-y) / len(y)
            self.weights -= self.learning_rate*gradient
        return self
    