import numpy as np
import math

class LogisticRegression:


    def __init__(self,learning_rate, n_iters, lambd=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.n_iters = n_iters
        self.lambd = lambd
    

    def _sigmoid(self,z):
        z = np.clip(z, -500,500)
        p = 1/ (1 + np.exp(-z))
        return p



    def compute_loss(self, y,p):
        loss = -np.mean(math.log(p)*y + math.log(1-p)*(1-y))
        return loss

    def validate_input_array(self,X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim < 2:
            X = X.reshape(-1,1)
        elif X.ndim >2:
            raise Exception('The input array is larger than 2 dimensions')
    
    def nomralize(self, X, method):
        if method == "MinMax":
            X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        elif method == "RobustScaler":
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            X_scaled = (X - np.mean(X, axis =0)) / (q3 - q1)
        elif method == "StandardSclaer":
            X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return X_scaled
    
    def cal_prob(self,X):
        X_b = np.c_(np.ones((X.shape[0],1)),X)
        z = np.dot(X_b, self.weights)
        p = self._sigmoid(z)
        return p
        
    def fit(self,X,y):
        X_b = np.c_(np.ones((X.shape[0],1)),X)
        X_b = self.nomralize(X_b)
        self.weights = np.zeros(X_b.shape[1])
        for i in range(self.n_iters):
            z = np.dot(X_b, self.weights)
            p = self._sigmoid(-z)
            gradient = X_b.T.dot(p-y) / len(y)
            reg_term = self.lambd * self.weights / len(y)
            reg_term[0] = 0
            gradient += reg_term
            self.weights -= self.learning_rate*gradient
        return self

    def preicit(self, X, threshold):
       return  self.cal_prob(X)>=threshold.astype('int')
        






    
