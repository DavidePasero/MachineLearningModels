import numpy as np
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin

class SGD_LinearRegression (BaseEstimator, RegressorMixin):
    
    # Best: n_iterations = 225, eta = 0.15, power_i = 0.346
    def __init__ (self, n_iterations = 225, eta = 0.15, power_i = 0.346):
        self.n_iterations = n_iterations
        self.eta = eta
        self.power_i = power_i
        self.weights = None
        self.bias = None


    # Calculates the decreasing learning rate
    def learning_rate (self, i):
        return self.eta / pow (i, self.power_i)


    def fit (self, X, y):
        # Gets the number of features
        n_features = X.shape[1]
        # Initialize coefficients with random values from a normal distribution
        self.weights = np.zeros (n_features)
        # Initialize bias with a random value from a uniform distribution
        self.bias = 0
        for _ in range (self.n_iterations):
            # Executes the SGD based on the min squared errors cost function
            for i, instance_label in enumerate (zip (X, y)):
                instance, label = instance_label
                # Calculate the prediction based on our current weights and bias
                prevision = self.bias + np.dot (instance, self.weights)
                # calculates learning rate for this iteration, i+1 to avoid division by zero
                alfa = self.learning_rate (i + 1)
                #print (alfa)
                # Updates the coefficients using the derivative of the squared errors cost function
                for j, _ in enumerate (self.weights):
                    self.weights [j] -= alfa * (prevision - label) * X [i, j]
                # Updates bias using the same function but without multiplying for the feature value (X_train [i, j])
                self.bias -= alfa * (prevision - label)
        return self
    
    
    def predict (self, X):
        result = np.dot (X, self.weights) + self.bias
        return result

    
    def score (self, X, y):
        return r2_score (y, self.predict (X))