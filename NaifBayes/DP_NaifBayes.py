import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, RegressorMixin

class DP_NaifBayes (BaseEstimator, RegressorMixin):
    def __init__ (self):
        self.target_occurrences = {}
        self.feature_occurrences = None
        self.mapping_target = {}
        self.mapping_features = []


    # For the target classes and each features' classes it creates a mapping
    # between the classes and numbers that are unique for each class
    # of the feature
    def _create_mapping (self, X, y):
        # Maps the target classes
        mapping_target = {}
        for target in y:
            if (target not in mapping_target):
                mapping_target [target] = len (mapping_target)

        # List of mappings, for each feature, its classes mapping
        mapping_features = []
        for i, column in enumerate (X.T):
            # create a new mapping for the i-th feature
            mapping_features.append({})
            for instance in column:
                if instance not in mapping_features[i]:
                    mapping_features [i][instance] = len (mapping_features [i])

        return mapping_target, mapping_features


    def fit (self, X, y):
        # data.shape[1] = num of columns
        num_feature = X.shape[1] # remove the result class label
        # Get the number of different target classes
        num_target_classes = len (np.unique (y))
        #Number of different feature classes
        num_classes = [len (np.unique (col)) for col in X.T]
        # Max number of classes of the features (excluding the result class)
        max_num_classes = max (num_classes [1 :])
        # Gets a mapping beetwen the different classes of each feature with corresponding numbers
        self.mapping_target, self.mapping_features = self._create_mapping (X, y)

        # Get the target class occurrences
        for target in y:
            if target not in self.target_occurrences:
                self.target_occurrences [target] = 0
            self.target_occurrences [target] += 1

        # Gets the number of occurrences of each feature_class-result_class pair
        # Initialize occurrences array filled with 1s to apply the laplace correction
        # It will be a 3D array:
        # Y: different result classes
        # X: different classes of the same feature
        # Z: different feature
        self.feature_occurrences = np.full ((num_target_classes, max_num_classes, num_feature), 1)
        # For each feature (in each instance), calculates how many times it gives
        # a certain class.
        for i, row in enumerate (X):
            for j, feature_class in enumerate (row):
                # mapping_target [target_classes[i]] = mapping of the target's class
                # mapping_list [j][feature_class] = mapping of the class we're considering (of the j-th feature)
                # We're considering the j-th feature
                self.feature_occurrences [self.mapping_target [y[i]], self.mapping_features [j][feature_class], j] += 1

        return self
    

    def predict (self, X):
        result = []

        for t in X:
            prob_target_class = {}
            # For each target class 
            for target_key, target_mapping in self.mapping_target.items():
                prob = 1
                # prob (E|B) for each E multiplied together.
                # To avoid overflows, we divide by 10 every multiplication we perform.
                for i, feature in enumerate (t):
                    prob *= self.feature_occurrences [target_mapping, self.mapping_features [i][feature], i] / 10
                # times the probability of B
                prob *= self.target_occurrences [target_key] / 10

                # Add probability to the mapping
                prob_target_class [target_key] = prob

            result.append (max (prob_target_class, key=prob_target_class.get))

        return result
    

    def score (self, X, y):
        return accuracy_score (y, self.predict (X))