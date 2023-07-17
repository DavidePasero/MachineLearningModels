import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numbers
import random

# ! Data Preprocessing
# Converts every list of numbers in data into a k-amount of bins so that we have a class value
def create_bins (features_classes, k):
    for j, column in enumerate(features_classes.T):
        # If the feature is numeric we convert it to bins
        if (isinstance (column [0], numbers.Number)):
            _, bin_edges = np.histogram (features_classes, bins = k)
            # Every number will get, as its value, the index of the bin it belongs to
            for i, instance in enumerate (column):
                features_classes [i][j] = np.digitize (instance, bin_edges)
    return features_classes

# ! ML Model
# For the target classes and each features' classes it creates a mapping
# between the classes and numbers that are unique for each class
# of the feature
def create_mapping (target_classes, features_classes):
    # Maps the target classes
    mapping_target = {}
    for x in target_classes:
        if (x not in mapping_target):
            mapping_target [x] = len (mapping_target)

    # List of mappings, for each feature, its classes mapping
    mapping_features = []
    for i, column in enumerate (features_classes.T):
        # create a new mapping for the i-th feature
        mapping_features.append({})
        for instance in column:
            if instance not in mapping_features[i]:
                mapping_features [i][instance] = len (mapping_features [i])

    return mapping_target, mapping_features
    

def calc_occurrences (target_classes, features_classes):
    # data.shape[1] = num of columns
    num_feature = features_classes.shape[1] # remove the result class label
    # Get the number of different target classes
    num_target_classes = len (np.unique (target_classes))
    #Number of different feature classes
    num_classes = [len (np.unique (col)) for col in features_classes.T]
    # Max number of classes of the features (excluding the result class)
    max_num_classes = max (num_classes [1 :])
    # Gets a mapping beetwen the different classes of each feature with corresponding numbers
    mapping_target, mapping_features = create_mapping (target_classes, features_classes)

    # Get the target class occurrences
    target_occurrences = {}
    for x in target_classes:
        if x not in target_occurrences:
            target_occurrences [x] = 0
        target_occurrences [x] += 1

    # Gets the number of occurrences of each feature_class-result_class pair
    # Initialize occurrences array filled with 1s to apply the laplace correction
    # It will be a 3D array:
    # Y: different result classes
    # X: different classes of the same feature
    # Z: different feature
    feature_occurrences = np.full ((num_target_classes, max_num_classes, num_feature), 1)
    # For each feature (in each instance), calculates how many times it gives
    # a certain class.
    for i, row in enumerate (features_classes):
        for j, feature_class in enumerate (row):
            # mapping_target [target_classes[i]] = mapping of the target's class
            # mapping_list [j][feature_class] = mapping of the class we're considering (of the j-th feature)
            # We're considering the j-th feature
            feature_occurrences [mapping_target [target_classes[i]], mapping_features [j][feature_class], j] += 1

    # I don't really need the probabilities, occurrences have enough information.
    return target_occurrences, feature_occurrences, mapping_target, mapping_features
    

# To test the model we use a simplified version of the bayes theorem 
# and we consider all the features to be independent from one another.
# Basically, we don't care about dividing for P(E), the denominator
# is the same throughout all the probabilities of the targets
def test_bayes_theorem (target_occurrences, feature_occurrences, mapping_target, mapping_feature, test):
    prob_target_class = {}
    # For each target class 
    for target_key, target_mapping in mapping_target.items():
        prob = 1
        # prob (E|B) for each E multiplied together.
        # To avoid overflows, we divide by 10 every multiplication we perform.
        for i, feature in enumerate (test):
            prob *= feature_occurrences [target_mapping, mapping_feature [i][feature], i] / 10
        # times the probability of B
        prob *= target_occurrences [target_key] / 10

        # Add probability to the mapping
        prob_target_class [target_key] = prob

    return prob_target_class

# ! Driver code

def main ():
    NUM_BINS = 10
    iris_data = load_iris()
    target_classes = iris_data.target
    # Make all the numeric features to be class features by creating NUM_BINS bins
    features_classes = create_bins (iris_data.data, NUM_BINS)

    # Splits the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(features_classes, target_classes, test_size=0.2, random_state=random.randint(0, 100))

    # Calculates all the model's data structures
    target_occurrences, feature_occurrences, mapping_target, mapping_features = calc_occurrences (y_train, X_train)
    # Test the model on the first test case 
    result = test_bayes_theorem (target_occurrences, feature_occurrences, mapping_target, mapping_features, X_test[0])
    print ("Result: ", result)
    print ("Solution: ", y_test[0])
    pass

if __name__ == '__main__':
    main ()

