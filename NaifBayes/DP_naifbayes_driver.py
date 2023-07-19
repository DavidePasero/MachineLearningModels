import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numbers
import random
import DP_NaifBayes

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

# ! Driver code

def main ():
    NUM_BINS = 10
    iris_data = load_iris()
    target_classes = iris_data.target
    # Make all the numeric features to be class features by creating NUM_BINS bins
    features_classes = create_bins (iris_data.data, NUM_BINS)

    # Splits the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(features_classes, target_classes, test_size=0.2, random_state=random.randint(0, 100))

    model = DP_NaifBayes.DP_NaifBayes()
    model.fit (X_train, y_train)
    score = model.score (X_test, y_test)

    print ("Accuracy score is", score)

if __name__ == '__main__':
    main ()

