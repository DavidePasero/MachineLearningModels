import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
import DP_LinearRegression as DP_LR
from sklearn.model_selection import RandomizedSearchCV


def main ():
    data = load_diabetes ()
    feature = data.data
    target = data.target

    # Splits the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2)

    dp_lr = DP_LR.SGD_LinearRegression ()
    skit_sdgr = SGDRegressor (loss = 'squared_error', max_iter = 3000)

    dp_lr.fit (X_train, y_train)
    skit_sdgr.fit (X_train, y_train)
    print ("My SGD linear regressor:", dp_lr.score(X_test, y_test))
    print ("Sklearn SGD regressor:", skit_sdgr.score(X_test, y_test))


# Calculates hyperparameters through a randomized search, could also be done with a grid search
def calc_hyperparameters (model, X_train, y_train):
    hyperparam_grid = {
        # Arbitrary values from previous randomized searches
        'eta': np.linspace (0.15, 0.5, 50),
        'power_i': np.linspace (0.098, 0.427, 50)
    }

    # Create a RandomizedSearchCV object
    random_search = RandomizedSearchCV (
        estimator = model,
        param_distributions = hyperparam_grid,
        n_iter = 75,
        cv = 3,
        scoring = 'r2',
        n_jobs=4
    )

    # Perform the random search
    random_search.fit(X_train, y_train)
    print ("Best params:", random_search.best_params_)
    print ("Best score:", random_search.best_score_)


if __name__ == '__main__':
    main ()