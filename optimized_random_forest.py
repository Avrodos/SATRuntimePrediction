# a lot of code from this script has been taken from:
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
import sys
from typing import Final

import numpy as np

# first argument should be path to feature file, second argument is path to label file
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score

PATH_FEATURES: Final[str] = sys.argv[1]
PATH_LABELS: Final[str] = sys.argv[2]


def load_df():
    # load features and labels
    feature_df = pd.read_csv(PATH_FEATURES)
    feature_df.set_index('hash', inplace=True)
    label_df = pd.read_csv(PATH_LABELS)
    label_df.set_index('hash', inplace=True)

    # remove nan's
    merged_df = label_df.join(feature_df, how='left')
    merged_df.dropna(inplace=True)
    # to drop timeouts and 0sec instances
    merged_df = merged_df[(merged_df.parity_two_label != 10000) & (merged_df.parity_two_label != 0)]
    # merged_df = merged_df[(merged_df.min_label != 0)]

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(merged_df.columns[0:len(label_df.columns)], axis=1)
    label_df = merged_df['log_parity_two_label']
    return feature_df, label_df.values.ravel()


# first argument should be path to feature file, second argument is path to label file
if __name__ == '__main__':
    # define random seed first
    np.random.seed(0)
    loaded_feature_df, loaded_label_df = load_df()

    # first we define the random grid
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=0, n_jobs=-1)  # Fit the random search model
    rf_random.fit(loaded_feature_df, loaded_label_df)

    # results of our search:
    print(rf_random.best_params_)

    # now compare random search results to base model:
    base_model = RandomForestRegressor(random_state=0)
    # cv_scores = cross_val_score(base_model, loaded_feature_df, loaded_label_df, cv=10, scoring='neg_root_mean_squared_error')
    cv_scores = cross_val_score(base_model, loaded_feature_df, loaded_label_df, cv=10)
    print("Base Model: %0.4f accuracy with a standard deviation of %0.4f" % (cv_scores.mean(), cv_scores.std()))

    # these values are based on random search results
    # best_random = RandomForestClassifier(random_state=0, n_estimators=400, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=60, bootstrap=True)
    # best_random = RandomForestRegressor(random_state=0, n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=90, bootstrap=True)
    best_random = RandomForestRegressor(random_state=0, n_estimators=400, min_samples_split=2, min_samples_leaf=1,
                                        max_features='sqrt', max_depth=None, bootstrap=True)
    # cv_scores = cross_val_score(best_random, loaded_feature_df, loaded_label_df, cv=10, scoring='neg_root_mean_squared_error')
    cv_scores = cross_val_score(best_random, loaded_feature_df, loaded_label_df, cv=10)
    print("Optimized Model: %0.4f accuracy with a standard deviation of %0.4f" % (cv_scores.mean(), cv_scores.std()))
    #
    # # TODO: We can further optimize using grid search near our current parameters and then train on the complete set in the end
