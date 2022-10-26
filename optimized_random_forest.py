# a lot of code from this script has been taken from:
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
import os
import sys
from typing import Final

import numpy as np
# first argument should be path to feature file, second argument is path to label file
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    # # to drop timeouts and 0sec instances
    # merged_df = merged_df[(merged_df.parity_two_label != 10000) & (merged_df.parity_two_label != 0)]
    # to only drop 0sec instances
    # merged_df = merged_df[(merged_df.log10_parity_two_label != np.log10(0))]

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(merged_df.columns[0:len(label_df.columns)], axis=1)
    label_df = merged_df['3-means_label']
    return feature_df, label_df.values.ravel()


# first argument should be path to feature file, second argument is path to label file
if __name__ == '__main__':
    # define random seed first
    np.random.seed(42)
    loaded_feature_df, loaded_label_df = load_df()

    # first split into train and test
    X_train, X_test, y_train, y_test = train_test_split(loaded_feature_df, loaded_label_df,
                                                        random_state=42)

    # First decide on the model to tune
    # model = RandomForestClassifier(random_state=42)
    # model = RandomForestRegressor(random_state=42)
    model = RandomForestClassifier(random_state=42, n_estimators=1600, min_samples_split=5, min_samples_leaf=2,
                                   max_features='auto', max_depth=100)
    # build pipeline
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', model)
    ])

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
    random_grid = {'model__n_estimators': n_estimators,
                   'model__max_features': max_features,
                   'model__max_depth': max_depth,
                   'model__min_samples_split': min_samples_split,
                   'model__'
                   'min_samples_leaf': min_samples_leaf}

    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using k-fold cross validation,
    # search across n_iter different combinations, and use four cores

    random_model = RandomizedSearchCV(estimator=pipeline, param_distributions=random_grid, n_iter=200, cv=3, verbose=2,
                                      random_state=42, n_jobs=4)
    # Fit the random search model
    random_model.fit(X_train, y_train)

    # results of our search:
    print(random_model.best_params_)

    # for more in depth inspection
    scores_df = pd.DataFrame(random_model.cv_results_)
    scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
    print(scores_df)
    scores_df.to_csv(os.getcwd() + '/data/measured_data/rf_par2_hyperparam_optimization_results.csv')

    # test the hyper params
    pipeline.fit(X_train, y_train)
    print("Accuracy of the model with all features: %0.4f" % pipeline.score(X_test, y_test))
