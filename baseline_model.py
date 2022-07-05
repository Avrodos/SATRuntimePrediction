import sys
from typing import Final

import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd

PATH_FEATURES: Final[str] = sys.argv[1]
PATH_LABELS: Final[str] = sys.argv[2]

def main():
    # define random seed first
    np.random.seed(0)
    # load features and labels
    feature_df = pd.read_csv(PATH_FEATURES)
    label_df = pd.read_csv(PATH_LABELS)
    # TODO: compare result to if I would scale beforehand:
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    regressor = RandomForestRegressor(random_state=0)
    cv = model_selection.KFold(n_splits=10)
    r2_scores = []
    mean_squared_errors = []
    for train_index, test_index in cv.split(feature_df):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = feature_df[train_index], feature_df[test_index]
        y_train, y_test = label_df[train_index], label_df[test_index]

        # For training, fit() is used
        regressor.fit(X_train, y_train)

        # Default metric is R2 for regression, which can be accessed by score()
        r2_scores.append(regressor.score(X_test, y_test))

        # For other metrics, we need the predictions of the model
        y_pred = regressor.predict(X_test)
        mean_squared_errors.append(regressor.mean_squared_error(y_test, y_pred))

# first argument should be path to feature file, second argument is path to label file
if __name__ == '__main__':
    main()