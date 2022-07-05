import sys
from typing import Final

import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
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
    cv_scores = cross_val_score(regressor, feature_df, label_df, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_scores.mean(), cv_scores.std()))


# first argument should be path to feature file, second argument is path to label file
if __name__ == '__main__':
    main()