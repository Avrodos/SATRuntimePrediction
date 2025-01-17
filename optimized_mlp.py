# a lot of code from this script has been taken from:
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
import os
import sys
from typing import Final

import numpy as np
# first argument should be path to feature file, second argument is path to label file
import pandas as pd
from numpy.random import uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PATH_FEATURES: Final[str] = sys.argv[1]
PATH_LABELS: Final[str] = sys.argv[2]


# inspired from:
# https://stackoverflow.com/questions/52032019/sklearn-mlp-classifier-hidden-layers-optimization-randomizedsearchcv
class RandIntMatrix(object):
    def __init__(self, low, high, shape=(1)):
        self.low = low
        self.high = high
        self.shape = shape

    def rvs(self, random_state=42):
        np.random.seed(random_state)
        return np.random.randint(self.low, self.high, self.shape)


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
    # to only drop 0sec instances
    merged_df = merged_df[(merged_df.log10_parity_two_label != np.log10(0))]

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(merged_df.columns[0:len(label_df.columns)], axis=1)
    label_df = merged_df['log10_parity_two_label']
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
    # model = MLPClassifier()
    # model = MLPRegressor(random_state=42)
    model = MLPRegressor(random_state=42, solver='adam', max_iter=1389, learning_rate='constant',
                         hidden_layer_sizes=(320,), alpha=0.01)

    # build pipeline
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    # get model specific param names:
    print(pipeline.get_params().keys())

    # first we define the random grid
    # number of hidden layers
    num_hidden_layer = [int(x) for x in np.linspace(start=1, stop=3, num=10)]
    # alpha values
    alphas = [0.0001, 0.001, 0.01, 0.005, 0.05]
    # number of neurons in the hidden layer
    min_hidden_layer_size = 50
    max_hidden_layer_size = 600
    hidden_layer_configs = []
    for x in num_hidden_layer:
        hidden_layer_size = np.random.randint(min_hidden_layer_size, max_hidden_layer_size)
        current_hidden_layer_config = tuple(np.repeat(hidden_layer_size, x))
        hidden_layer_configs.append(current_hidden_layer_config)
    # solver for weight optimization
    solvers = ['lbfgs', 'adam']
    # number of max iterations
    max_iters = [int(x) for x in np.linspace(start=100, stop=10000, num=500)]
    # replace with model specific values!
    random_grid = {'model__hidden_layer_sizes': hidden_layer_configs,
                   'model__solver': solvers,
                   'model__alpha': alphas,
                   'model__learning_rate': ['constant', 'adaptive'],
                   'model__max_iter': max_iters
                   }

    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using k-fold cross validation,
    # search across n_iter different combinations, and use four cores

    random_model = RandomizedSearchCV(estimator=pipeline, param_distributions=random_grid, n_iter=200, cv=3, verbose=2,
                                      random_state=42, n_jobs=4)  # Fit the random search model
    random_model.fit(X_train, y_train)

    # results of our search:
    print(random_model.best_params_)

    # for more in depth inspection
    scores_df = pd.DataFrame(random_model.cv_results_)
    scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
    print(scores_df)
    scores_df.to_csv(os.getcwd() + '/data/measured_data/mlp_log10par2_hyperparam_optimization_results.csv')

    # # test the hyper params
    pipeline.fit(X_train, y_train)
    print("Accuracy of the model with all features: %0.4f" % pipeline.score(X_test, y_test))
