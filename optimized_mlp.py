# a lot of code from this script has been taken from:
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
import sys
from typing import Final

import numpy as np
# first argument should be path to feature file, second argument is path to label file
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PATH_FEATURES: Final[str] = sys.argv[1]
PATH_LABELS: Final[str] = sys.argv[2]


# taken from: https://stackoverflow.com/questions/52032019/sklearn-mlp-classifier-hidden-layers-optimization-randomizedsearchcv
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
    # merged_df = merged_df[(merged_df.parity_two_label != 10000) & (merged_df.parity_two_label != 0)]
    # merged_df = merged_df[(merged_df.min_label != 0)]

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(merged_df.columns[0:len(label_df.columns)], axis=1)
    label_df = merged_df['3-means_label']
    return feature_df, label_df.values.ravel()


# first argument should be path to feature file, second argument is path to label file
if __name__ == '__main__':
    # define random seed first
    np.random.seed(42)
    loaded_feature_df, loaded_label_df = load_df()

    # First create the base model to tune
    model = MLPClassifier()
    # build pipeline
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    # get model specific param names:
    print(pipeline.get_params().keys())

    # first we define the random grid
    # number of hidden layers
    num_hidden_layer = [int(x) for x in np.linspace(start=1, stop=5, num=10)]
    # number of neurons in the hidden layer
    min_hidden_layer_size = 10
    max_hidden_layer_size = 1000
    hidden_layer_configs = []
    for x in num_hidden_layer:
        hidden_layer_size = np.random.randint(min_hidden_layer_size, max_hidden_layer_size)
        current_hidden_layer_config = tuple(np.repeat(hidden_layer_size, x))
        hidden_layer_configs.append(current_hidden_layer_config)
    # solver for weight optimization
    solvers = ['lbfgs', 'adam']
    # init value
    learning_rate_inits = [float(x) for x in np.linspace(start=0.0001, stop=0.9, num=500)]
    # number of max iterations
    max_iters = [int(x) for x in np.linspace(start=100, stop=10000, num=500)]
    # replace with model specific values!
    random_grid = {'model__hidden_layer_sizes': hidden_layer_configs,
                   'model__solver': solvers,
                   'model__learning_rate_init': learning_rate_inits,
                   'model__max_iter': max_iters
                   }

    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using k-fold cross validation,
    # search across n_iter different combinations, and use all available cores

    random_model = RandomizedSearchCV(estimator=pipeline, param_distributions=random_grid, n_iter=200, cv=3, verbose=2,
                                      random_state=42, n_jobs=4)  # Fit the random search model
    random_model.fit(loaded_feature_df, loaded_label_df)

    # results of our search:
    print(random_model.best_params_)

    # for more in depth inspection
    scores_df = pd.DataFrame(random_model.cv_results_)
    scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
    print(scores_df)
    scores_df.to_csv('hyperparam_optimization_results.csv')

    # # now compare random search results to base model:
    # base_model = MLPClassifier(random_state=42)
    # # cv_scores = cross_val_score(base_model, loaded_feature_df, loaded_label_df, cv=10, scoring='neg_root_mean_squared_error')
    # cv_scores = cross_val_score(base_model, loaded_feature_df, loaded_label_df, cv=10)
    # print("Base Model: %0.4f accuracy with a standard deviation of %0.4f" % (cv_scores.mean(), cv_scores.std()))
    #
    # # these values are based on random search results
    # best_random = MLPClassifier(random_state=42)
    # # cv_scores = cross_val_score(best_random, loaded_feature_df, loaded_label_df, cv=10, scoring='neg_root_mean_squared_error')
    # cv_scores = cross_val_score(best_random, loaded_feature_df, loaded_label_df, cv=10)
    # print("Optimized Model: %0.4f accuracy with a standard deviation of %0.4f" % (cv_scores.mean(), cv_scores.std()))
    # #
    # # # TODO: We can further optimize using grid search near our current parameters and then train on the complete set in the end
