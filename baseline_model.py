import sys
from typing import Final

import numpy as np
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend

PATH_FEATURES: Final[str] = sys.argv[1]
PATH_LABELS: Final[str] = sys.argv[2]

def load_df():
    # load features and labels
    feature_df = pd.read_csv(PATH_FEATURES)
    feature_df.set_index('hash', inplace=True)
    label_df = pd.read_csv(PATH_LABELS)
    label_df.set_index('hash', inplace=True)
    return feature_df, label_df


def regression_model_preprocessing(loaded_feature_df, loaded_label_df):
    feature_df = loaded_feature_df
    label_df = loaded_label_df[['min_label', 'log_min_label']]

    # # some rows have nan values in 'DegreeAssortativity'
    # feature_df['DegreeAssortativity'] = feature_df['DegreeAssortativity'].fillna(0)

    # let's try dropping timeouts (log(5000)) and log(0) instances
    merged_df = label_df.join(feature_df, how='left')
    merged_df.dropna(inplace=True)
    merged_df = merged_df[(merged_df.min_label != 5000) & (merged_df.min_label != 0)]

    # use only one label
    label_df = label_df['min_label']

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(merged_df.columns[[0, 1, 2]], axis=1)
    label_df = merged_df['log_min_label']

    # scale features beforehand:
    sc = StandardScaler()
    feature_df = sc.fit_transform(feature_df)

    # dimensionality reduction using a PCA:
    pca = PCA(n_components=40)
    principal_components = pca.fit_transform(feature_df)
    feature_df = pd.DataFrame(data=principal_components)

    return feature_df, label_df


def regression_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df):
    # train and evaluate Random Forest
    regressor = RandomForestRegressor(random_state=0, verbose=1)
    cv_scores = cross_val_score(regressor, preprocessed_feature_df, preprocessed_label_df.values.ravel(), cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_scores.mean(), cv_scores.std()))


def classifier_model_preprocessing(loaded_feature_df, loaded_label_df):
    # currently no real preprocessing needed
    merged_df = loaded_label_df.join(loaded_feature_df, how='left')
    merged_df.dropna(inplace=True)

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(merged_df.columns[[0, 1, 2]], axis=1)
    label_df = merged_df['three_categories_label']

    # # scale features beforehand:
    # sc = StandardScaler()
    # feature_df = sc.fit_transform(feature_df)
    #
    # # dimensionality reduction using a PCA:
    # pca = PCA(n_components=30)
    # principal_components = pca.fit_transform(feature_df)
    # feature_df = pd.DataFrame(data=principal_components)

    return feature_df, label_df


def classifier_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df):
    # train and evaluate Random Forest
    classifier = RandomForestClassifier(random_state=0, verbose=1)
    cv_scores = cross_val_score(classifier, preprocessed_feature_df, preprocessed_label_df.values.ravel(), cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_scores.mean(), cv_scores.std()))


def pipeline():
    loaded_feature_df, loaded_label_df = load_df()

    # # if we want to use a regression model:
    # preprocessed_feature_df, preprocessed_label_df = regression_model_preprocessing(loaded_feature_df, loaded_label_df)
    # with parallel_backend('threading', n_jobs=4):
    #     regression_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df)

    # if we want to use a classifier model:
    preprocessed_feature_df, preprocessed_label_df = classifier_model_preprocessing(loaded_feature_df, loaded_label_df)
    with parallel_backend('threading', n_jobs=4):
        classifier_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df)


# first argument should be path to feature file, second argument is path to label file
if __name__ == '__main__':
    # define random seed first
    np.random.seed(0)
    pipeline()