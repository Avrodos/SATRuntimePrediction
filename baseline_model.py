import os
import sys
from typing import Final

import numpy as np
from sklearn import model_selection, metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
    current_label = ['log10_parity_two_label']
    # # some rows have nan values in 'DegreeAssortativity'
    # feature_df['DegreeAssortativity'] = feature_df['DegreeAssortativity'].fillna(0)

    # let's try dropping nan's, timeouts (log(5000)) and log(0) instances
    merged_df = loaded_label_df.join(loaded_feature_df, how='left')
    merged_df.dropna(inplace=True)
    # merged_df = merged_df[(merged_df.log_parity_two_label != np.log(10000)) & (merged_df.log_parity_two_label != np.log(0))]
    merged_df = merged_df[(merged_df.log10_parity_two_label != np.log10(0))]
    # # use only one label
    # label_df = label_df[current_label]

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(merged_df.columns[0:len(loaded_label_df.columns)], axis=1)
    label_df = merged_df[current_label]
    #
    # scale features beforehand:
    sc = StandardScaler()
    feature_df = sc.fit_transform(feature_df)
    #
    # # dimensionality reduction using a PCA:
    # pca = PCA(n_components=40)
    # principal_components = pca.fit_transform(feature_df)
    # feature_df = pd.DataFrame(data=principal_components)

    return feature_df, label_df


def regression_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df):
    # train and evaluate Random Forest
    # model = RandomForestRegressor(random_state=0, verbose=1)
    model = MLPRegressor(random_state=42, max_iter=800)
    cv_scores = cross_val_score(model, preprocessed_feature_df, preprocessed_label_df.values.ravel(), cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_scores.mean(), cv_scores.std()))


def classifier_model_preprocessing(loaded_feature_df, loaded_label_df):
    # currently no real preprocessing needed
    merged_df = loaded_label_df.join(loaded_feature_df, how='left')
    merged_df.dropna(inplace=True)

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(merged_df.columns[0:len(loaded_label_df.columns)], axis=1)
    label_df = merged_df['all_families_threshold_10']

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
    # # train Multi-layer Perceptron - don't forget to scale features first
    # classifier = MLPClassifier(random_state=0, max_iter=900)


    # train Random Forest
    classifier = RandomForestClassifier(random_state=0, verbose=1)

    # evaluate using cross validation
    cv_scores = cross_val_score(classifier, preprocessed_feature_df, preprocessed_label_df.values.ravel(), cv=10)
    print("%0.4f accuracy with a standard deviation of %0.4f" % (cv_scores.mean(), cv_scores.std()))

    # # without cv
    # X_train, X_test, y_train, y_test = train_test_split(preprocessed_feature_df, preprocessed_label_df.values.ravel(), test_size = 0.33, random_state = 0)
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
    #
    # # let's have a look at the wrong predictions
    # indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    # wrong_predictions = preprocessed_feature_df.iloc[indices, :]
    # family_df = pd.read_csv(os.getcwd() + "/data/meta_data/hash_to_family_mapping.csv")
    # family_df.set_index('hash', inplace=True)
    # merged_df = wrong_predictions.join(family_df, how='left')
    # print(merged_df['family'].value_counts())
    # print(merged_df['family'].value_counts() / merged_df.shape[0])
    #
    # # to see the importance of the trained classifier for all features
    # feature_imp = pd.Series(classifier.feature_importances_, index=preprocessed_feature_df.columns).sort_values(ascending=False)
    # print(feature_imp)

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