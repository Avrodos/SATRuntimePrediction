# first argument should be path to feature file, second argument is path to label file
import os
import sys
from typing import Final

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PATH_FEATURES: Final[str] = sys.argv[1]
PATH_LABELS: Final[str] = sys.argv[2]
FAMILY_COUNT_THRESHOLD: Final[int] = 10
CURRENT_FAMILY_LABEL: Final = ['all_families_threshold_10']
CURRENT_RUNTIME_LABEL: Final = ['parity_two_label']
le = LabelEncoder()


def string_to_numerical_feature_encoder(feature_df):
    columns_to_encode = list(feature_df.select_dtypes(include=['category', 'object']))
    for feature in columns_to_encode:
        try:
            feature_df[feature] = le.fit_transform(feature_df[feature])
        except:
            print('Error encoding ' + feature)
    return feature_df


def add_family_as_feature(preprocessed_feature_df):
    # We can add the family to the features
    family_df = pd.read_csv(os.getcwd() + "/data/meta_data/hash_to_family_mapping.csv")
    family_df.set_index('hash', inplace=True)
    preprocessed_feature_df = preprocessed_feature_df.join(family_df, how='left')
    return preprocessed_feature_df


def load_df():
    # load features and labels
    feature_df = pd.read_csv(PATH_FEATURES, index_col=0)
    feature_df = add_family_as_feature(feature_df)

    label_df = pd.read_csv(PATH_LABELS, index_col=0)

    return feature_df, label_df


def family_classifier_model_preprocessing(loaded_feature_df, loaded_label_df):
    # to ensure we have a label on each feature
    merged_df = loaded_label_df[CURRENT_FAMILY_LABEL].join(loaded_feature_df, how='left')
    # we need to drop nan instances
    merged_df.dropna(inplace=True)

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(labels=CURRENT_FAMILY_LABEL, axis=1)
    label_df = merged_df[CURRENT_FAMILY_LABEL]

    return feature_df, label_df.values.ravel()


def pipeline():
    loaded_feature_df, loaded_label_df = load_df()
    family_feature_df, family_label_df = family_classifier_model_preprocessing(loaded_feature_df, loaded_label_df)
    # replace a family name with an arbitrary, but consistent, number
    family_feature_df = string_to_numerical_feature_encoder(family_feature_df)
    # we will join the labels to the features for slicing purposes, but we will drop them before using them in a model
    family_feature_df = family_feature_df.join(loaded_label_df[CURRENT_RUNTIME_LABEL], how='left')
    family_feature_df.dropna(inplace=True)
    family_label_df = family_feature_df['family']
    # first split into train and test
    X_train, X_test, y_train, y_test = train_test_split(family_feature_df, family_label_df, random_state=42)
    family_classifier = RandomForestClassifier(random_state=42, verbose=1)
    base_runtime_regressor = RandomForestRegressor(random_state=42)
    # drop all labels
    X_train_base_features = X_train.drop(labels=[CURRENT_RUNTIME_LABEL[0], 'family'], axis=1)
    X_test_base_features = X_test.drop(labels=[CURRENT_RUNTIME_LABEL[0], 'family'], axis=1)
    y_train_family = X_train['family']
    y_train_runtime = X_train[CURRENT_RUNTIME_LABEL[0]]
    family_classifier.fit(X_train_base_features, y_train_family)
    base_runtime_regressor.fit(X_train_base_features, y_train_runtime)

    # now we also want a random forest model for each family
    # first we need to splice the df
    df_sliced_dict_train = {}
    for family in X_train['family'].unique():
        # each entry contains base features + family_label + runtime_label
        df_sliced_dict_train[family] = X_train[X_train['family'] == family]
        df_sliced_dict_train[family] = df_sliced_dict_train[family].dropna()

    # this dict will map the family to a regressor trained only on one family
    family_specific_models = {}
    with parallel_backend('threading', n_jobs=4):
        for family in df_sliced_dict_train.keys():
            if len(df_sliced_dict_train[family] > FAMILY_COUNT_THRESHOLD):
                regressor = RandomForestRegressor(random_state=42, verbose=1)
                runtime_X_train = df_sliced_dict_train[family].drop(labels=[CURRENT_RUNTIME_LABEL[0], 'family'], axis=1)
                runtime_y_train = df_sliced_dict_train[family][CURRENT_RUNTIME_LABEL[0]]
                regressor.fit(runtime_X_train, runtime_y_train)
                family_specific_models[family] = regressor
            else:
                family_specific_models[family] = base_runtime_regressor

    # now we have a hierarchical model:
    # first let 'family_classifier' predict the family
    # then let the family specific regressor predict the runtime
    runtime_pred = []
    y_pred = family_classifier.predict(X_test_base_features)
    for idx, predicted_family in enumerate(y_pred):
        if predicted_family in family_specific_models.keys():
            current_row_df = X_test_base_features.iloc[[idx]]
            curr_pred = family_specific_models[predicted_family].predict(current_row_df)
            runtime_pred.append(curr_pred[0])
    y_test_true = X_test[CURRENT_RUNTIME_LABEL]
    print("Accuracy of the hierarchical model with all features: %0.4f" % metrics.r2_score(y_test_true, runtime_pred))



if __name__ == '__main__':
    # define random seed first
    np.random.seed(0)
    pipeline()