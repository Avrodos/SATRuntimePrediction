# first argument should be path to feature file, second argument is path to label file
import sys
from typing import Final

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PATH_FEATURES: Final[str] = sys.argv[1]
PATH_LABELS: Final[str] = sys.argv[2]
FAMILY_COUNT_THRESHOLD: Final[int] = 10
CURRENT_FAMILY_LABEL: Final = ['all_families_threshold_10']
CURRENT_RUNTIME_LABEL: Final = ['3-means_label']
le = LabelEncoder()


def string_to_numerical_feature_encoder(feature_df):
    columns_to_encode = list(feature_df.select_dtypes(include=['category', 'object']))
    for feature in columns_to_encode:
        try:
            feature_df[feature] = le.fit_transform(feature_df[feature])
        except:
            print('Error encoding ' + feature)
    return feature_df


def load_df():
    # load features and labels
    feature_df = pd.read_csv(PATH_FEATURES, index_col=0)
    label_df = pd.read_csv(PATH_LABELS, index_col=0)

    return feature_df, label_df


def family_classifier_model_preprocessing(loaded_feature_df, loaded_label_df):
    # to ensure we have a label on each feature and to add the family as a feature
    merged_df = loaded_label_df[CURRENT_FAMILY_LABEL].join(loaded_feature_df, how='left')
    # we need to drop nan instances
    merged_df.dropna(inplace=True)

    # replace a family name with an arbitrary, but consistent, number
    family_feature_df = string_to_numerical_feature_encoder(merged_df)
    # we will join the runtime_labels to the features as well for slicing purposes,
    # but we will drop them before using them in a model
    extended_feature_df = family_feature_df.join(loaded_label_df[CURRENT_RUNTIME_LABEL], how='left')
    extended_feature_df.dropna(inplace=True)

    return extended_feature_df, extended_feature_df[CURRENT_FAMILY_LABEL]


def pipeline():
    loaded_feature_df, loaded_label_df = load_df()
    extended_feature_df, family_label_df = family_classifier_model_preprocessing(loaded_feature_df, loaded_label_df)

    # first split into train and test
    X_train, X_test, y_train, y_test = train_test_split(extended_feature_df, family_label_df, random_state=42)
    family_classifier = RandomForestClassifier(random_state=42, verbose=1)
    # base_runtime_model = RandomForestRegressor(random_state=42)
    base_runtime_model = RandomForestClassifier(random_state=42)
    # drop all labels  before using them for training
    X_train_base_features = X_train.drop(labels=[CURRENT_RUNTIME_LABEL[0], CURRENT_FAMILY_LABEL[0]], axis=1)
    X_test_base_features = X_test.drop(labels=[CURRENT_RUNTIME_LABEL[0], CURRENT_FAMILY_LABEL[0]], axis=1)
    y_train_family = X_train[CURRENT_FAMILY_LABEL[0]]
    y_train_runtime = X_train[CURRENT_RUNTIME_LABEL[0]]
    family_classifier.fit(X_train_base_features, y_train_family)
    # as a backup, in case we don't want or can't use the family specific regressor
    base_runtime_model.fit(X_train_base_features, y_train_runtime)

    # now we also want a random forest model for each family
    # first we need to slice the df
    df_sliced_dict_train = {}
    for family in X_train[CURRENT_FAMILY_LABEL[0]].unique():
        # each entry contains base features + family_label + runtime_label
        df_sliced_dict_train[family] = X_train[X_train[CURRENT_FAMILY_LABEL[0]] == family]
        df_sliced_dict_train[family] = df_sliced_dict_train[family].dropna()
        df_sliced_dict_train[family] = df_sliced_dict_train[family].drop(labels=CURRENT_FAMILY_LABEL, axis=1)
        print()

    # this dict will map the family to a regressor trained only on one family
    family_specific_models = {}
    with parallel_backend('threading', n_jobs=4):
        for family in df_sliced_dict_train.keys():
            if len(df_sliced_dict_train[family]) > FAMILY_COUNT_THRESHOLD:
                # the model shouldn't use any labels as features:
                runtime_X_train = df_sliced_dict_train[family].drop(labels=CURRENT_RUNTIME_LABEL, axis=1)
                runtime_y_train = df_sliced_dict_train[family][CURRENT_RUNTIME_LABEL[0]]
                # we are using a random forest regression
                model = RandomForestClassifier(random_state=42, verbose=1)
                # # if we use a model that needs scaling/pipeline based fittig:
                # model = Pipeline([
                #     ('scaling', StandardScaler()),
                #     ('classification', MLPClassifier(random_state=42, max_iter=500))
                #     ])
                model.fit(runtime_X_train, runtime_y_train)
                family_specific_models[family] = model
            else:
                family_specific_models[family] = base_runtime_model

    # now we have a hierarchical model:
    # first let 'family_classifier' predict the family
    # then let the family specific regressor predict the runtime
    runtime_pred = []
    y_pred_family = family_classifier.predict(X_test_base_features)
    for idx, predicted_family in enumerate(y_pred_family):
        if predicted_family in family_specific_models.keys():
            current_row_df = X_test_base_features.iloc[[idx]]
            curr_pred = family_specific_models[predicted_family].predict(current_row_df)
            runtime_pred.append(curr_pred[0])
        else:
            # should never occur
            current_row_df = X_test_base_features.iloc[[idx]]
            curr_pred = base_runtime_model.predict(current_row_df)
            runtime_pred.append(curr_pred[0])

    y_test_runtime = X_test[CURRENT_RUNTIME_LABEL]
    print("Accuracy of the class prediction: %0.4f" % metrics.accuracy_score(y_test, y_pred_family))
    # print("Accuracy of the runtime prediction: %0.4f" % metrics.r2_score(y_test_runtime, runtime_pred))
    print("Accuracy of the runtime prediction: %0.4f" % metrics.accuracy_score(y_test_runtime, runtime_pred))



if __name__ == '__main__':
    # define random seed first
    np.random.seed(0)
    pipeline()