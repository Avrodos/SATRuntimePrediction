import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn import model_selection, metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

PATH_FEATURES: Final[str] = sys.argv[1]
PATH_LABELS: Final[str] = sys.argv[2]


def load_df():
    # load features and labels
    feature_df = pd.read_csv(PATH_FEATURES, index_col=0)
    label_df = pd.read_csv(PATH_LABELS, index_col=0)
    return feature_df, label_df


def add_family_to_df(preprocessed_df):
    # We can add the family to the features
    family_df = pd.read_csv(os.getcwd() + "/data/meta_data/hash_to_family_mapping.csv")
    family_df.set_index('hash', inplace=True)
    preprocessed_feature_df = preprocessed_df.join(family_df, how='left')
    return preprocessed_feature_df


def string_to_numerical_encoder(feature_df):
    # many models need numerical labels, even for categorical ones.
    # this method encodes the string label into a random but definite numerical label
    columns_to_encode = list(feature_df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    for feature in columns_to_encode:
        try:
            feature_df[feature] = le.fit_transform(feature_df[feature])
        except:
            print('Error encoding ' + feature)
    return feature_df


def regression_model_preprocessing(loaded_feature_df, loaded_label_df):
    # choose current label
    current_label = ['log10_parity_two_label']

    # to ensure we have a label on each feature
    merged_df = loaded_label_df[current_label].join(loaded_feature_df, how='left')
    # we need to drop nan instances
    merged_df.dropna(inplace=True)
    # # drop timeouts and 0's
    # merged_df = merged_df[
    #     (merged_df.log10_parity_two_label != np.log10(10000)) & (merged_df.log10_parity_two_label != np.log10(0))]
    # drop only the 0's
    merged_df = merged_df[(merged_df.log10_parity_two_label != np.log10(0))]

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(labels=current_label, axis=1)
    label_df = merged_df[current_label]
    # if more information about the label df are desired:
    print(pd.DataFrame(label_df).describe())
    # scale features beforehand:
    sc = StandardScaler()
    feature_df = sc.fit_transform(feature_df)

    # # if dimensionality reduction using a PCA is desired:
    # pca = PCA(n_components=40)
    # principal_components = pca.fit_transform(feature_df)
    # feature_df = pd.DataFrame(data=principal_components)

    return feature_df, label_df.values.ravel()


def compare_model_to_features_with_family(model, preprocessed_feature_df, preprocessed_label_df):
    # add family as feature
    features_with_family = add_family_to_df(preprocessed_feature_df)

    # replace a family name with an arbitrary, but consistent, number
    features_with_family = string_to_numerical_encoder(features_with_family)

    # # one-hot encoding - alternative to the numerical encoder:
    # features_with_family = pd.get_dummies(features_with_family)

    # # to plot the distribution of our families
    plt.hist(features_with_family['family'], 88)
    plt.ylabel('Frequency')
    plt.xlabel('Family')
    plt.show()

    cv_scores = cross_val_score(model, features_with_family, preprocessed_label_df, cv=10)
    print(
        "Features with family: %0.4f accuracy with a standard deviation of %0.4f" % (cv_scores.mean(), cv_scores.std()))


def compare_model_to_features_with_category(model, preprocessed_feature_df, preprocessed_label_df):
    loaded_feature_df, loaded_label_df = load_df()
    features_with_category = preprocessed_feature_df.join(loaded_label_df['three_categories_label'], how='left')

    # replace a category name with an arbitrary, but consistent, number
    features_with_category = string_to_numerical_encoder(features_with_category)

    cv_scores = cross_val_score(model, features_with_category, preprocessed_label_df, cv=10)
    print("Features with category: %0.4f accuracy with a standard deviation of %0.4f" % (
    cv_scores.mean(), cv_scores.std()))


def test_train_regressor_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df):
    # train and evaluate Random Forest
    model = RandomForestRegressor(random_state=42, verbose=1)

    # first split into train and test
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_feature_df, preprocessed_label_df,
                                                        random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy of the model with all features: %0.4f" % metrics.r2_score(y_test, y_pred))

    # to calc and see hierarchical clustering based on spearman rank-order
    # necessary to handle multi collinear features
    calc_hierarchical_clustering(preprocessed_feature_df, X_train, X_test, y_train, y_test)


def cv_regression_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df):
    # choose desired ml model:
    model = RandomForestRegressor(random_state=42, verbose=1)
    # model = SVR()
    # model = MLPRegressor(random_state=42)
    # model = DecisionTreeRegressor(random_state=42)
    # model = AdaBoostRegressor(random_state=42)
    # model = VotingRegressor(estimators=[('rf',RandomForestRegressor(random_state=42)), ('mlp', MLPRegressor(random_state=42)), ('svc', SVR())])
    # model = VotingRegressor(n_jobs=-1,estimators=[('rf1',RandomForestRegressor(random_state=42)), ('rf2',RandomForestRegressor(random_state=0)), ('rf3',RandomForestRegressor(random_state=5))])

    # for R2-score
    cv_scores = cross_val_score(model, preprocessed_feature_df, preprocessed_label_df, cv=10, scoring='r2')
    print("R2 Score: %0.4f score with a standard deviation of %0.4f" % (cv_scores.mean(), cv_scores.std()))

    # for RMSE:
    cv_scores = cross_val_score(model, preprocessed_feature_df, preprocessed_label_df, cv=10,
                                scoring='neg_root_mean_squared_error')
    print("RMSE: %0.4f RMSE with a standard deviation of %0.4f" % (cv_scores.mean(), cv_scores.std()))

    # interestingly, adding the family to the features barely improves our model
    compare_model_to_features_with_family(model, preprocessed_feature_df, preprocessed_label_df)

    # adding 'easy', 'medium', etc to the features as label makes regression a lot more accurate
    # as expected, due to smaller search space
    compare_model_to_features_with_category(model, preprocessed_feature_df, preprocessed_label_df)


def classifier_model_preprocessing(loaded_feature_df, loaded_label_df):
    # optional: add family as label
    loaded_label_df = add_family_to_df(loaded_label_df)

    # choose current label
    current_label = ['family']

    # to ensure we have a label on each feature
    merged_df = loaded_label_df[current_label].join(loaded_feature_df, how='left')
    # we need to drop nan instances
    merged_df.dropna(inplace=True)

    # if we are using merged_df, we have to split into feature and labels df again
    feature_df = merged_df.drop(labels=current_label, axis=1)
    label_df = merged_df[current_label]
    print(pd.DataFrame(label_df).describe())

    # scale features beforehand:
    sc = StandardScaler()
    feature_df = sc.fit_transform(feature_df)

    # # dimensionality reduction using a PCA:
    # pca = PCA(n_components=30)
    # principal_components = pca.fit_transform(feature_df)
    # feature_df = pd.DataFrame(data=principal_components)

    return feature_df, label_df.values.ravel()


def see_false_predictions(feature_df, y_test, y_pred):
    # let's have a look at the wrong predictions
    indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    wrong_predictions = feature_df.iloc[indices, :]
    family_df = pd.read_csv(os.getcwd() + "/data/meta_data/hash_to_family_mapping.csv")
    family_df.set_index('hash', inplace=True)
    merged_df = wrong_predictions.join(family_df, how='left')
    print(merged_df['family'].value_counts())
    print(merged_df['family'].value_counts() / merged_df.shape[0])


def calc_permutation_importance(feature_df, X_train, y_train, classifier):
    # first: permutation based importance - we will see it doesn't tell us the whole truth
    result = permutation_importance(classifier, X_train, y_train, n_repeats=10, random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(classifier.feature_importances_)
    tree_indices = np.arange(0, len(classifier.feature_importances_)) + 0.5

    # visualise our results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices, classifier.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(feature_df.columns[tree_importance_sorted_idx])
    ax1.set_ylim((0, len(classifier.feature_importances_)))
    ax2.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=feature_df.columns[perm_sorted_idx],
    )
    fig.tight_layout()
    plt.show()


def calc_hierarchical_clustering(feature_df, X_train, X_test, y_train, y_test):
    # let's try hierarchical clustering : (taken from sklearn's docs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # since these have 0 variance, we do not gain any information from them and it would result it in a 0 division
    feature_df = feature_df \
        .drop(feature_df[['IsProperClustering', 'IsSingletonClustering']], axis=1)
    corr = spearmanr(feature_df).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    feature_names = feature_df.columns.tolist()
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=feature_names, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.show()

    # let's pick a threshold based on the dendogram
    threshold = 0.05
    cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    # take only the selected columns
    X_train_sel = X_train.iloc[:, selected_features]
    X_test_sel = X_test.iloc[:, selected_features]
    print(X_train_sel.columns)
    # if using a classifier
    model_on_selected_features = RandomForestClassifier(random_state=42)
    model_on_selected_features.fit(X_train_sel, y_train)
    print(
        "Accuracy on test data with features removed: {:.4f}".format(
            model_on_selected_features.score(X_test_sel, y_test)
        )
    )

    # # if using a regressor
    # model_on_selected_features = RandomForestRegressor(random_state=42)
    # model_on_selected_features.fit(X_train_sel, y_train)
    # print(
    #     "Accuracy on test data with features removed: {:.4f}".format(
    #         model_on_selected_features.score(X_test_sel, y_test)
    #     )
    # )


def cv_classifier_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df):
    # decide on the desired ml model
    classifier = RandomForestClassifier(random_state=42, verbose=1)
    # classifier = MLPClassifier(random_state=42)
    # classifier = SVC(random_state=42, verbose=1)
    # classifier = AdaBoostClassifier(random_state=42)
    # classifier = KNeighborsClassifier(4)
    # classifier = DecisionTreeClassifier(random_state=42)
    # classifier = VotingClassifier(estimators=[('rf',RandomForestClassifier(random_state=42)), ('mlp', MLPClassifier(random_state=42)), ('svc', SVC(random_state=42))])
    # classifier = VotingClassifier(estimators=[('rf1',RandomForestClassifier(random_state=42)), ('rf2',RandomForestClassifier(random_state=0)), ('rf3',RandomForestClassifier(random_state=5))])

    # evaluate using cross validation
    cv_scores = model_selection.cross_val_score(classifier, preprocessed_feature_df,
                                                preprocessed_label_df, cv=10)
    print("%0.4f accuracy with a standard deviation of %0.4f" % (cv_scores.mean(), cv_scores.std()))


def test_train_classifier_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df):
    # fit and test using test & train split - (worse) alternative to CV

    # # decide on model:
    # classifier = RandomForestClassifier(random_state=42)
    # classifier = MLPClassifier(random_state=42)
    classifier = MLPClassifier(random_state=42, solver='lbfgs', max_iter=238, learning_rate_init=0.6204719438877756,
                               hidden_layer_sizes=(710, 710, 710))

    # first split into train and test
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_feature_df, preprocessed_label_df,
                                                        random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Accuracy of the model with all features: %0.4f" % metrics.accuracy_score(y_test, y_pred))

    # let's have a look at the false predictions
    see_false_predictions(preprocessed_feature_df, y_test, y_pred)
    # to calc and see permutation based importance - won't help us much due to correlations
    calc_permutation_importance(preprocessed_feature_df, X_train, y_train, classifier)
    # to calc and see hierarchical clustering based on spearman rank-order
    # necessary to handle multi collinear features
    calc_hierarchical_clustering(preprocessed_feature_df, X_train, X_test, y_train, y_test)


def find_min_features_with_family(preprocessed_feature_df, preprocessed_label_df):
    # Only the 4 most important features:
    most_important_feature_names = ["#Nodes", "#ConnectedComponents", "MeanDegreeCentrality", "EntropyDegreeCentrality"]
    most_important_features = preprocessed_feature_df[
        np.intersect1d(preprocessed_feature_df.columns, most_important_feature_names)]

    # add family as feature
    features_with_family = add_family_to_df(most_important_features)

    # replace a family name with an arbitrary, but consistent, number
    features_with_family = string_to_numerical_encoder(features_with_family)

    # Create combinations of features as list
    combo_len = 6
    combo_fall_feat = sum([list(combinations(features_with_family, i)) for i in range(1, combo_len)], [])
    results = []

    # set x to be our features used in training
    for comb_features in list(combo_fall_feat):
        # Filter data by comb_features
        X = features_with_family[list(comb_features)]

        # # Train and evaluate your model using train/test split:
        # X_train, X_test, y_train, y_test = train_test_split(X, preprocessed_label_df)
        # clf = RandomForestRegressor(random_state=0, verbose=1)
        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # r2_score = metrics.r2_score(y_test, y_pred)
        #     # this feature importance might not be really meaingful due to correlation. Not sure though, since we have
        #     # only taken features with the lowest correlation using spearman ranking and hierarchical clustering
        # row_res = {"features": comb_features, "r2_score": r2_score, "feature_importances_": clf.feature_importances_}
        # current_measure = "r2_score"

        # evaluate using CV:
        model = RandomForestRegressor(random_state=0, verbose=1)
        # model = RandomForestClassifier(random_state=0, verbose=1)
        cv_scores = cross_val_score(model, X, preprocessed_label_df, cv=10)
        print("Base Features: %0.4f accuracy with a standard deviation of %0.4f" % (cv_scores.mean(), cv_scores.std()))
        row_res = {"features": comb_features, "cv_scores_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
        current_measure = "cv_scores_mean"

        # Store result
        results.append(row_res)

    # Sort result by accuracy
    sorted_results = sorted(results, key=lambda i: i[current_measure], reverse=True)

    # convert to df
    sorted_results = pd.DataFrame(sorted_results)

    # Print results
    print(sorted_results)

    # save cv results
    sorted_results.to_csv(
        os.getcwd() + "/data/measured_data/only_new_regression_cv_scores_max_5_features.csv")


def pipeline():
    # build your pipeline here. comment in all steps you currently want.
    loaded_feature_df, loaded_label_df = load_df()

    # if we want to use a regression model:
    preprocessed_feature_df, preprocessed_label_df = regression_model_preprocessing(loaded_feature_df, loaded_label_df)
    print(pd.DataFrame(preprocessed_label_df).describe())
    with parallel_backend('threading', n_jobs=4):
        # decide on evaluation method: CV or test/train
        cv_regression_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df)
        # test_train_regressor_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df)

    # # if we want to use a classifier model:
    # preprocessed_feature_df, preprocessed_label_df = classifier_model_preprocessing(loaded_feature_df, loaded_label_df)
    # with parallel_backend('threading', n_jobs=4):
    #     # decide on evaluation method: CV or test/train
    #     cv_classifier_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df)
    #     test_train_classifier_model_train_and_evaluate(preprocessed_feature_df, preprocessed_label_df)

    # # find min features with instance-family given as feature, or find most important features:
    # preprocessed_feature_df, preprocessed_label_df = regression_model_preprocessing(loaded_feature_df, loaded_label_df)
    # preprocessed_feature_df, preprocessed_label_df = classifier_model_preprocessing(loaded_feature_df, loaded_label_df)
    # with parallel_backend('threading', n_jobs=4):
    #     find_min_features_with_family(preprocessed_feature_df, preprocessed_label_df)


# first argument should be path to feature file, second argument is path to label file
if __name__ == '__main__':
    # define random seed first
    np.random.seed(0)
    pipeline()
