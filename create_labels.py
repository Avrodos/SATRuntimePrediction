import os
import sys
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

RUNTIME_LABELS_PATH: Final[str] = sys.argv[1]
# comment out if not needed
ALL_RUNTIMES_PATH: Final[str] = sys.argv[2]
LABEL_FILE_NAME: Final[str] = 'runtime_labels.csv'


def write_to_csv(label_df):
    output_path = os.getcwd() + "/data/runtimes/" + LABEL_FILE_NAME
    label_df.to_csv(output_path)


def read_labels_from_csv():
    label_df = pd.read_csv(RUNTIME_LABELS_PATH)
    label_df.set_index('hash', inplace=True)
    return label_df


def read_runtimes_from_csv():
    runtime_df = pd.read_csv(ALL_RUNTIMES_PATH)
    runtime_df.set_index('hash', inplace=True)

    # cast to floats, all timeouts will become 'nan'
    runtime_df = runtime_df.apply(pd.to_numeric, errors='coerce')
    return runtime_df


def add_family_label():
    label_df = read_labels_from_csv()
    # # read and merge family labels
    # family_df = pd.read_csv(os.getcwd() + "/data/meta_data/hash_to_family_mapping.csv")
    # family_df.set_index('hash', inplace=True)
    # merged_df = label_df.join(family_df, how='left')
    # family_counts = merged_df['family'].value_counts()
    merged_df = label_df

    # # we will keep the three most common family labels and replace all others with 'other'
    # most_common_families = ['cryptography', 'bitvector', 'antibandwidth']
    # merged_df['four_families'] = np.where(merged_df['family'].isin(most_common_families), merged_df['family'], 'other')
    # family_counts = merged_df['four_families'].value_counts()

    # group all families with less than threshold occurences as 'other'
    threshold = 10
    col = 'all_families_threshold_' + str(threshold)
    merged_df[col] = merged_df['all_families']
    merged_df.loc[merged_df[col].value_counts()[merged_df[col]].values < 10, col] = "other "

    family_counts = merged_df[col].value_counts()
    print("Number of families:")
    print(family_counts.shape[0])
    print("Percentage of each family: ")
    print(family_counts / merged_df.shape[0])

    write_to_csv(merged_df)


# label = num of solvers that timed out
def calculate_timeout_label():
    label_df = read_labels_from_csv()
    runtime_df = read_runtimes_from_csv()
    # number of timeouts per row
    label_df['timeout_count_label'] = runtime_df.isnull().sum(axis=1)
    solver_count = 18
    # factor of solvers having a timeout
    label_df['timeout_factor_label'] = label_df['timeout_count_label'] / solver_count

    write_to_csv(label_df)


def calculate_parity_two():
    label_df = read_labels_from_csv()
    runtime_df = read_runtimes_from_csv()

    # a timeout happens at 5000s. parity 2 doubles its weight to punish a timeout
    runtime_df.fillna(10000, inplace=True)
    # calculate mean over a row
    label_df['parity_two_label'] = runtime_df.mean(axis=1)
    print(label_df)

    # calculate log of par2
    label_df['log10_parity_two_label'] = np.log10(label_df['parity_two_label'])

    # write to file
    write_to_csv(label_df)


def calculate_categorical_labels():
    label_df = read_labels_from_csv()
    current_label = 'parity_two_label'
    # calculate a 3 category label: easy <= x seconds, hard = timeout, medium is inbetween
    easy_threshold = 4000.0
    timeout_threshold = 10000
    label_df['three_categories_label'] = label_df[current_label]
    label_df['three_categories_label'].mask(label_df[current_label] <= easy_threshold, 'easy', inplace=True)
    label_df['three_categories_label'].mask(label_df[current_label] >= timeout_threshold, 'hard', inplace=True)
    label_df['three_categories_label'].mask((label_df[current_label] > easy_threshold)
                                            & (label_df[current_label] < timeout_threshold), 'medium', inplace=True)
    print(label_df['three_categories_label'].value_counts())
    print(label_df['three_categories_label'].value_counts() / label_df.shape[0])
    write_to_csv(label_df)


def try_casting_to_float(x):
    res = None
    try:
        res = x.astype(float)
    except:
        res = x
    return res


# receives a csv with runtimes of 18 solvers and creates labels out of them
def calculate_min_labels():
    label_df = read_labels_from_csv()
    runtime_df = read_runtimes_from_csv()

    # min runtime as label - e.g. portfolio solver
    # calculate min over a row
    label_df['min_label'] = runtime_df.min(axis=1)

    # calculate log of min
    label_df['log10_min_label'] = np.log10(label_df['min_label'])

    # write to file
    write_to_csv(label_df)


def calculate_time_mean_per_features():
    input_time_path = os.getcwd() + '/data/measured_data/SAT_time.csv'
    output_time_path = os.getcwd() + '/data/measured_data/SAT_time_labels.csv'

    # read the time df
    input_time_df = pd.read_csv(input_time_path, index_col=0)

    # now calculate the mean for every column
    time_mean_per_feature = input_time_df.mean(axis=0)
    time_mean_per_feature.drop(labels=['Time_ReadGraph'], inplace=True)
    time_df = time_mean_per_feature.to_frame()
    time_df.rename(columns={0: 'time_mean_per_feature'}, inplace=True)

    # visualize the mean-times
    num_colors = 4
    cmap = plt.get_cmap('viridis', num_colors)
    cmap.set_over('red')
    scatter_plot = plt.scatter(time_mean_per_feature, list(time_df.index.values), c=time_mean_per_feature, cmap=cmap,
                               vmin=1.3, vmax=160)
    plt.colorbar(scatter_plot, extend='max')
    plt.title("Extraction Time of Features")
    plt.xlabel("Mean Extraction Time per Feature in s")
    plt.ylabel("Feature Classes")
    plt.tight_layout()
    plt.show()

    # calculate a 4 category label, depending on how long the feature calculation took on average:
    # thresholds are taken by having a look at mean times and identifying clusters
    cheap_threshold = 1.5
    moderate_threshold = 70.0
    expensive_threshold = 160.0
    time_df['time_cost_category'] = time_df['time_mean_per_feature']
    time_df['time_cost_category'].mask(time_df['time_mean_per_feature'] <= cheap_threshold, 'cheap', inplace=True)
    time_df['time_cost_category'].mask((time_df['time_mean_per_feature'] <= moderate_threshold)
                                       & (time_df['time_mean_per_feature'] > cheap_threshold), 'moderate', inplace=True)
    time_df['time_cost_category'].mask((time_df['time_mean_per_feature'] <= expensive_threshold)
                                       & (time_df['time_mean_per_feature'] > moderate_threshold), 'expensive',
                                       inplace=True)
    time_df['time_cost_category'].mask((time_df['time_mean_per_feature'] > expensive_threshold), 'very expensive',
                                       inplace=True)
    print(time_df['time_cost_category'].value_counts())

    # write it to a file
    time_df.to_csv(output_time_path)


def kmeans_cluster_labels():
    label_df = read_labels_from_csv()
    current_label = 'parity_two_label'
    # number of clusters
    k = 3
    new_label_name = f"{k}-means_label"
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_df = pd.DataFrame(label_df[current_label])
    # we have to drop nan's
    kmeans_df.dropna(inplace=True)
    # execute kmeans and retrieve labels
    kmeans_df[new_label_name] = kmeans.fit_predict(kmeans_df)
    # drop old label column
    kmeans_df = kmeans_df.drop(labels=current_label, axis=1)
    print(kmeans_df)

    # we have to merge due to the drops
    merged_df = label_df.join(kmeans_df, how='left')
    # plot our clusters
    frame = plt.scatter(merged_df[current_label], merged_df.index, c=merged_df[new_label_name])
    frame.axes.get_yaxis().set_visible(False)
    plt.title(f'{k}-Means Clustering of Average Par-2 Time for Labelling')
    plt.xlabel('Average Par-2 Time of Each Instance')
    plt.show()

    # write to file
    write_to_csv(merged_df)


def find_optimal_num_of_clusters():
    # inspired from https://medium.com/pursuitnotes/k-means-clustering-model-in-6-steps-with-python-35b532cfa8ad
    wcss = []
    label_df = read_labels_from_csv()
    current_label = 'parity_two_label'
    kmeans_df = pd.DataFrame(label_df[current_label])
    # we have to drop nan's
    kmeans_df.dropna(inplace=True)
    # this loop will fit the k-means algorithm to our data and
    # second we will compute the within cluster sum of squares and #appended to our wcss list.
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(kmeans_df)
        # kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
        wcss.append(kmeans.inertia_)

    # Plot the elbow graph
    fix, ax = plt.subplots()
    ax.plot(range(1, 11), wcss)
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    # to have a tick at every integer
    plt.xticks(np.arange(0, 11, 1))

    # adjust the elbow_point by looking at the graph
    elbow_point = (3, wcss[2])
    ax.annotate('Elbow Point', xy=elbow_point, xycoords='data',
                xytext=(0.375, 0.2), textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.2', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='black', relpos=(0, 0.5)),
                horizontalalignment='right', verticalalignment='top', )

    plt.show()
    print('done')


if __name__ == '__main__':
    find_optimal_num_of_clusters()
