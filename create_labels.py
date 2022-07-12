import os
import sys
from typing import Final

import numpy as np
import pandas as pd

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


if __name__ == '__main__':
    add_family_label()
