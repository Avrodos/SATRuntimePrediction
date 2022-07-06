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

def read_from_csv():
    label_df = pd.read_csv(RUNTIME_LABELS_PATH)
    label_df.set_index('hash', inplace=True)
    return label_df

def calculate_parity_two():
    label_df = read_from_csv()
    runtime_df = pd.read_csv(ALL_RUNTIMES_PATH)
    runtime_df.set_index('hash', inplace=True)

    # cast to floats, all timeouts will become 'nan'
    runtime_df = runtime_df.apply(pd.to_numeric, errors='coerce')
    # a timeout happens at 5000s. parity 2 doubles its weight to punish a timeout
    runtime_df.fillna(10000, inplace=True)
    # calculate mean over a row
    label_df['parity_two_label'] = runtime_df.mean(axis=1)
    print(label_df)
    # write to file
    write_to_csv(label_df)


def calculate_categorical_labels():
    label_df = read_from_csv()
    # calculate a 3 category label: easy = < x seconds, hard = timeout, medium is inbetween
    easy_threshold = 10.0
    timeout_threshold = 5000
    label_df['three_categories_label'] = label_df['min_label']
    label_df['three_categories_label'].mask(label_df['min_label'] <= easy_threshold, 'easy', inplace=True)
    label_df['three_categories_label'].mask(label_df['min_label'] >= timeout_threshold, 'hard', inplace=True)
    label_df['three_categories_label'].mask((label_df['min_label'] > easy_threshold)
                                            & (label_df['min_label'] < timeout_threshold), 'medium', inplace=True)
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
    runtime_df = read_from_csv()
    label_df = pd.DataFrame(runtime_df['hash'])
    label_df.set_index('hash', inplace=True)

    # cast to floats, all timeouts will become 'nan'
    runtime_df = runtime_df.apply(pd.to_numeric, errors='coerce')

    # min runtime as label - e.g. portfolio solver
    # calculate min over a row
    label_df['min_label'] = runtime_df.min(axis=1)

    # calculate log of min
    label_df['log_min_label'] = np.log(label_df['min_label'])

    # write to file
    write_to_csv()


if __name__ == '__main__':
    calculate_parity_two()
