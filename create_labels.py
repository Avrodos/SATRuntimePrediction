import os
import sys
from typing import Final

import numpy as np
import pandas as pd

SRC_DIR: Final[str] = sys.argv[1]
LABEL_FILE_NAME: Final[str] = 'runtime_labels.csv'


def write_to_csv(label_df):
    output_path = os.getcwd() + "/data/runtimes/" + LABEL_FILE_NAME
    label_df.to_csv(output_path)


def calculate_categorical_labels():
    label_df = pd.read_csv(SRC_DIR)
    label_df.set_index('hash', inplace=True)
    # calculate a 3 category label: easy = < x seconds, hard = timeout, medium is inbetween
    easy_threshold = 60.0
    timeout_threshold = 5000
    label_df['three_categories_label'] = label_df['min_label']
    label_df['three_categories_label'].mask(label_df['min_label'] <= easy_threshold, 'easy', inplace=True)
    label_df['three_categories_label'].mask(label_df['min_label'] >= timeout_threshold, 'hard', inplace=True)
    label_df['three_categories_label'].mask((label_df['min_label'] > easy_threshold)
                                            & (label_df['min_label'] < timeout_threshold), 'medium', inplace=True)
    print(label_df['three_categories_label'].value_counts())
    write_to_csv(label_df)


def try_casting_to_float(x):
    res = None
    try:
        res = x.astype(float)
    except:
        res = x
    return res


# receives a csv with runtimes of 18 solvers and creates labels out of them
def main():
    runtime_df = pd.read_csv(SRC_DIR)
    label_df = pd.DataFrame(runtime_df['hash'])
    label_df.set_index('hash', inplace=True)
    runtime_df.set_index('hash', inplace=True)

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
    calculate_categorical_labels()
