# only keep rows whose runtime we have
import os
import sys
from typing import Final

import numpy as np
import pandas as pd

PATH_DF: Final[str] = sys.argv[1]
PATH_LABELS: Final[str] = sys.argv[2]


def main():
    # current working directory
    cwd = os.getcwd()

    # used to split the two merged data sets
    df = pd.read_csv(PATH_DF, index_col=0)
    df['CaDiCaL_DVDL_V1'].replace('               ', np.nan, inplace=True)
    df.dropna(subset='CaDiCaL_DVDL_V1', inplace=True)
    print(df.columns.get_loc('kissat_relaxed'))
    df.drop(df.iloc[:, 28:], axis=1, inplace=True)
    df.to_csv(cwd + "/data/runtimes/only_new_solver_runtimes.csv")

    # remove suffixes from feature df
    df = pd.read_csv(PATH_DF)
    df["hash"] = df['ID'].str.split('-').str[0]
    df = df.drop('ID', axis=1)
    df = df.drop('Time_Diameter', axis=1)
    df = df.drop('Time_EffectiveDiameterApproximation', axis=1)
    df.set_index('hash', inplace=True)
    df.to_csv(cwd + "/data/measured_data/with_duplicate_hashes/SAT_time.csv")

    # remove non-unique rows from features df
    df = df.drop_duplicates(['hash'])
    df.to_csv(cwd + "/data/measured_data/SAT_features.csv")

    # merge df to only keep rows for which we have features AND labels
    label_df = pd.read_csv(PATH_LABELS)
    merged_df = df.merge(label_df, on='hash', how='left')
    merged_df = merged_df[['hash', 'min_label', 'log_min_label']]
    merged_df.to_csv(cwd + "/data/runtimes/runtime_labels.csv")

    # replace timeouts (nan)
    df['min_label'] = df['min_label'].fillna(5000)
    df['log_min_label'] = df['log_min_label'].fillna(np.log(5000))
    df.to_csv(cwd + "/data/runtimes/runtime_labels.csv")

    # remove unnamed rows
    df.drop(df.columns[[0,1]], axis=1, inplace=True)
    df.set_index('hash', inplace=True)
    df.to_csv(cwd + "/data/runtimes/runtime_labels.csv")


# first argument should be path to feature file, second argument is path to label file
if __name__ == '__main__':
    main()