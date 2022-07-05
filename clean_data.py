# only keep rows whose runtime we have
import os
import sys
from typing import Final

import pandas as pd

PATH_DF: Final[str] = sys.argv[1]
PATH_LABELS: Final[str] = sys.argv[2]


def main():
    # current working directory
    cwd = os.getcwd()
    # remove suffixes from feature df
    df = pd.read_csv(PATH_DF)
    df["hash"] = df['ID'].str.split('-').str[0]
    df = df.drop('ID', axis=1)
    df = df.drop('Time_Diameter', axis=1)
    df = df.drop('Time_EffectiveDiameterApproximation', axis=1)
    df.set_index('hash', inplace=True)
    df.to_csv(cwd + "/data/measured_data/with_duplicate_hashes/SAT_time")

    # remove non-unique rows from features df
    df = df.drop_duplicates(['hash'])
    df.to_csv(cwd + "/data/measured_data/SAT_features")

    # merge df to only keep rows for which we have features AND labels
    label_df = pd.read_csv(PATH_LABELS)
    merged_df = df.merge(label_df, on='hash', how='left')
    merged_df = merged_df[['hash', 'min_label', 'log_min_label']]
    merged_df.to_csv(cwd + "/data/runtimes/runtime_labels.csv")


# first argument should be path to feature file, second argument is path to label file
if __name__ == '__main__':
    main()