import glob
import os
import sys
from typing import Final
import pandas as pd

SRC_DIR: Final[str] = sys.argv[1]
OUTPUT_DIR: Final[str] = 'new_features.csv'

if __name__ == '__main__':
    # get a list of all files in the directory
    files = glob.glob(SRC_DIR + "/*_features")
    df = []
    # append them all to a list
    for f in files:
        csv = pd.read_csv(f)
        df.append(csv)
    # concat to df
    df = pd.concat(df)
    df.to_csv(OUTPUT_DIR)
