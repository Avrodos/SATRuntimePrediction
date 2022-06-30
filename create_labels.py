import sys
from typing import Final

import numpy as np
import pandas as pd
from numpy import ma

SRC_DIR: Final[str] = sys.argv[1]
LABEL_FILE_NAME: Final[str] = 'runtime_labels.csv'


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

    print(label_df)


if __name__ == '__main__':
    main()
