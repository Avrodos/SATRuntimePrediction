import os
import sys
from typing import Final

import pandas as pd

SRC_DIR: Final[str] = sys.argv[1]
COLUMNS: Final = ['hash', 'value']
START_FILE: Final[str] = 'cadical_runtimes.csv'
RUNTIME_ENDING: Final[str] = '_runtime'


# receives the path containing all runtime csv's and combines them into one
def main():
    files = os.listdir(SRC_DIR)
    first_full_path = os.path.join(SRC_DIR, START_FILE)
    df = pd.read_csv(first_full_path)
    # to be able to identify the corresponding solver
    file_name_without_ending = os.path.splitext(START_FILE)[0]
    df = df.rename(columns={'value': file_name_without_ending})

    for file in files:
        full_path = os.path.join(SRC_DIR, file)
        if os.path.isfile(full_path) and full_path != first_full_path and file.endswith('.csv'):
            temp_df = pd.read_csv(full_path)
            # to be able to identify the corresponding solver
            file_name_without_ending = os.path.splitext(file)[0]
            temp_df = temp_df.rename(columns={'value': file_name_without_ending})
            df = df.merge(temp_df, on='hash', how='left')
    df.set_index(COLUMNS[0], inplace=True)
    output_path = os.getcwd()
    df.to_csv(output_path + "/solver_runtimes.csv")


if __name__ == '__main__':
    main()
