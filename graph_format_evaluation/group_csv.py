import os
import sys
from typing import Final
import pandas as pd

FILE_ENDING: Final[str] = "_features"
SRC_DIR: Final[str] = sys.argv[1]

# receives directory of the CSV's that have to be grouped
# outputs one CSV with all data
if __name__ == '__main__':
    files = os.listdir(SRC_DIR)
    df = pd.DataFrame()

    for file in files:
        full_path = os.path.join(SRC_DIR, file)
        if os.path.isfile(full_path) and file.endswith(FILE_ENDING):
            temp_df = pd.read_csv(full_path)
            df = pd.concat([df, temp_df])
    df.set_index('ID', inplace=True)

    # write results to CSV's
    output_path = os.getcwd()
    df.to_csv(output_path + "/" + "SAT_" + FILE_ENDING[1:])
