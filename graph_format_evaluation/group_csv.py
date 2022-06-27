import os
import sys
from typing import Final
import pandas as pd

COLUMNS: Final = ['ID', 'Time to create graph', 'Time to write graph', 'Space Needed']
FILE_ENDING: Final[str] = "_time_and_size_data.csv"
SRC_DIR: Final[str] = sys.argv[1]

# receives directory of the CSV's that have to be grouped
# outputs one CSV for time data and one CSV for size data
if __name__ == '__main__':
    files = os.listdir(SRC_DIR)
    df = pd.DataFrame(columns=COLUMNS)

    for file in files:
        full_path = os.path.join(SRC_DIR, file)
        if os.path.isfile(full_path) and file.endswith(FILE_ENDING):
            temp_df = pd.read_csv(full_path)
            df = pd.concat([df, temp_df])
    df.set_index(COLUMNS[0], inplace=True)

    # write results to CSV's
    output_path = os.getcwd()
    df.to_csv(output_path + "/" + FILE_ENDING[1:])
