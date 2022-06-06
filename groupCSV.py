import os
import sys
from typing import Final

import pandas as pd

SRC_DIR: Final[str] = sys.argv[1]
TIME_ENDING: Final[str] = "_time_taken_to_write_graph.csv"
SIZE_ENDING: Final[str] = "_size_taken_to_write_graph.csv"
COLUMNS: Final = ['NetworkitBinary', 'METIS', 'EdgeListSpaceOne', 'ThrillBinary']
TIME_FILE_NAME: Final[str] = "time_taken_to_write_graph.csv"
SIZE_FILE_NAME: Final[str] = "size_taken_to_write_graph.csv"

# receives directory of the CSV's that have to be grouped
# outputs one CSV for time data and one CSV for size data
if __name__ == '__main__':
    files = os.listdir(SRC_DIR)
    time_df = pd.DataFrame(columns=COLUMNS)
    size_df = pd.DataFrame(columns=COLUMNS)
    for file in files:
        full_path = os.path.join(SRC_DIR, file)
        if os.path.isfile(full_path):
            temp_df = pd.read_csv(full_path)
            if file.endswith(TIME_ENDING):
                time_df = pd.concat([time_df, temp_df])
            elif file.endswith(SIZE_ENDING):
                size_df = pd.concat([size_df, temp_df])

    # now remove the unnecessary last column
    time_df = time_df.drop(time_df.columns[-1], axis=1)
    size_df = size_df.drop(size_df.columns[-1], axis=1)

    # write results to CSV's
    output_path = os.getcwd()
    time_df.to_csv(output_path + "/" + TIME_FILE_NAME)
    time_df.to_csv(output_path + "/" + SIZE_FILE_NAME)

