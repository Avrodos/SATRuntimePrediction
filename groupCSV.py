import os
import sys
from typing import Final
import create_METIS_graphs as metis
import pandas as pd

SRC_DIR: Final[str] = sys.argv[1]
TIME_ENDING: Final[str] = "_time_taken_to_write_graph.csv"
SIZE_ENDING: Final[str] = "_size_taken_to_write_graph.csv"
FORMAT_COLUMNS: Final = ['NetworkitBinary', 'METIS', 'EdgeListSpaceOne', 'ThrillBinary']
TIME_FILE_NAME: Final[str] = "time_taken_to_write_graph2.csv"
SIZE_FILE_NAME: Final[str] = "size_taken_to_write_graph2.csv"


# receives directory of the CSV's that have to be grouped
# outputs one CSV for time data and one CSV for size data
if __name__ == '__main__':
    files = os.listdir(SRC_DIR)
    # time_df = pd.DataFrame(columns=FORMAT_COLUMNS)
    # size_df = pd.DataFrame(columns=FORMAT_COLUMNS)
    df = pd.DataFrame(columns=metis.COLUMNS)

    for file in files:
        full_path = os.path.join(SRC_DIR, file)
        if os.path.isfile(full_path) and file.endswith(metis.FILE_ENDING):
            temp_df = pd.read_csv(full_path)
            df = pd.concat([df, temp_df])
            # if file.endswith(TIME_ENDING):
            #     time_df = pd.concat([time_df, temp_df])
            # elif file.endswith(SIZE_ENDING):
            #     size_df = pd.concat([size_df, temp_df])
    df.set_index(metis.COLUMNS[0], inplace=True)

    # now remove the unnecessary last column
    # time_df = time_df.drop(time_df.columns[-1], axis=1)
    # size_df = size_df.drop(size_df.columns[-1], axis=1)

    # write results to CSV's
    output_path = os.getcwd()
    # time_df.to_csv(output_path + "/" + TIME_FILE_NAME)
    # size_df.to_csv(output_path + "/" + SIZE_FILE_NAME)
    df.to_csv(output_path + "/" + metis.FILE_ENDING[1:])
