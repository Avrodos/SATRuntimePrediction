import os
from typing import Final

import pandas as pd
from matplotlib import pyplot as plt

COLUMNS: Final = {'Measure', 'EdgeListSpaceOne', 'METIS', 'ThrillBinary', 'NetworkitBinary'}


def evaluate_graph_formats(size_table_path, time_table_path):
    if os.path.isfile(size_table_path) and os.path.isfile(time_table_path):
        read_time_df = pd.read_csv(time_table_path, index_col=0)
        read_size_df = pd.read_csv(size_table_path, index_col=0)

        time_table = pd.DataFrame(columns=COLUMNS)
        size_table = pd.DataFrame(columns=COLUMNS)

        # TODO: Create proper table and decide on format
        time_means = read_time_df.mean()
        time_median = read_time_df.median()
        time_025_quantile = read_time_df.quantile(0.25)
        time_075_quantile = read_time_df.quantile(0.75)

        size_means = read_size_df.mean()
        size_median = read_size_df.median()
        size_025_quantile = read_size_df.quantile(0.25)
        size_075_quantile = read_size_df.quantile(0.75)

        time_boxplot = read_time_df.boxplot(showmeans=True)
        time_boxplot.set_ylabel('Time in s')
        plt.title('Graph-Format Comparison of Writing-Times')
        plt.show()
        size_boxplot = read_size_df.boxplot(showmeans=True)
        size_boxplot.set_ylabel('File size in Bytes')
        plt.title('Graph-Format Comparison of File Sizes')
        plt.show()


# script takes in the measured time and size for diff. graph formats and creates boxplots for evaluation
if __name__ == '__main__':
    size_csv_path = "../data/measured_data/size_taken_to_write_graph.csv"
    time_csv_path = "../data/measured_data/time_taken_to_write_graph.csv"
    evaluate_graph_formats(size_csv_path, time_csv_path)