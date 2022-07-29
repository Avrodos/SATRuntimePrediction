import math
import os
import sys
import time
from typing import Final

import networkit as nk

from extract_features import extract_features

SRC_DIR: Final[str] = sys.argv[1]
FEATURE_OUTPUT_DIR: Final[str] = 'data/measured_data/'
FEATURE_FILE_ENDING: Final[str] = '_features'
TIME_FILE_ENDING: Final[str] = '_time'


def create_vig_from_file(path: str):
    with open(path) as file:
        for line in file:
            words = line.split()
            # ignore comments and unit clauses
            if len(words) <= 2 or words[0] == str('c'):
                continue
            # since each line has a '0' as EOL we have to rem 1 from len
            bin_coeff = math.comb(len(words) - 1, 2)
            weight = 1.0 / bin_coeff
            if words[0] == str('p') and words[1] == 'cnf' and len(words) >= 4:
                # the first line gives is num_vars and num_clauses
                num_vars = int(words[2])
                num_clauses = int(words[3])
                file_graph = nk.Graph(num_vars, True)
            else:
                # we have to remove the '0' from the len as EOL marker
                curr_len = len(words) - 1
                for i in range(curr_len):
                    for j in range(i + 1, curr_len):
                        # since indices in the graph are 0-based, but in CNF they are 1-based
                        node_a = abs(int(words[i])) - 1
                        node_b = abs(int(words[j])) - 1
                        if not file_graph.hasEdge(node_a, node_b):
                            file_graph.addEdge(node_a, node_b, weight)
                        else:
                            file_graph.increaseWeight(node_a, node_b, weight)
    return file_graph


def create_graph():
    # to strip ending of file path, e.g. '.cnf'
    file_name = os.path.basename(SRC_DIR)
    file_name_without_ending = os.path.splitext(file_name)[0]
    # we will measure time from reading the cnf to creating the graph
    time_start = time.process_time()
    graph = create_vig_from_file(SRC_DIR)
    time_create_graph = time.process_time() - time_start
    return file_name_without_ending, graph, time_create_graph


# script receives file path  of a .cnf as input parameter
def pipeline():
    current_id, graph, time_create_graph = create_graph()

    full_output_path_features = FEATURE_OUTPUT_DIR + current_id + FEATURE_FILE_ENDING
    full_output_path_time = FEATURE_OUTPUT_DIR + current_id + TIME_FILE_ENDING
    # check whether the given file already exists
    if os.path.exists(full_output_path_features):
        return

    feature_df, time_df = extract_features(current_id, graph, time_create_graph)
    # write features into a file
    feature_df.to_csv(full_output_path_features)
    time_df.to_csv(full_output_path_time)


if __name__ == '__main__':
    pipeline()
