# creates and returns a VIG from a file
# assumes that the file is in DIMACS CNF format
import math
import os
import sys
from os.path import exists
from typing import Final
import networkit as nk
import time
import pandas as pd

SRC_DIR: Final[str] = sys.argv[1]
COLUMNS: Final = ['ID', 'Time to create graph', 'Time to write graph', 'Space Needed']
GRAPH_OUTPUT_DIR: Final[str] = "data/created_vigs/"
MEASURED_OUTPUT_DIR: Final[str] = "data/measured_data/"
FILE_ENDING: Final[str] = "_time_and_size_data.csv"


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


# write graph to a file and return the file size
def write_graph_to_METIS(test_graph, output_path):
    if not os.path.isdir(GRAPH_OUTPUT_DIR):
        os.makedirs(GRAPH_OUTPUT_DIR)
    # example_path: str = OUTPUT_PATH + $id + _METIS
    nk.writeGraph(test_graph, output_path, nk.Format.METIS)
    return os.path.getsize(output_path)


def main():
    # to strip ending of file path, e.g. '.cnf'
    file_name = os.path.basename(SRC_DIR)
    file_name_without_ending = os.path.splitext(file_name)[0]
    full_output_path = GRAPH_OUTPUT_DIR + file_name_without_ending + "_METIS"
    # first check if we already created the file in a previous run
    if os.path.exists(full_output_path):
        return
    # we will use the file names for the id during the rest of the project
    current_id = file_name_without_ending
    # we will measure time from reading the cnf to creating the graph
    # and from having the graph to writing it in a file
    time_start = time.process_time()
    graph = create_vig_from_file(SRC_DIR)
    time_create_graph = time.process_time() - time_start
    print(graph.numberOfNodes(), graph.numberOfEdges())
    print("Time taken to create graph: " + str(time_create_graph))

    time_start = time.process_time()
    file_size = write_graph_to_METIS(graph, full_output_path)
    time_write_graph = time.process_time() - time_start

    data = {
        COLUMNS[0]: [current_id],
        COLUMNS[1]: [time_create_graph],
        COLUMNS[2]: [time_write_graph],
        COLUMNS[3]: [file_size]
    }
    df = pd.DataFrame(data, columns=COLUMNS)
    df.set_index(COLUMNS[0], inplace=True)
    # save to csv
    os.makedirs(os.path.dirname(MEASURED_OUTPUT_DIR), exist_ok=True)
    df.to_csv(MEASURED_OUTPUT_DIR + file_name_without_ending + FILE_ENDING)


# script receives file path  of a .cnf as input parameter
if __name__ == '__main__':
    main()
