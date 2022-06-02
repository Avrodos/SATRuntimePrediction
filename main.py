import math
from typing import Final, Any
import os
import networkit as nk
import pandas as pd
import time

SRC_DIR: Final[str] = "data/benchmarked_instances/"
GRAPH_OUTPUT_DIR: Final[str] = "data/created_vigs/"
MEASURED_OUTPUT_DIR: Final[str] = "data/measured_data"
GRAPH_FORMATS: Final = {'EdgeListSpaceOne', 'METIS', 'ThrillBinary', 'NetworkitBinary'}


# creates and returns a VIG from a file
# assumes that the file is in DIMACS CNF format
def create_vig_from_file(path: str):
    with open(path) as file:
        for line in file:
            words = line.split()
            # ignore comments and unit clauses
            if words[0] == str('c') or len(words) <= 2:
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


# tests to decide on a format
def test_graph_formats(test_graph, output_path):
    if not os.path.isdir(GRAPH_OUTPUT_DIR):
        os.makedirs(GRAPH_OUTPUT_DIR)
    # example_path: str = OUTPUT_PATH + "000a41cdca43be89ed62ea3abf2d0b64-snw_13_9_pre"

    local_df = pd.DataFrame(columns=GRAPH_FORMATS)
    result = [[],[]]
    time_row = []
    file_size_row = []

    time_start = time.process_time()
    graph_file = output_path + "_EdgeListSpaceOne"
    nk.writeGraph(test_graph, graph_file, nk.Format.EdgeListSpaceOne)
    time_row.append(time.process_time() - time_start)
    file_size_row.append(os.path.getsize(graph_file))

    time_start = time.process_time()
    graph_file = output_path + "_METIS"
    nk.writeGraph(test_graph, graph_file, nk.Format.METIS)
    time_row.append(time.process_time() - time_start)
    file_size_row.append(os.path.getsize(graph_file))

    time_start = time.process_time()
    graph_file = output_path + "_ThrillBinary_" + str(test_graph.numberOfNodes())
    nk.writeGraph(test_graph, graph_file, nk.Format.ThrillBinary)
    time_row.append(time.process_time() - time_start)
    file_size_row.append(os.path.getsize(graph_file))

    time_start = time.process_time()
    graph_file = output_path + "_NetworkitBinaryGraph"
    nk.writeGraph(test_graph, graph_file, nk.Format.NetworkitBinary, chunks=16, NetworkitBinaryWeights=3)
    time_row.append(time.process_time() - time_start)
    file_size_row.append(os.path.getsize(graph_file))

    result[0] = time_row
    result[1] = file_size_row
    return result


if __name__ == '__main__':
    time_df = pd.DataFrame(columns=GRAPH_FORMATS)
    size_df = pd.DataFrame(columns=GRAPH_FORMATS)
    file_list = sorted(os.listdir(SRC_DIR))
    # we are testing 25 files for now
    for curr_file in file_list:
        print("Current File Being Processed is: " + curr_file)
        full_src_path = SRC_DIR + curr_file
        graph = create_vig_from_file(full_src_path)
        print(graph.numberOfNodes(), graph.numberOfEdges())
        # to strip ending of file path, e.g. '.cnf'
        full_output_path = os.path.splitext(GRAPH_OUTPUT_DIR + curr_file)[0]
        # func result contains a list of lists.
        # 1 list of time data and then 1 list of file size data
        file_save_list = test_graph_formats(graph, full_output_path)
        # append row to df, use loc to access individual rows
        time_df.loc[len(time_df)] = file_save_list[0]
        size_df.loc[len(size_df)] = file_save_list[1]
    # save to csv
    time_df.to_csv(MEASURED_OUTPUT_DIR + "time_taken_to_write_graph.csv")
    size_df.to_csv(MEASURED_OUTPUT_DIR + "size_taken_to_write_graph.csv")

# TODO: use hash (first part of the filename) as id
# collect data during feature extraction: how long does it take to compute a feature
# save features in a csv for instance
# feature selection: correlation analysis,
#                   random forest inherently select features (or rate importance)
#


# identifier for the graph format? (opt)
# might have to unpack the binary
