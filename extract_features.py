import os
import sys
import time
from typing import Final
import networkit as nk
import numpy as np
import pandas as pd

SRC_DIR: Final[str] = sys.argv[1]
FEATURE_OUTPUT_DIR: Final[str] = 'data/measured_data/'
FEATURE_FILE_ENDING: Final[str] = '_features'
TIME_FILE_ENDING: Final[str] = '_time'


# A connected component is a set of nodes in which each pair of nodes is connected by a path.
# returns two df: data_df & time_data_df
def calculate_components(given_graph):
    time_start = time.process_time()
    connected_components = nk.components.ConnectedComponents(given_graph)
    connected_components.run()
    time_connected_components = time.process_time() - time_start

    data_df = pd.DataFrame({'#ConnectedComponents': [connected_components.numberOfComponents()]})
    time_df = pd.DataFrame({'Time_#ConnectedComponents': [time_connected_components]})
    return data_df, time_df


# calculates centrality measures
# returns two df: data_df & time_data_df
def calculate_centrality(given_graph):
    time_start = time.process_time()
    deg_cen = nk.centrality.DegreeCentrality(given_graph)
    deg_cen.run()
    mean_deg_cen = np.mean(np.array(deg_cen.scores(), float))
    time_deg_cen = time.process_time() - time_start

    time_start = time.process_time()
    centralization = deg_cen.centralization()
    time_centralization = time.process_time() - time_start

    time_start = time.process_time()
    eigenvector_cen = nk.centrality.EigenvectorCentrality(given_graph)
    eigenvector_cen.run()
    mean_eigenvector_cen = np.mean(np.array(eigenvector_cen.scores(), float))
    time_eigenvector_cen = time.process_time() - time_start

    time_start = time.process_time()
    page_rank = nk.centrality.PageRank(given_graph)
    page_rank.run()
    mean_page_rank = np.mean(np.array(page_rank.scores(), float))
    time_page_rank = time.process_time() - time_start

    time_start = time.process_time()
    katz_cen = nk.centrality.KatzCentrality(given_graph)
    katz_cen.run()
    mean_katz_cen = np.mean(np.array(katz_cen.scores(), float))
    time_katz_cen = time.process_time() - time_start

    time_start = time.process_time()
    betweenness_cen = nk.centrality.EstimateBetweenness(given_graph, 50)  # (graph, nSamples)
    betweenness_cen.run()
    mean_betweenness_cen = np.mean(np.array(betweenness_cen.scores(), float))
    time_betweenness_cen = time.process_time() - time_start

    # putting it all into a df
    data = {
        'MeanDegreeCentrality': [mean_deg_cen],
        'Centralization': [centralization],
        'MeanEigenvectorCentrality': [mean_eigenvector_cen],
        'MeanPageRank': [mean_page_rank],
        'MeanKatzCentrality': [mean_katz_cen],
        'MeanBetweennessCentrality': [mean_betweenness_cen]
    }

    time_data = {
        'Time_MeanDegreeCentrality': [time_deg_cen],
        'Time_Centralization': [time_centralization],
        'Time_MeanEigenvectorCentrality': [time_eigenvector_cen],
        'Time_MeanPageRank': [time_page_rank],
        'Time_MeanKatzCentrality': [mean_katz_cen],
        'Time_MeanBetweennessCentrality': [time_betweenness_cen]
    }
    return pd.DataFrame(data), pd.DataFrame(time_data)


# calculates community measures
# returns a df
def calculate_community(given_graph):
    time_start = time.process_time()
    communities = nk.community.detectCommunities(given_graph)
    time_communities = time.process_time() - time_start

    time_start = time.process_time()
    num_communities = communities.numberOfElements()
    mean_community_size = np.mean(np.array(communities.subsetSizes(), float))
    time_mean_community_size = time_communities + time.process_time() - time_start

    min_community_size = min(communities.subsetSizes())
    max_community_size = max(communities.subsetSizes())
    # for simplicity’s sake, I will round up, such that min and max calculation
    # take up the same time as mean

    time_start = time.process_time()
    intercommunity_edge_weight = nk.community.EdgeCut().getQuality(communities, given_graph)
    time_intercommunity_edge_weight = time_communities + time.process_time() - time_start

    time_start = time.process_time()
    modularity = nk.community.Modularity().getQuality(communities, given_graph)
    time_modularity = time_communities + time.process_time() - time_start

    # putting it all into two df
    data = {
        '#Communities': [num_communities],
        'MinCommunitySize': [min_community_size],
        'MaxCommunitySize': [max_community_size],
        'MeanCommunitySize': [mean_community_size],
        'TotalInter-CommunityEdgeWeight': [intercommunity_edge_weight],
        'Modularity': [modularity]
    }

    time_data = {
        'Time_#Communities': [time_communities],
        'Time_MinCommunitySize': [time_mean_community_size],
        'Time_MaxCommunitySize': [time_mean_community_size],
        'Time_MeanCommunitySize': [time_mean_community_size],
        'Time_TotalInter-CommunityEdgeWeight': [time_intercommunity_edge_weight],
        'Time_Modularity': [time_modularity]
    }
    return pd.DataFrame(data), pd.DataFrame(time_data)


# calculates clustering coefficients
# returns a df
def calculate_clustering(given_graph):
    time_start = time.process_time()
    global_clustering_coefficient = nk.globals.clustering(given_graph)
    time_clustering = time.process_time() - time_start

    # putting it into two df
    data_df = pd.DataFrame({'GlobalClusteringCoefficient': [global_clustering_coefficient]})
    time_df = pd.DataFrame({'Time_GlobalClusteringCoefficient': [time_clustering]})
    return data_df, time_df


# script receives file path  of a METIS graph as input parameter
if __name__ == '__main__':
    # to strip ending of file path, e.g. '.cnf' or '.csv'
    file_name = os.path.basename(SRC_DIR)
    file_name_without_ending = os.path.splitext(file_name)[0]
    # we will use the file names for the id during the rest of the project
    current_id = file_name_without_ending.replace('_METIS', '')
    full_output_path_features = FEATURE_OUTPUT_DIR + current_id + FEATURE_FILE_ENDING
    full_output_path_time = FEATURE_OUTPUT_DIR + current_id + TIME_FILE_ENDING

    # read the graph from file
    metisReader = nk.graphio.METISGraphReader()
    graph = metisReader.read(SRC_DIR)
    # our feature df which should contain all features in the end
    feature_df = pd.DataFrame({'ID': [current_id]})
    # time df which tells us how long the computation of each feature took
    time_feature_df = pd.DataFrame({'ID': [current_id]})

    # calculate and concatenate features
    # connected components
    component_features, time_component_features = calculate_components(graph)

    # centrality measures
    centrality_features, time_centrality_features = calculate_centrality(graph)

    # community measures
    community_features, time_community_features = calculate_community(graph)

    # clustering coefficients
    clustering_features, time_clustering_features = calculate_clustering(graph)

    # put it all into corresponding dataframes
    feature_df = pd.concat([feature_df, component_features,
                            centrality_features, community_features], axis=1)
    feature_df.set_index('ID', inplace=True)

    time_feature_df = pd.concat([time_feature_df, time_component_features,
                                 time_centrality_features, time_community_features], axis=1)
    time_feature_df.set_index('ID', inplace=True)

    feature_df.to_csv(full_output_path_features)
    time_feature_df.to_csv(full_output_path_time)
    print(feature_df)
    print(time_feature_df)
