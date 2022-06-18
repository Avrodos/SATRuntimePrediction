import os
import sys
import time
from math import e, log
from typing import Final
import networkit as nk
import numpy as np
import pandas as pd
import networkx as nx

SRC_DIR: Final[str] = sys.argv[1]
FEATURE_OUTPUT_DIR: Final[str] = 'data/measured_data/'
FEATURE_FILE_ENDING: Final[str] = '_features'
TIME_FILE_ENDING: Final[str] = '_time'


# helpers:
# Computes entropy of array distribution.
# fastest entropy calculation taken from:
# https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
def calculate_entropy(given_array, base=None):
    len_array = len(given_array)

    if len_array <= 1:
        return 0

    value, counts = np.unique(given_array, return_counts=True)
    probs = counts / len_array
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


# A connected component is a set of nodes in which each pair of nodes is connected by a path.
# returns two df: data_df & time_data_df
def calculate_components(given_graph):
    time_start = time.process_time()
    num_connected_components = nk.components.ConnectedComponents(given_graph).run().numberOfComponents()
    time_connected_components = time.process_time() - time_start

    data_df = pd.DataFrame({'#ConnectedComponents': [num_connected_components]})
    time_df = pd.DataFrame({'Time_#ConnectedComponents': [time_connected_components]})
    return data_df, time_df


# TODO: graph core

# calculates centrality measures
# a distribution is measured using 5 stats:
# mean, min, max, variance & entropy
# returns two df: data_df & time_data_df
def calculate_centrality(given_graph):
    time_start = time.process_time()
    deg_cen = nk.centrality.DegreeCentrality(given_graph)
    deg_cen.run()

    # Degree Centrality
    np_array_deg_cen = np.array(deg_cen.scores(), float)
    mean_deg_cen = np.mean(np_array_deg_cen)
    min_deg_cen = np.min(np_array_deg_cen)
    max_deg_cen = np.max(np_array_deg_cen)
    var_deg_cen = np.var(np_array_deg_cen)
    entropy_deg_cen = calculate_entropy(np_array_deg_cen)
    time_deg_cen = time.process_time() - time_start

    # Degree Assortativity
    time_start = time.process_time()
    deg_assortativity = nk.correlation.Assortativity(given_graph, deg_cen.scores()).run().getCoefficient()
    time_deg_assortativity = time.process_time() + time_deg_cen - time_start

    # Centralization
    time_start = time.process_time()
    centralization = deg_cen.centralization()
    time_centralization = time.process_time() - time_start

    # Eigenvector Centrality
    time_start = time.process_time()
    eigenvector_cen = nk.centrality.EigenvectorCentrality(given_graph)
    eigenvector_cen.run()
    np_array_eigenvector_cen = np.array(eigenvector_cen.scores(), float)
    mean_eigenvector_cen = np.mean(np_array_eigenvector_cen)
    min_eigenvector_cen = np.min(np_array_eigenvector_cen)
    max_eigenvector_cen = np.max(np_array_eigenvector_cen)
    var_eigenvector_cen = np.var(np_array_eigenvector_cen)
    entropy_eigenvector_cen = calculate_entropy(np_array_eigenvector_cen)
    time_eigenvector_cen = time.process_time() - time_start

    # Page Rank
    time_start = time.process_time()
    page_rank = nk.centrality.PageRank(given_graph)
    page_rank.run()
    np_array_page_rank = np.array(page_rank.scores(), float)
    mean_page_rank = np.mean(np_array_page_rank)
    min_page_rank = np.min(np_array_page_rank)
    max_page_rank = np.max(np_array_page_rank)
    var_page_rank = np.var(np_array_page_rank)
    entropy_page_rank = calculate_entropy(np_array_page_rank)
    time_page_rank = time.process_time() - time_start

    # Katz Centrality
    time_start = time.process_time()
    katz_cen = nk.centrality.KatzCentrality(given_graph)
    katz_cen.run()
    np_array_katz_cen = np.array(katz_cen.scores(), float)
    mean_katz_cen = np.mean(np_array_katz_cen)
    min_katz_cen = np.min(np_array_katz_cen)
    max_katz_cen = np.max(np_array_katz_cen)
    var_katz_cen = np.var(np_array_katz_cen)
    entropy_katz_cen = calculate_entropy(np_array_katz_cen)
    time_katz_cen = time.process_time() - time_start

    # Betweenness Centrality
    time_start = time.process_time()
    betweenness_cen = nk.centrality.EstimateBetweenness(given_graph, 50)  # (graph, nSamples)
    betweenness_cen.run()
    np_array_betweenness_cen = np.array(betweenness_cen.scores(), float)
    mean_betweenness_cen = np.mean(np_array_betweenness_cen)
    min_betweenness_cen = np.min(np_array_betweenness_cen)
    max_betweenness_cen = np.max(np_array_betweenness_cen)
    var_betweenness_cen = np.var(np_array_betweenness_cen)
    entropy_betweenness_cen = calculate_entropy(np_array_betweenness_cen)
    time_betweenness_cen = time.process_time() - time_start

    # putting it all into a df
    data = {
        'MeanDegreeCentrality': [mean_deg_cen],
        'MinDegreeCentrality': [min_deg_cen],
        'MaxDegreeCentrality': [max_deg_cen],
        'VarDegreeCentrality': [var_deg_cen],
        'EntropyDegreeCentrality': [entropy_deg_cen],

        'DegreeAssortativity': [deg_assortativity],
        'Centralization': [centralization],

        'MeanEigenvectorCentrality': [mean_eigenvector_cen],
        'MinEigenvectorCentrality': [min_eigenvector_cen],
        'MaxEigenvectorCentrality': [max_eigenvector_cen],
        'VarEigenvectorCentrality': [var_eigenvector_cen],
        'EntropyEigenvectorCentrality': [entropy_eigenvector_cen],

        'MeanPageRank': [mean_page_rank],
        'MinPageRank': [min_page_rank],
        'MaxPageRank': [max_page_rank],
        'VarPageRank': [var_page_rank],
        'EntropyPageRank': [entropy_page_rank],

        'MeanKatzCentrality': [mean_katz_cen],
        'MinKatzCentrality': [min_katz_cen],
        'MaxKatzCentrality': [max_katz_cen],
        'VarKatzCentrality': [var_katz_cen],
        'EntropyKatzCentrality': [entropy_katz_cen],

        'MeanBetweennessCentrality': [mean_betweenness_cen],
        'MinBetweennessCentrality': [min_betweenness_cen],
        'MaxBetweennessCentrality': [max_betweenness_cen],
        'VarBetweennessCentrality': [var_betweenness_cen],
        'EntropyBetweennessCentrality': [entropy_betweenness_cen]
    }

    time_data = {
        'Time_DegreeCentrality': [time_deg_cen],
        'Time_DegreeAssortativity': [time_deg_assortativity],
        'Time_Centralization': [time_centralization],
        'Time_EigenvectorCentrality': [time_eigenvector_cen],
        'Time_PageRank': [time_page_rank],
        'Time_KatzCentrality': [time_katz_cen],
        'Time_BetweennessCentrality': [time_betweenness_cen]
    }
    return pd.DataFrame(data), pd.DataFrame(time_data)


# calculates community measures
# returns two df: data_df & time_data_df
def calculate_community(given_graph):
    # partition into communities
    time_start = time.process_time()
    communities = nk.community.detectCommunities(given_graph)
    time_communities = time.process_time() - time_start

    # Community Measurements
    time_start = time.process_time()
    num_communities = communities.numberOfElements()
    np_array_community_size = np.array(communities.subsetSizes(), float)
    mean_community_size = np.mean(np_array_community_size)
    min_community_size = np.min(np_array_community_size)
    max_community_size = np.max(np_array_community_size)
    var_community_size = np.var(np_array_community_size)
    entropy_community_size = calculate_entropy(np_array_community_size)
    time_community_size = time_communities + time.process_time() - time_start

    # Graph Imbalance
    time_start = time.process_time()
    imbalance = nk.community.GraphClusteringTools().getImbalance(communities)
    time_imbalance = time_communities + time.process_time() - time_start

    # Check whether community partition is a proper clustering
    time_start = time.process_time()
    is_proper_clustering = nk.community.GraphClusteringTools().isProperClustering(given_graph, communities)
    time_is_proper_clustering = time_communities + time.process_time() - time_start

    # Check whether community partition is a singleton clustering
    time_start = time.process_time()
    is_singleton_clustering = nk.community.GraphClusteringTools().isSingletonClustering(given_graph, communities)
    time_is_singleton_clustering = time_communities + time.process_time() - time_start

    # Check whether community partition is a One Clustering
    time_start = time.process_time()
    is_one_clustering = nk.community.GraphClusteringTools().isOneClustering(given_graph, communities)
    time_is_one_clustering = time_communities + time.process_time() - time_start

    # Edge cut = total weight of intercommunity edges, Edge cut fraction = the fraction compared to total num of edges
    time_start = time.process_time()
    edge_cut = nk.community.EdgeCut().getQuality(communities, given_graph)
    edge_cut_fraction = edge_cut / given_graph.numberOfEdges()
    time_edge_cut = time_communities + time.process_time() - time_start

    # Community Modularity
    time_start = time.process_time()
    modularity = nk.community.Modularity().getQuality(communities, given_graph)
    time_modularity = time_communities + time.process_time() - time_start

    # Hub Dominance
    time_start = time.process_time()
    hub_dominance = nk.community.HubDominance().getQuality(communities, given_graph)
    time_hub_dominance = time_communities + time.process_time() - time_start

    # Intrapartition Density
    time_start = time.process_time()
    global_intrapartition_density = nk.community.IntrapartitionDensity(given_graph, communities).run().getGlobal()
    time_intrapartition_density = time_communities + time.process_time() - time_start

    # putting it all into two df
    data = {
        '#Communities': [num_communities],
        'MeanCommunitySize': [mean_community_size],
        'MinCommunitySize': [min_community_size],
        'MaxCommunitySize': [max_community_size],
        'VarCommunitySize': [var_community_size],
        'EntropyCommunitySize': [entropy_community_size],
        'ClusterImbalance': [imbalance],
        'IsProperClustering': [is_proper_clustering],
        'IsSingletonClustering': [is_singleton_clustering],
        'IsOneClustering': [is_one_clustering],
        'EdgeCut': [edge_cut],
        'EdgeCutFraction': [edge_cut_fraction],
        'Modularity': [modularity],
        'HubDominance': [hub_dominance],
        'GlobalIntrapartitionDensity': [global_intrapartition_density]
    }

    time_data = {
        'Time_#Communities': [time_communities],
        'Time_CommunitySize': [time_community_size],
        'Time_ClusterImbalance': [time_imbalance],
        'Time_IsProperClustering': [time_is_proper_clustering],
        'Time_IsSingletonClustering': [time_is_singleton_clustering],
        'Time_IsOneClustering': [time_is_one_clustering],
        'Time_EdgeCut': [time_edge_cut],
        'Time_Modularity': [time_modularity],
        'Time_HubDominance': [time_hub_dominance],
        'Time_GlobalIntrapartitionDensity': [time_intrapartition_density]
    }
    return pd.DataFrame(data), pd.DataFrame(time_data)


# calculates clustering coefficients
# note: some clustering measurements depend on a partition and are therefore calculated in 'calculate_community'
# returns two df: data_df & time_data_df
def calculate_clustering(given_graph):
    time_start = time.process_time()
    global_clustering_coefficient = nk.globals.clustering(given_graph)
    time_clustering = time.process_time() - time_start

    # putting it into two df
    data_df = pd.DataFrame({'GlobalClusteringCoefficient': [global_clustering_coefficient]})
    time_df = pd.DataFrame({'Time_GlobalClusteringCoefficient': [time_clustering]})
    return data_df, time_df


# calculates the diameter measures of the largest connected component in the graph
# returns two df: data_df & time_data_df
def calculate_diameters(given_graph):
    time_start = time.process_time()
    # if the graph is not connected, diameter would be inf
    # therefore extract largest connected component first
    largest_con_comp = nk.components.ConnectedComponents.extractLargestConnectedComponent(given_graph, True)

    # Compute the diameter of the input graph
    diameter = nk.distance.Diameter(largest_con_comp).run().getDiameter()
    time_diameter = time.process_time() - time_start

    # approximates effective diameter, which is defined as the number of edges on average to reach a
    # given ratio (def: 0.9) of all other nodes
    time_start = time.process_time()
    effective_diameter_approx = nk.distance.EffectiveDiameterApproximation(largest_con_comp).run().getEffectiveDiameter()
    time_effective_diameter_approx = time.process_time() - time_start

    # putting it into two df
    data_df = pd.DataFrame({'Diameter': [diameter], 'EffectiveDiameterApproximation': [effective_diameter_approx]})
    time_df = pd.DataFrame({'Time_Diameter': [time_diameter],
                            'Time_EffectiveDiameterApproximation': [time_effective_diameter_approx]})
    return data_df, time_df


# Takes too long for current purpose
def calculate_small_world(given_graph):
    time_start = time.process_time()
    nx_graph = nk.nxadapter.nk2nx(given_graph)  # convert from NetworKit.Graph to networkx.Graph
    time_graph_conversion = time.process_time() - time_start

    time_start = time.process_time()
    # graph is commonly classified as small-world for sigma > 1
    sigma = nx.sigma(nx_graph)
    time_sigma = time_graph_conversion + time.process_time() - time_start

    time_start = time.process_time()
    # graph is commonly classified as small-world for omega close to 0
    omega = nx.omega(nx_graph)
    time_omega = time_graph_conversion + time.process_time() - time_start

    # putting it into two df
    data_df = pd.DataFrame({'SmallWorldSigma': [], 'SmallWorldOmega': [omega]})
    time_df = pd.DataFrame({'Time_SmallWorldSigma': [], 'Time_SmallWorldOmega': [time_omega]})
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
    main_time_start = time.process_time()
    metisReader = nk.graphio.METISGraphReader()
    graph = metisReader.read(SRC_DIR)
    num_nodes = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()
    time_read_graph = time.process_time() - main_time_start
    # our feature df which should contain all features in the end
    feature_df = pd.DataFrame({'ID': [current_id], '#Nodes': [num_nodes], '#Edges': [num_edges]})
    # time df which tells us how long the computation of each feature took
    time_feature_df = pd.DataFrame({'ID': [current_id], 'Time_ReadGraph': [time_read_graph]})

    # calculate and concatenate features
    # connected components
    component_features, time_component_features = calculate_components(graph)

    # centrality measures
    centrality_features, time_centrality_features = calculate_centrality(graph)

    # community measures
    community_features, time_community_features = calculate_community(graph)

    # clustering coefficients
    clustering_features, time_clustering_features = calculate_clustering(graph)

    # graph diameters
    diameter_features, time_diameter_features = calculate_diameters(graph)

    # small-world measures
    # small_world_features, time_small_world_features = calculate_small_world(graph)

    # put it all into corresponding dataframes
    feature_df = pd.concat([feature_df, component_features,
                            centrality_features, community_features, clustering_features, diameter_features], axis=1)
    feature_df.set_index('ID', inplace=True)

    time_feature_df = pd.concat(
        [time_feature_df, time_component_features, time_centrality_features, time_community_features,
         time_clustering_features, time_diameter_features], axis=1)
    time_feature_df.set_index('ID', inplace=True)

    feature_df.to_csv(full_output_path_features)
    time_feature_df.to_csv(full_output_path_time)
    print(feature_df)
    print(time_feature_df)
