import os
import sys
from typing import Final
import networkit as nk
import numpy as np
import pandas as pd

SRC_DIR: Final[str] = sys.argv[1]
FEATURE_OUTPUT_DIR: Final[str] = 'data/measured_data/'
FEATURE_FILE_ENDING: Final[str] = '_features'


# A connected component is a set of nodes in which each pair of nodes is connected by a path.
def calculate_components(given_graph):
    connected_components = nk.components.ConnectedComponents(given_graph)
    connected_components.run()
    return pd.DataFrame({'#ConnectedComponents': [connected_components.numberOfComponents()]})


# calculates centrality measures
# returns a df
def calculate_centrality(given_graph):
    deg_cen = nk.centrality.DegreeCentrality(given_graph)
    deg_cen.run()
    centralization = deg_cen.centralization()
    mean_deg_cen = np.mean(np.array(deg_cen.scores(), float))
    eigenvector_cen = nk.centrality.EigenvectorCentrality(given_graph)
    eigenvector_cen.run()
    mean_eigenvector_cen = np.mean(np.array(eigenvector_cen.scores(), float))
    page_rank = nk.centrality.PageRank(given_graph)
    page_rank.run()
    mean_page_rank = np.mean(np.array(page_rank.scores(), float))
    katz_cen = nk.centrality.KatzCentrality(given_graph)
    katz_cen.run()
    mean_katz_cen = np.mean(np.array(katz_cen.scores(), float))
    betweenness_cen = nk.centrality.Betweenness(given_graph)
    betweenness_cen.run()
    mean_betweenness_cen = np.mean(np.array(betweenness_cen.scores(), float))

    # putting it all into a df
    data = {
        'Centralization': [centralization],
        'MeanDegreeCentrality': [mean_deg_cen],
        'MeanEigenvectorCentrality': [mean_eigenvector_cen],
        'MeanPageRank': [mean_page_rank],
        'MeanKatzCentrality': [mean_katz_cen],
        'MeanBetweennessCentrality': [mean_betweenness_cen]
    }
    return pd.DataFrame(data)


# calculates community measures
# returns a df
def calculate_community(given_graph):
    communities = nk.community.detectCommunities(given_graph)
    num_communities = communities.numberOfElements()
    min_community_size = min(communities.subsetSizes())
    max_community_size = max(communities.subsetSizes())
    mean_community_size = np.mean(np.array(communities.subsetSizes(), float))
    modularity = nk.community.Modularity().getQuality(communities, given_graph)
    # we could also calculate 'imbalance', 'edge cut'...

    # putting it all into a df
    data = {
        '#Communities': [num_communities],
        'MinCommunitySize': [min_community_size],
        'MaxCommunitySize': [max_community_size],
        'MeanCommunitySize': [mean_community_size],
        'Modularity': [modularity]
    }
    return pd.DataFrame(data)


# script receives file path  of a METIS graph as input parameter
if __name__ == '__main__':
    # to strip ending of file path, e.g. '.cnf' or '.csv'
    file_name = os.path.basename(SRC_DIR)
    file_name_without_ending = os.path.splitext(file_name)[0]
    # we will use the file names for the id during the rest of the project
    current_id = file_name_without_ending.replace('_METIS', '')
    full_output_path = FEATURE_OUTPUT_DIR + current_id + FEATURE_FILE_ENDING
    # read the graph from file
    metisReader = nk.graphio.METISGraphReader()
    graph = metisReader.read(SRC_DIR)
    # out feature df which should contain all features in the end
    feature_df = pd.DataFrame({'ID': [current_id]})

    # calculate and concatenate features
    # connected components
    component_features = calculate_components(graph)
    # centrality measures
    # centrality_features = calculate_centrality(graph)

    feature_df = pd.concat([feature_df, component_features, centrality_features], axis=1)
    feature_df.set_index('ID', inplace=True)
    feature_df.to_csv(full_output_path)
    print(feature_df)
