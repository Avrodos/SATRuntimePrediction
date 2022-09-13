import numpy as np
import pandas as pd

if __name__ == '__main__':
    # # load and merge both runtime files
    # all_new_runtimes_df = pd.read_csv('data/instances/prepared-anni-seq.csv', index_col=0)
    # all_new_runtimes_df.sort_index(inplace=True)
    # old_runtimes_df = pd.read_csv('data/runtimes/solver_runtimes.csv', index_col=0)
    # old_runtimes_df.sort_index(inplace=True)
    # merged_df = all_new_runtimes_df.join(old_runtimes_df, how='outer')
    #
    # # remove non-runtime columns
    # merged_df.drop(merged_df.iloc[:, 0:4], inplace=True, axis=1)
    # merged_df.to_csv('data/runtimes/extended_solver_runtimes.csv')

    # calculate new labels
    runtime_df = pd.read_csv('data/runtimes/extended_solver_runtimes.csv', index_col=0)
    old_runtimes_df = pd.read_csv('data/runtimes/solver_runtimes.csv', index_col=0)
    # a timeout happens at 5000s. parity 2 doubles its weight to punish a timeout
    runtime_df.replace('timeout', 10000, inplace=True)
    runtime_df = runtime_df.apply(pd.to_numeric)
    runtime_df['parity_two_label'] = runtime_df.mean(axis=1)
    runtime_df.sort_index(inplace=True)

    # calculate log of par2
    runtime_df['log10_parity_two_label'] = np.log10(runtime_df['parity_two_label'])

    # TODO: categorical KNN labels

    # drop non-label columns
    runtime_df = runtime_df[['parity_two_label', 'log10_parity_two_label']]

    # merge new and old feature dataframes
    new_feature_df = pd.read_csv('data/measured_data/new_SAT_features.csv', index_col=0)
    new_feature_df.sort_index(inplace=True)
    old_feature_df = pd.read_csv('data/measured_data/SAT_features.csv', index_col=0)
    obsolete_columns = ['MeanEigenvectorCentrality', 'MinEigenvectorCentrality', 'MaxEigenvectorCentrality',
                        'VarEigenvectorCentrality', 'EntropyEigenvectorCentrality']
    old_feature_df.drop(obsolete_columns, axis=1, inplace=True)
    old_feature_df.sort_index(inplace=True)
    feature_df = pd.concat([old_feature_df, new_feature_df]).drop_duplicates()

    # we only want to use instances, for which we have the features AND the runtime
    merged_df = pd.merge(feature_df, runtime_df, left_index=True, right_index=True)

    # split into two df's again and save
    runtime_df = merged_df[['parity_two_label', 'log10_parity_two_label']]
    feature_df = merged_df.drop(['parity_two_label', 'log10_parity_two_label'], axis=1)

    # save
    runtime_df.to_csv('data/runtimes/extended_runtime_labels.csv')
    feature_df.to_csv('data/measured_data/extended_SAT_features.csv')
    print()
