# with this file I want to prepare the CNF for feature extraction without having to downlooad them
# This is done by using referenced paths in a db

import sqlite3

import pandas as pd

if __name__ == '__main__':
    # open connection for meta data
    meta_cnx = sqlite3.connect('data/meta_data/meta.db')
    meta_df = pd.read_sql_query("SELECT * FROM local", meta_cnx)
    meta_df.set_index('hash', inplace=True)
    meta_df.rename({'value': 'path'}, axis='columns', inplace=True)
    # drop duplicates
    meta_df = meta_df[~meta_df.index.duplicated(keep='first')]
    # close connection again
    meta_cnx.commit()
    meta_cnx.close()

    # same for runtime data
    runtime_df = pd.read_csv('data/instances/anni-seq.csv', delim_whitespace=True)
    runtime_df.set_index('hash', inplace=True)

    # we only want to use instances, for which we have the file AND the runtime
    merged_df = pd.merge(meta_df, runtime_df, left_index=True, right_index=True)
    merged_df.to_csv('data/instances/prepared-anni-seq.csv')
    print(runtime_df)
