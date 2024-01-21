import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.linear_model import ElasticNet


def string_columns_to_float(df):
    # Convert all columns which are composed of strings to floats
    # input: df (dataframe)
    # output: none (modifies in-place)
    for colname in df.head():
        if any([isinstance(elem, str) for elem in df[colname]]):
            values = list(set(df[colname]))
            for i in range(len(values)):
                # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
                # df[colname][df[colname] == values[i]] = i+1
                df.loc[df[colname] == values[i], colname] = i+1
            # print(values)
            # print(df[colname])


def print_corr(df):
    # Print columns in df that are closely correlated
    # Input: df (a dataframe)
    # Output: none (printed)
    corr = df.corr()
    # corr is a dataframe
    # Print highly correlated columns
    corr_arr = corr.to_numpy()
    for row in range(corr_arr.shape[0]):
        # Get all columns in upper triangle whose abs is greater than 0.5
        high_corr_idxs = np.nonzero(np.abs(corr_arr[row]) > 0.5)[0]
        # print(high_corr_idxs)
        for idx in [i for i in high_corr_idxs if i < row]:  # Only need lower triangle
            print(corr_arr[row, idx], list(corr.head())[row], list(corr.head())[idx])

"""
Returns dataframe of cleaned training.csv
"""
def clean_data():
    arr = np.loadtxt("data/training.csv",
                     delimiter=",", dtype=str)

    df = pd.read_csv("data/training.csv")
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('pad_id', axis = 1)
    df = df.drop('frac_type', axis=1)
    ##df.dropna(subset = ['bh_x','bh_y'], inplace = True)
    df.dropna(subset = ['OilPeakRate'], inplace = True)

    string_columns_to_float(df)

    # for o in range(15):
    #     df.drop(df['OilPeakRate'].idxmax(), inplace=True)
    #     df.drop(df['OilPeakRate'].idxmin(), inplace=True)

    df_stages = df[~df["number_of_stages"].isna()]
    df = df[df["number_of_stages"].isna()]
    df = df.drop('number_of_stages', axis=1)
    df = df.drop('average_stage_length', axis=1)
    df = df.drop('average_proppant_per_stage', axis=1)
    df = df.drop('average_frac_fluid_per_stage', axis=1)
    df = df.dropna()

    # infs will cause models to fail
    # # https://stackoverflow.com/questions/21827594/raise-linalgerrorsvd-did-not-converge-linalgerror-svd-did-not-converge-in-m
    for column in df:
        df = df[~df[column].isin([np.inf])]

    return df

def is_independent(graph, vertex_set):
    """ Check if a set of vertices is an independent set in the graph """
    for vertex in vertex_set:
        for neighbor in graph[vertex]:
            if neighbor in vertex_set:
                return False
    return True

def largest_independent_set(graph, start, independent_set, max_set):
    """ Recursive backtracking function to find the largest independent set """
    if start == len(graph):
        if len(independent_set) > len(max_set[0]):
            max_set[0] = independent_set.copy()
        return

    # Include the vertex at 'start' in the set if it's independent
    if is_independent(graph, independent_set | {start}):
        largest_independent_set(graph, start + 1, independent_set | {start}, max_set)

    # Exclude the vertex at 'start' from the set
    largest_independent_set(graph, start + 1, independent_set, max_set)

def find_largest_independent_set(graph):
    """ Wrapper function to find the largest independent set """
    max_set = [set()]
    largest_independent_set(graph, 0, set(), max_set)
    return max_set[0]

def remove_correlations(df, base):
    df.drop("OilPeakRate", axis = 1)
    corr = df.corr()
    corr_arr = corr.to_numpy()



    graph = {}
    for row in range(corr_arr.shape[0]):
        graph[row] = []

    for row in range(corr_arr.shape[0]):
        for col in range(corr_arr.shape[1]):
            if col < row:
                if corr_arr[row,col] > base:
                    graph[col].append(row)
                    graph[row].append(col)


    independent_set = find_largest_independent_set(graph)

    # Iterate through all vertices in the graph
    for vertex in graph:
        # Check if the vertex is not adjacent to any vertex in the independent set
        if all(neighbour not in independent_set for neighbour in graph[vertex]):
            independent_set.add(vertex)


    selected = [list(corr.head())[col] for col in independent_set]
    print(selected)
    return df[selected]

