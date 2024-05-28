import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif

from pair_grid import df_columns_as_float


def rank_mutual_info_response(dataframe, responses, filename):  # returns all features bigger than 1/100 of the max
    dataframe.drop('Response', axis=1, inplace=True)
    mutual_info_scores = mutual_info_classif(dataframe, responses)
    column_scores = [(dataframe.columns[i], mutual_info_scores[i]) for i in range(65)]
    column_scores.sort(key=lambda x: x[1], reverse=True)
    max_value = max(column_scores, key=lambda x: x[1])[1]
    features_to_return = []
    with open(filename, "w") as file:
        for column, score in column_scores:
            if score > (max_value/100):
                file.write(f"{score} <-- {column}\n")
                features_to_return.append(column)
    return features_to_return


def joint_entropies(data, nbins=2):
    n_variables = data.shape[-1]
    n_samples = data.shape[0]
    if nbins is None:
        nbins = int((n_samples/5)**.5)
    histograms2d = np.zeros((n_variables, n_variables, nbins, nbins))
    for i in range(n_variables):
        for j in range(n_variables):
            histograms2d[i,j] = np.histogram2d(data[:,i], data[:,j], bins=nbins)[0]
    probs = histograms2d / len(data) + 1e-100
    joint_entropies = -(probs * np.log2(probs)).sum((2,3))
    return joint_entropies


def mutual_info_matrix(df,  filename, nbins=2, normalized=True):
    df_columns_as_float(df)
    data = df.to_numpy()
    n_variables = data.shape[-1]
    j_entropies = joint_entropies(data, nbins)
    entropies = j_entropies.diagonal()
    entropies_tile = np.tile(entropies, (n_variables, 1))
    sum_entropies = entropies_tile + entropies_tile.T
    mi_matrix = sum_entropies - j_entropies
    if normalized:
        mi_matrix = mi_matrix * 2 / sum_entropies
    row_names = df.columns
    col_names = df.columns
    flattened_matrix = []
    for i, row in enumerate(mi_matrix):
        for j, value in enumerate(row):
            if j > i:
                flattened_matrix.append((value, row_names[i], col_names[j]))
    sorted_matrix = sorted(flattened_matrix, key=lambda x: x[0])
    counter = 0
    for value, row_name, col_name in sorted_matrix:
        counter = counter+1
        print({counter}, f" - Value: {value} <-- {row_name}, {col_name}")
    return mi_matrix


def mutual_info_features(dataframe, plot_filename, values_filename):
    mutual_info_features_matrix = mutual_info_matrix(dataframe)
    plt.figure(figsize=(10, 10))
    sns.heatmap(mutual_info_features_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Mutual Information Heatmap', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.xticks(rotation=90, fontsize=6, ha='right')
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(plot_filename)

