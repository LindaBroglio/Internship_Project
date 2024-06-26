import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from local_ts_data.pair_grid import df_columns_as_float


def plot_correlation(df):
    df_columns_as_float(df)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

    upper_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # 10 features with the highest absolute correlation values
    print("Features with highest absolute correlation:")
    print(upper_corr.abs().unstack().sort_values(ascending=False)[:10])
    highest_corr_features = upper_corr.abs().unstack().sort_values(ascending=False)[:10].index
    highest_corr_features = ([f"{feature[0]}" for feature in highest_corr_features] +
                             [f"{feature[1]}" for feature in highest_corr_features])

    # 10 features with the lowest absolute correlation values
    print("\nFeatures with lowest absolute correlation:")
    print(upper_corr.abs().unstack().sort_values(ascending=True)[:10])
    lowest_corr_features = upper_corr.abs().unstack().sort_values(ascending=True)[:10].index
    lowest_corr_features = ([f"{feature[0]}" for feature in lowest_corr_features] +
                            [f"{feature[1]}" for feature in lowest_corr_features])

    return highest_corr_features, lowest_corr_features


def rank_corr_to_response(df, filename):  # returns only corr that have abs value > 0.1
    resp_corr_col = df.corr()['Response'][:-1]
    data = resp_corr_col.abs().sort_values(ascending=False)[1:]
    data = data.loc[data.values > 0.1]
    with open(filename, "w") as file:
        for i in range(data.index.shape[0]):
            file.write(f"{data.values[i]} <-- {data.index[i]}\n")
    return data.index


def subset_corr(df, wanted_features, vmin=None, vmax=None):
    subset_df = df[wanted_features]
    corr = subset_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Set the correlation values outside the specified range to None
    corr[(corr < vmin) | (corr > vmax)] = None

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.title('Correlation Heatmap of Given Features')
    plt.show()

