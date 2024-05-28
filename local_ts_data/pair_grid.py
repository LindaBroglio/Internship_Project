import os

from matplotlib import pyplot as plt
import seaborn as sns


def get_feature_list_given_word_filter(df, word_filter):
    original_cols = df.columns
    filtered_list = [elem for elem in original_cols if word_filter in elem]
    return filtered_list


def filter_df_by_word(dataframe, word_filter):
    filtered_df = dataframe[get_feature_list_given_word_filter(dataframe, word_filter)].copy()
    df_columns_as_float(filtered_df)
    return filtered_df


def df_columns_as_float(dataframe):
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].astype(float)


def pair_grid_plot(og_df, word_filter, height=0.8, aspect=1.5, marker_size=8,
                   label_fontsize=8, label_rotation=30, save_path="pair_grid_plots/"):
    df = filter_df_by_word(og_df, word_filter)
    df_columns_as_float(df)
    g = sns.PairGrid(df, vars=df.columns, height=height, aspect=aspect, corner=True)
    g.map_diag(plt.hist)
    g.map_lower(sns.scatterplot, s=marker_size)
    g.add_legend()
    # for ax in g.axes.flatten():
    #    ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize, rotation=label_rotation)
    #    ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize, rotation=label_rotation)
    for ax in g.axes[-1, :]:
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize, rotation=label_rotation)
    for ax in g.axes[:, 0]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize, rotation=label_rotation)
    plt.tight_layout()
    if save_path:
        save_path = save_path + word_filter + ".pdf"
        folder, filename = os.path.split(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path)
    else:
        plt.show()


def pair_grid_plot_all(df_breast_100):
    pair_grid_plot(df_breast_100, "shape")
    pair_grid_plot(df_breast_100, "firstorder")
    pair_grid_plot(df_breast_100, "glcm")
    pair_grid_plot(df_breast_100, "gldm")
    pair_grid_plot(df_breast_100, "glrlm")
    pair_grid_plot(df_breast_100, "glszm")
    pair_grid_plot(df_breast_100, "ngtdm")
