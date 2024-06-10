import os

from matplotlib import pyplot as plt
import seaborn as sns


def get_feature_list_given_word_filter(df, word_filter, inclusion):
    if inclusion:
        original_cols = df.columns
        filtered_list = [elem for elem in original_cols for word in word_filter if word in elem]
    else:
        original_cols = df.columns
        exclude_list = [elem for elem in original_cols for word in word_filter if word in elem]
        filtered_list = list(set(original_cols) - set(exclude_list))
    return filtered_list


def filter_df_by_word(dataframe, word_filter, inclusion):
    filtered_df = dataframe[get_feature_list_given_word_filter(dataframe, word_filter, inclusion)].copy()
    return filtered_df


def pair_grid_plot(og_df, word_filter, inclusion, pre_name, height=0.8, aspect=1.5, marker_size=8,
                   label_fontsize=8, label_rotation=30, save_path="pair_grid_plots/"):
    save_path = save_path+pre_name+"/"
    df = filter_df_by_word(og_df, word_filter, inclusion)
    g = sns.PairGrid(df, vars=df.columns, height=height, aspect=aspect, corner=True)
    g.map_diag(plt.hist)
    g.map_lower(sns.scatterplot, s=marker_size)
    g.add_legend()
    for ax in g.axes[-1, :]:
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize, rotation=label_rotation)
    for ax in g.axes[:, 0]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize, rotation=label_rotation)
    plt.tight_layout()
    if save_path:
        if inclusion: word_list = ""
        else: word_list = "not"
        for word in word_filter:
            word_list = word_list + "_" + word
        save_path = save_path + word_list + ".pdf"
        folder, filename = os.path.split(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path)
    else:
        plt.show()


def pair_grid_plot_all(df, pre_name):
    pair_grid_plot(df, ["GLCM", "NGLDM"], True, pre_name)
    pair_grid_plot(df, ["GLRLM"], True, pre_name)
    pair_grid_plot(df, ["GLZLM"], True, pre_name)
    pair_grid_plot(df, ["GLCM", "GLRLM", "NGLDM", "GLZLM"], False, pre_name)

