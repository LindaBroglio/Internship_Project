from matplotlib import pyplot as plt
import seaborn as sns


def get_feature_list_given_word_filter(df, word_filter):
    original_cols = df.columns
    filtered_list = [elem for elem in original_cols if word_filter in elem]
    return filtered_list


def pair_grid_plot(df):
    #cols = [col for col in df.columns if 'shape' in col]
    #shape_df = df[cols]
    g = sns.PairGrid(df, vars=df.columns)
    #for ax in g.axes.flatten():
     #   ax.set_xlabel(ax.get_xlabel(), rotation=45)
      #  ax.set_ylabel(ax.get_ylabel(), rotation=45)

    #sns.set_context("paper", rc={"axes.labelsize": 10})

    g.map_diag(plt.hist)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    #g.set(yticklabels=[])
    #g.set(ylabel=None)
    plt.show()

