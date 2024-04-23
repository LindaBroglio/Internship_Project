from box_plot import plot_boxes
from correlation import plot_correlation, subset_corr
from data_preprocessing.I_O_folders import get_in_out_folders_and_responses
from data_preprocessing.patients_data import process_data
from feature_selection import initial_cleaning, random_feature_selection, create_cols_file_from_df
from gnb_classification import box_plot
from pair_grid import get_feature_list_given_word_filter, pair_grid_plot


def main():
    input_folder, output_folder, real_responses_file = get_in_out_folders_and_responses()
    df_breast, df_lymphnode = process_data(input_folder, output_folder, real_responses_file)
    #df_breast.drop('Patient', axis=1, inplace=True)

    df_breast_100 = initial_cleaning(df_breast)
    df_breast_100.drop('Patient', axis=1, inplace=True)
    df_breast_100.drop('Response', axis=1, inplace=True)
    create_cols_file_from_df(df_breast_100, "original_cols")

    df_breast_100_shape = df_breast_100[get_feature_list_given_word_filter(df_breast_100, "shape")]
    print(df_breast_100_shape.columns)
    pair_grid_plot(df_breast_100)
    create_cols_file_from_df(df_breast_100, "dfb_shape")

    #plot_boxes(df_breast, df_breast_cleaned)

    #highest_corr_features, lowest_corr_features = plot_correlation(df_breast_cleaned)
    #subset_corr(df_breast_cleaned, highest_corr_features, vmin=-1, vmax=-0.7)
    #subset_corr(df_breast_cleaned, lowest_corr_features, vmin=-0.075, vmax=0.075)


if __name__ == "__main__":
    main()
