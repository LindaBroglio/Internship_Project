from correlation import plot_correlation, subset_corr
from data_preprocessing.I_O_folders import get_in_out_folders_and_responses
from data_preprocessing.patients_data import process_data
from feature_selection import initial_cleaning, random_feature_selection, create_cols_file_from_df
from pair_grid import pair_grid_plot, filter_df_by_word, df_columns_as_float


def main():
    input_folder, output_folder, real_responses_file = get_in_out_folders_and_responses()
    df_breast, df_lymphnode = process_data(input_folder, output_folder, real_responses_file)

    df_breast_100 = initial_cleaning(df_breast)
    df_breast_100.drop('Patient', axis=1, inplace=True)
    df_breast_100.drop('Response', axis=1, inplace=True)
    create_cols_file_from_df(df_breast_100, "original_cols")

    pair_grid_plot(df_breast_100, "shape")
    pair_grid_plot(df_breast_100, "firstorder")
    pair_grid_plot(df_breast_100, "glcm")
    pair_grid_plot(df_breast_100, "gldm")
    pair_grid_plot(df_breast_100, "glrlm")
    pair_grid_plot(df_breast_100, "glszm")
    pair_grid_plot(df_breast_100, "ngtdm")


if __name__ == "__main__":
    main()





    # highest_corr_features, lowest_corr_features = plot_correlation(df_breast_cleaned)
    # subset_corr(df_breast_cleaned, highest_corr_features, vmin=-1, vmax=-0.7)
    # subset_corr(df_breast_cleaned, lowest_corr_features, vmin=-0.075, vmax=0.075)


