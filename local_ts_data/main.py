from local_ts_data.box_plot import plot_boxes
from local_ts_data.data_preprocessing.I_O_folders import get_in_out_folders_and_responses
from local_ts_data.data_preprocessing.patients_data import process_data
from local_ts_data.feature_selection import initial_cleaning, remove_cols


def main():
    input_folder, output_folder, real_responses_file = get_in_out_folders_and_responses()
    df_breast, df_lymphnode = process_data(input_folder, output_folder, real_responses_file)

    df_breast_100 = initial_cleaning(df_breast)

    responses = df_breast['Response'].values

    df_breast_100.drop('Patient', axis=1, inplace=True)
    # df_breast_100.drop('Response', axis=1, inplace=True)
    df_breast.drop('Patient', axis=1, inplace=True)
    #df_breast.drop('Response', axis=1, inplace=True)
    # mutual_info_response(df_breast, responses, "mutual_info_response_complete")
    # mutual_info_response(df_breast_100, responses, "mutual_info_response_100")

    # create_cols_file_from_df(df_breast_100, "original_cols")
    # pair_grid_plot_all(df_breast_100)

    df_breast_65 = remove_cols(df_breast_100, "selection_to_remove", "file")

    # mutual_info_matrix(df_breast_65, "mutual_info_features_65")

    #feature_correlation = rank_corr_to_response(df_breast_65, "correlation_response_65")
    #features_mutual_info = rank_mutual_info_response(df_breast_65, responses, "mutual_info_response_65")
    #list_12 = list(set(feature_correlation).intersection(features_mutual_info))
    with open('12_cols', 'r') as file:
        list_12 = file.read().splitlines()
    df_breast_12 = df_breast_65[list_12]

    plot_boxes(df_breast_100, df_breast_12)


if __name__ == "__main__":
    main()

    # highest_corr_features, lowest_corr_features = plot_correlation(df_breast_cleaned)
    # subset_corr(df_breast_cleaned, highest_corr_features, vmin=-1, vmax=-0.7)
    # subset_corr(df_breast_cleaned, lowest_corr_features, vmin=-0.075, vmax=0.075)


