import string
from itertools import combinations, combinations_with_replacement, permutations, product

from data_preprocessing.I_O_folders import get_in_out_folders_and_responses
from data_preprocessing.patients_data import process_data
from dummy_classifier import dummy_classifier
from feature_selection import remove_WL_features, create_cols_file_from_df


def main():
    input_folder, output_folder, real_responses_file = get_in_out_folders_and_responses()
    df_breast, df_lymphnode = process_data(input_folder, output_folder, real_responses_file)
    # create_temp_cols_file(df_breast, "original_cols")
    df_breast_reduced = remove_WL_features(df_breast)
    create_cols_file_from_df(df_breast, "reduced_cols")

    #dummy_classifier(df_breast)




if __name__ == "__main__":
    main()
