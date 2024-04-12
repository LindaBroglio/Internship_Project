import string
from itertools import combinations, combinations_with_replacement, permutations, product

from data_preprocessing.I_O_folders import get_in_out_folders_and_responses
from data_preprocessing.patients_data import process_data
from dummy_classifier import dummy_classifier, print_confusion_matrix
from feature_selection import initial_cleaning, random_feature_selection, create_cols_file_from_df


def main():
    input_folder, output_folder, real_responses_file = get_in_out_folders_and_responses()
    df_breast, df_lymphnode = process_data(input_folder, output_folder, real_responses_file)
    df_breast = initial_cleaning(df_breast)
    df_breast = random_feature_selection(df_breast, 50)
    #create_cols_file_from_df(df_breast, "random_features_selected")
    dummy_classifier(df_breast)

    #y =      [1, 0, 1, 0, 1, 0]  # TP, FP, FN, TN, FN, FP
    #y_pred = [1, 1, 0, 0, 0, 1]
    #print_confusion_matrix(y, y_pred)




if __name__ == "__main__":
    main()
