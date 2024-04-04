from data_preprocessing.I_O_folders import get_in_out_folders_and_responses
from data_preprocessing.patients_data import process_data
from data_preprocessing.responses_data import patients_dict, add_pat_response_column_and_create_final_dataset
from dummy_classifier import dummy_classifier


def main():
    input_dir, output_dir, real_responses_file = get_in_out_folders_and_responses()
    process_data(input_dir, output_dir)
    pat_response = patients_dict(input_dir, real_responses_file)
    final_df_breast, final_df_lymphnode = add_pat_response_column_and_create_final_dataset(pat_response, output_dir)
    dummy_classifier(final_df_breast)



if __name__ == "__main__":
    main()
