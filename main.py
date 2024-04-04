from data_preprocessing.I_O_folders import get_in_out_folders_and_responses
from data_preprocessing.patients_data import process_data
from dummy_classifier import dummy_classifier


def main():
    input_folder, output_folder, real_responses_file = get_in_out_folders_and_responses()
    df_breast, df_lymphnode = process_data(input_folder, output_folder, real_responses_file)
    print(df_breast.columns)
    #dummy_classifier(df_breast)


if __name__ == "__main__":
    main()
