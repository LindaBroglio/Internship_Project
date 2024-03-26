from I_O_processing import get_in_out_folders_and_responses
from patients_data_processing import process_data
from responses_data_processing import patients_tuple


def main():
    input_dir, output_dir, real_responses_file = get_in_out_folders_and_responses()
    process_data(input_dir, output_dir)
    patients_tuple(input_dir, real_responses_file)


if __name__ == "__main__":
    main()
