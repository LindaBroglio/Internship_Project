from I_O_processing import get_in_out_folders
from data_processing import process_data


def main():
    input_dir, output_dir = get_in_out_folders()
    #process_data(input_dir, output_dir)


if __name__ == "__main__":
    main()
