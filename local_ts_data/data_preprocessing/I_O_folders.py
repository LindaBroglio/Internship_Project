import os
import argparse
import sys


def get_in_out_folders_and_responses():
    input_path, output_path, real_responses_file = parse_arguments()
    check_in_folder(input_path)
    output_path = check_out_folder(output_path)
    check_real_responses_file(real_responses_file)
    return input_path, output_path, real_responses_file


def parse_arguments():
    parser = argparse.ArgumentParser(description="Get I/O directories and real responses file path.")
    parser.add_argument("-i", "--input", type=str,
                        help="String indicating the patients files directory path", required=True)
    parser.add_argument("-o", "--output", type=str,
                        help="String indicating the output directory path", required=False)
    parser.add_argument("-r", "--response", type=str,
                        help="String indicating the real responses file path", required=True)
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    response_file = args.response
    return input_path, output_path, response_file


def check_in_folder(input_path):
    if not os.path.exists(input_path):
        sys.exit("Exiting program: invalid input directory")
    if len(os.listdir(input_path)) == 0:
        sys.exit("Exiting program: no files to process")
    print("Input directory:", input_path)


def check_out_folder(output_path):
    if output_path is not None:
        print("Output directory:", output_path)
    else:
        output_path = make_default_out_folder()
    return output_path


def make_default_out_folder():
    output_path = os.path.join(os.getcwd(), "raw_data", "output_default_folder")
    print("No output directory given, results will be at", output_path)
    return output_path


def check_real_responses_file(real_responses_file):
    if not os.path.isfile(real_responses_file):
        sys.exit("Exiting program: the given responses file does not exist")
    print("Real responses file:", real_responses_file)
