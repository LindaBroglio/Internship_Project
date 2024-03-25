import os
import argparse
import sys


def get_in_out_folders():
    input_path, output_path = parse_arguments()
    check_in_folder(input_path)
    if output_path is not None:
        print("Output directory:", output_path)
    else:
        output_path = make_default_out_folder()
    return input_path, output_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process input and output directories.")
    parser.add_argument("-i", "--input", type=str,
                        help="String indicating the patients files directory path", required=True)
    parser.add_argument("-o", "--output", type=str,
                        help="String indicating the output directory path", required=False)
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    return input_path, output_path


def check_in_folder(input_path):
    print("Input directory:", input_path)
    if not os.path.exists(input_path):
        sys.exit("Exiting program: invalid input directory")
    if len(os.listdir(input_path)) == 0:
        sys.exit("Exiting program: no files to process")


def make_default_out_folder():
    output_path = os.path.join(os.getcwd(), "data", "output_default_folder")
    print("No output directory given, results will be at", output_path)
    return output_path
