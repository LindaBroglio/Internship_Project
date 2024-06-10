import os
import argparse
import sys

import pandas as pd


def get_input_folder_and_filename():
    input_path, file_name = parse_arguments()
    check_in_folder(input_path)
    return input_path, file_name


def parse_arguments():
    parser = argparse.ArgumentParser(description="Get input directory")
    parser.add_argument("-i", "--input", type=str,
                        help="String indicating the file directory path", required=True)
    parser.add_argument("-f", "--filename", type=str,
                        help="String indicating file name", required=True)
    args = parser.parse_args()
    input_path = args.input
    file_name = args.filename
    return input_path, file_name


def check_in_folder(input_path):
    if not os.path.exists(input_path):
        sys.exit("Exiting program: invalid input directory")
    if len(os.listdir(input_path)) == 0:
        sys.exit("Exiting program: no files to process")
    print("Input directory:", input_path)


def from_xls_into_df(header):
    path, filename = get_input_folder_and_filename()
    sheet_names = ['PET', 'CT']
    dfs = pd.read_excel(path + "/" + filename, sheet_name=sheet_names, header=header)
    return dfs['PET'], dfs['CT']


