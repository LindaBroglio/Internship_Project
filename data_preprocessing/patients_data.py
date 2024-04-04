import re
import pandas as pd
import os

from data_preprocessing.responses_data import create_responses_dict, create_X_y_dataset


def process_into_row(file):
    patient_name = os.path.basename(file).split('_')[0]
    start_processing = False
    row_breast, row_lymphnode = initiate_rows(patient_name)
    with open(file, "r") as file:
        for line in file:
            # check if it's ok to start reading
            if line.startswith('"original","shape","Elongation"'):
                start_processing = True
            # column name and cell input
            if start_processing:
                process_line_of_raw_file(line, row_breast, row_lymphnode)
    return row_breast, row_lymphnode


def get_patients_id(input_folder):
    patients_id = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            patients_id.append(os.path.basename(filename).split('_')[0])
    return patients_id


def initiate_rows(patient_name):
    row_breast = {}
    row_lymphnode = {}
    row_breast['Patient'] = patient_name
    row_lymphnode['Patient'] = patient_name
    return row_breast, row_lymphnode


def process_line_of_raw_file(line, row_breast, row_lymphnode):
    parts = line.strip().split(",")
    key = '_'.join(map(lambda x: re.sub(r'"', '', x), [parts[2], parts[1], parts[0]]))
    for i in range(3, len(parts) - 1, 2):
        row_breast[key] = re.sub(r'"', '', parts[i])
        row_lymphnode[key] = re.sub(r'"', '', parts[i + 1])


def add_patient_row(patient_file, data_breast, data_lymphnode):
    row_breast, row_lymphnode = process_into_row(patient_file)
    data_breast.append(row_breast)
    data_lymphnode.append(row_lymphnode)


def from_csv_folder_into_df(input_folder):
    data_breast = []
    data_lymphnode = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            add_patient_row(file_path, data_breast, data_lymphnode)
    df_breast = pd.DataFrame(data_breast)
    df_lymphnode = pd.DataFrame(data_lymphnode)
    return df_breast, df_lymphnode


def create_df_xlsx_files(output_folder, df_breast, df_lymphnode):
    # create excel files
    os.makedirs(output_folder, exist_ok=True)
    df_breast.index += 1
    df_lymphnode.index += 1
    df_breast.to_excel(os.path.join(output_folder, "Data_Breast.xlsx"), index=True)
    df_lymphnode.to_excel(os.path.join(output_folder, "Data_Lymphnode.xlsx"), index=True)


def process_data(input_folder, output_folder, responses_file=None, make_file=False):
    df_breast, df_lymphnode = from_csv_folder_into_df(input_folder)
    if make_file:
        create_df_xlsx_files(output_folder, df_breast, df_lymphnode)
    if responses_file is None:
        return df_breast, df_lymphnode
    else:
        responses_dict = create_responses_dict(input_folder, responses_file, get_patients_id(input_folder))
        df_breast, df_lymphnode = create_X_y_dataset(responses_dict, df_breast, df_lymphnode, output_folder)
        return df_breast, df_lymphnode



