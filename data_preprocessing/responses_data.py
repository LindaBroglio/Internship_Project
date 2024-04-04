import os

import pandas as pd
from data_preprocessing.patients_data import get_patients_id


def patients_dict(input_dir, responses_file):
    responses_df = pd.read_excel(responses_file)
    patients = get_patients_id(input_dir)
    pat_response = {}
    for index, patient in enumerate(responses_df.iloc[:, 0].values):
        if patient in patients:
            pat_response[patient] = responses_df.iloc[index, 1]
    #print(pat_response)
    return pat_response


def add_pat_response_column_and_create_final_dataset(pat_response, output_dir, make_file=False):
    final_df_breast = pd.read_excel(os.path.join(output_dir, "Data_Breast.xlsx"), index_col=0)
    final_df_breast.insert(1, 'Response', final_df_breast['Patient'].map(pat_response))
    if make_file: final_df_breast.to_excel(os.path.join(output_dir, "Final_Data_Breast.xlsx"), index=True)

    final_df_lymphnode = pd.read_excel(os.path.join(output_dir, "Data_Lymphnode.xlsx"), index_col=0)
    final_df_lymphnode.insert(1, 'Response', final_df_lymphnode['Patient'].map(pat_response))
    if make_file: final_df_lymphnode.to_excel(os.path.join(output_dir, "Final_Data_Lymphnode.xlsx"), index=True)

    return final_df_breast, final_df_lymphnode
