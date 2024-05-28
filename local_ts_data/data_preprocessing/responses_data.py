import os
import pandas as pd


def create_responses_dict(input_dir, responses_file, patients):
    responses_df = pd.read_excel(responses_file)
    pat_response = {}
    for index, patient in enumerate(responses_df.iloc[:, 0].values):
        if patient in patients:
            pat_response[patient] = responses_df.iloc[index, 1]
    return pat_response


def create_X_y_dataset(pat_response, df_breast, df_lymphnode, output_dir, make_file=False):
    final_df_breast, final_df_lymphnode = add_responses_column(df_breast, df_lymphnode, pat_response)
    if make_file:
        final_df_breast.to_excel(os.path.join(output_dir, "Final_Data_Breast.xlsx"), index=True)
        final_df_lymphnode.to_excel(os.path.join(output_dir, "Final_Data_Lymphnode.xlsx"), index=True)
    return final_df_breast, final_df_lymphnode


def add_responses_column(df_breast, df_lymphnode, pat_response):
    df_breast.insert(1, 'Response', df_breast['Patient'].map(pat_response))
    df_lymphnode.insert(1, 'Response', df_lymphnode['Patient'].map(pat_response))
    return df_breast, df_lymphnode

