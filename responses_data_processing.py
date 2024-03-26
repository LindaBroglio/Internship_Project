import pandas as pd
from patients_data_processing import get_patients_id


def patients_tuple(input_dir, responses_file):
    responses_df = pd.read_excel(responses_file)
    patients = get_patients_id(input_dir)
    pat_response = {}
    for index, patient in enumerate(responses_df.iloc[:, 0].values):
        if patient in patients:
            pat_response[patient] = responses_df.iloc[index, 1]
    print(pat_response)




