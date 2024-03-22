import re
import pandas as pd


def process_into_row(file):
    patient_name = file.split('_')[0]
    start_processing = False
    row_breast = {}
    row_lymphnode = {}
    row_breast['Patient'] = patient_name
    row_lymphnode['Patient'] = patient_name

    with open(file, "r") as file:
        for line in file:

            # check if it's ok to start reading
            if line.startswith('"original","shape","Elongation"'):
                start_processing = True

            # column name and cell input
            if start_processing:
                parts = line.strip().split(",")
                key = '_'.join(map(lambda x: re.sub(r'"', '', x), [parts[2], parts[1], parts[0]]))
                for i in range(3, len(parts) - 1, 2):
                    row_breast[key] = re.sub(r'"', '', parts[i])
                    row_lymphnode[key] = re.sub(r'"', '', parts[i + 1])

    return row_breast, row_lymphnode


def add_patient_row(patient_file, data_breast, data_lymphnode):
    row_breast, row_lymphnode = process_into_row(patient_file)
    data_breast.append(row_breast)
    data_lymphnode.append(row_lymphnode)


file1 = "ANON6477_filters.csv"
file2 = "RAD1_filters.csv"

data_breast = []
data_lymphnode = []

add_patient_row(file1, data_breast, data_lymphnode)
add_patient_row(file2, data_breast, data_lymphnode)

# separate df for mammella and linfonodo data
df_breast = pd.DataFrame(data_breast)
df_lymphnode = pd.DataFrame(data_lymphnode)

# create excel files
df_breast.index += 1
df_lymphnode.index += 1
df_breast.to_excel("Data_Breast.xlsx", index=True)
df_lymphnode.to_excel("Data_Lymphnode.xlsx", index=True)
