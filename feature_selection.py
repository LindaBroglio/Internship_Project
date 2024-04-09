from itertools import product


def remove_cols(df, cols, cols_format):
    if cols_format == "file":
        cols = get_cols_list_from_file(cols)
    elif cols_format != "list":
        raise ValueError("Unsupported format for the columns to remove. Please use 'file' or 'list'.")
    df.drop(cols, axis=1, inplace=True)
    return df


def get_cols_list_from_file(columns_file):
    with open(columns_file) as col_file:
        col_list = [line.rstrip('\n') for line in col_file]
    return col_list


def create_cols_file_from_df(df, filename):
    with open(filename, "w") as outfile:
        outfile.write("\n".join(df.columns))


def remove_WL_features(df):
    original_cols = df.columns
    chars = ['L', 'H']
    unique_strings = set()
    for combination in product(chars, repeat=3):
        unique_strings.add(''.join(combination))
    filtered_list = [elem for elem in original_cols if any(substr in elem for substr in unique_strings)]
    return remove_cols(df, filtered_list, "list")

