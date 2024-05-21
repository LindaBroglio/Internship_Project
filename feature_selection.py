from itertools import product


def remove_cols(df, cols, cols_format):
    if cols_format == "file":
        cols = get_cols_list_from_file(cols)
    elif cols_format != "list":
        raise ValueError("Unsupported format for the columns to remove. Please use 'file' or 'list'.")
    df = df.drop(cols, axis=1)
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


def remove_unique_value_features(df):
    not_unique = df.nunique()
    cols_to_drop = not_unique[not_unique == 1].index
    return remove_cols(df, cols_to_drop, "list")


def remove_the_word_original(df):
    df.columns = df.columns.str.replace('_original', "")
    return df


def initial_cleaning(df):
    create_cols_file_from_df(df, "original_cols")
    df_reduced = remove_WL_features(df)
    df_reduced = remove_the_word_original(df_reduced)
    create_cols_file_from_df(df_reduced, "filters_cols_removed")
    df_reduced = remove_unique_value_features(df_reduced)
    #create_cols_file_from_df(df_reduced, "no_unique_values_cols_removed")
    return df_reduced


def random_feature_selection(df, n):
    patients = df['Patient'].tolist()
    responses = df['Response'].tolist()
    df1 = df.drop('Patient', axis=1)
    dfnew = df1.drop('Response', axis=1)
    dfret = dfnew.sample(n, axis=1)
    dfret.insert(loc=0, column='Patient', value=patients)
    dfret.insert(loc=1, column='Response', value=responses)
    return dfret




