import os
from data_processing import from_input_folder_into_df, from_df_into_output_folder


def main(input_folder, output_folder=None):
    df_breast, df_lymphnode = from_input_folder_into_df(input_folder)
    if output_folder is None:
        output_folder = os.path.join(os.getcwd(), "output")
    from_df_into_output_folder(output_folder, df_breast, df_lymphnode)


if __name__ == "__main__":
    input_folder = input("Enter the input folder path: ")
    output_folder = input("Enter the output folder path (press Enter for default): ").strip()
    main(input_folder, output_folder or None)


