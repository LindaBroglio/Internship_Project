import sys

from lung_data.correlation_analysis.pair_grid import pair_grid_plot, pair_grid_plot_all
from lung_data.data_preprocessing.importing import from_xls_into_df
from lung_data.data_preprocessing.pre_process import check_for_NaN


def main():
    df_PET, df_CT = from_xls_into_df(1)
    check_for_NaN(df_PET, "df_PET")
    check_for_NaN(df_CT, "df_CT")

    resultpet = df_PET.dtypes.value_counts().to_dict()
    print("\ninitial data types in df_PET:", resultpet)
    df_PET['Patient_ID'] = df_PET['Patient_ID'].astype('string')
    df_PET['Type'] = df_PET['Type'].astype('string')
    resultpet = df_PET.dtypes.value_counts().to_dict()
    print("final data types in df_PET:  ", resultpet)

    resultct = df_CT.dtypes.value_counts().to_dict()
    print("initial data types in df_CT: ", resultct)
    df_CT['Patient_ID'] = df_CT['Patient_ID'].astype('string')
    df_CT['Type'] = df_CT['Type'].astype('string')
    resultct = df_CT.dtypes.value_counts().to_dict()
    print("final data types in df_CT:   ", resultct)

    df_PET.drop(['Patient_ID', 'Type'], axis=1, inplace=True)
    df_CT.drop(['Patient_ID', 'Type'], axis=1, inplace=True)
    pair_grid_plot_all(df_PET, "PET")
    pair_grid_plot_all(df_CT, "CT")


if __name__ == "__main__":
    main()
