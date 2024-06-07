import sys
import pandas as pd
from lung_data.LDA import LDA_classifier, lda_bootstrap_multiclass, lda_bootstrap_binary, write_results_to_file
from lung_data.correlation_analysis.anova import get_anova_test
from lung_data.correlation_analysis.pair_grid import pair_grid_plot, pair_grid_plot_all
from lung_data.data_preprocessing.importing import from_xls_into_df
from lung_data.data_preprocessing.pre_process import check_for_NaN
from lung_data.other_models import plot_boxes


def main():
    df_PET, df_CT = from_xls_into_df(1)

   # check_for_NaN(df_PET, "df_PET")
   # check_for_NaN(df_CT, "df_CT")

    resultpet = df_PET.dtypes.value_counts().to_dict()
   # print("\ninitial data types in df_PET:", resultpet)
    df_PET['Patient_ID'] = df_PET['Patient_ID'].astype('string')
    df_PET['Type'] = df_PET['Type'].astype('string')
    resultpet = df_PET.dtypes.value_counts().to_dict()
   # print("final data types in df_PET:  ", resultpet)

    resultct = df_CT.dtypes.value_counts().to_dict()
   # print("initial data types in df_CT: ", resultct)
    df_CT['Patient_ID'] = df_CT['Patient_ID'].astype('string')
    df_CT['Type'] = df_CT['Type'].astype('string')
    resultct = df_CT.dtypes.value_counts().to_dict()
   # print("final data types in df_CT:   ", resultct)

    # df_PET.drop(['Patient_ID', 'Type'], axis=1, inplace=True)
    # df_CT.drop(['Patient_ID', 'Type'], axis=1, inplace=True)
    # pair_grid_plot_all(df_PET, "PET")
    # pair_grid_plot_all(df_CT, "CT")

    df_PET.drop('Patient_ID', axis=1, inplace=True)
    df_CT.drop('Patient_ID', axis=1, inplace=True)

    #irrelevant_pet = get_anova_test(df_PET, 'Type')
    #irrelevant_ct = get_anova_test(df_CT, 'Type')

    #print('\nPET least relevant features for anova:\n', irrelevant_pet)
    #print('\nCT least relevant features for anova:\n', irrelevant_ct)

    #combined_results = pd.concat([irrelevant_pet.reset_index(drop=True), irrelevant_ct.reset_index(drop=True)], axis=1)
    #print(combined_results)
    #df_PET.drop(['EnergyH', 'Energy_GLCM', 'LGR_GLRLM', 'SRLGE_GLRLM', 'LRLGE_GLRLM',
                 #'LGZE_GLZLM', 'LZLGE_GLZLM', 'LZHGE_GLZLM'], axis=1, inplace=True)


    #results = lda_bootstrap_multiclass(df_PET, 'Type', n_iterations=100)
    #results = lda_bootstrap_binary(df_PET, 'Type', 'Binary_Type', 100, 'plot_shaded_smote_reduced')
    #write_results_to_file(df_PET, results, 'results_smote_reduced.txt')
    plot_boxes(df_PET, 'Type', 'Binary_Type')

    """   tot_patients = len(df_PET['Binary_Type'])
    tot_met_patients = len(df_PET[df_PET['Binary_Type'] == "met"])

    print(f'\nTotal number of patients: {tot_patients}, of which {tot_met_patients} with metastasis')
    print(f'\nPatients in training set: {results["tot_train"]}')
    print(f'Patients in test set:      {results["tot_test"]}')

    print(f'\nMetastasis patients counter in:\n'
          f' - Training set              {results["mean_met_train"]:.4f} ± {results["std_met_train"]:.4f}\n'
          f' - Validation set (actual)   {results["mean_met_test"]:.4f} ± {results["std_met_test"]:.4f}\n'
          f' - Validation set (predicted){results["mean_met_pred"]:.4f} ± {results["std_met_pred"]:.4f}'
          )

    print(f'\nConfusion Matrix:\n{results["confusion_matrix"]}')
    print(f'\nAccuracy: {results["mean_accuracy"]:.4f} ± {results["std_accuracy"]:.4f}')
    print(f'Cut-off: {results["optimal_cutoff"]:.4f} ± {results["std_cutoff"]:.4f}')
    print(f'Sensitivity ("met" accuracy): {results["mean_sensitivity"]:.4f} ± {results["std_sensitivity"]:.4f}')
    print(f'Specificity ("primary" acc.): {results["mean_specificity"]:.4f} ± {results["std_specificity"]:.4f}')
"""


if __name__ == "__main__":
    main()


# -i "input_data" -f "input_file.xlsx"