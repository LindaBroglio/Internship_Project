import sys
import pandas as pd
from lung_data.LDA import LDA_classifier, lda_bootstrap
from lung_data.correlation_analysis.anova import get_anova_test
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

    # df_PET.drop(['Patient_ID', 'Type'], axis=1, inplace=True)
    # df_CT.drop(['Patient_ID', 'Type'], axis=1, inplace=True)
    # pair_grid_plot_all(df_PET, "PET")
    # pair_grid_plot_all(df_CT, "CT")

    df_PET.drop('Patient_ID', axis=1, inplace=True)
    df_CT.drop('Patient_ID', axis=1, inplace=True)

    """ irrelevant_pet = get_anova_test(df_PET, 'Type')
    irrelevant_ct = get_anova_test(df_CT, 'Type')

    print('\nPET least relevant features for anova:\n', irrelevant_pet)
    print('\nCT least relevant features for anova:\n', irrelevant_ct)

    combined_results = pd.concat([irrelevant_pet.reset_index(drop=True), irrelevant_ct.reset_index(drop=True)], axis=1)
    print(combined_results)"""

    #LDA_classifier(df_PET, 'Type')

    results = lda_bootstrap(df_PET, 'Type', n_iterations=100)
    
    print(f'\nMean Accuracy: {results["mean_accuracy"]:.4f}')
    print(f'Standard Deviation of Accuracy: {results["std_accuracy"]:.4f}')
    print(f'\nConfusion Matrix:\n{results["confusion_matrix"]}')

    print('\nPer-Class Accuracies:')
    # Define the width for alignment
    label_width = max(len(str(label)) for label in results['mean_per_class_accuracy'].keys()) + 2
    number_width = 10

    for label in results['mean_per_class_accuracy'].keys():
        mean_accuracy = results['mean_per_class_accuracy'][label]
        std_accuracy = results['std_per_class_accuracy'][label]
        print(f'  Class {label:<{label_width}}: {mean_accuracy:>{number_width}.4f} ± {std_accuracy:<{number_width}.4f}')


if __name__ == "__main__":
    main()
