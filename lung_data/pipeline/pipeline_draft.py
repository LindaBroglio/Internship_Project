import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from lung_data.data_preprocessing.importing import from_xls_into_df


class NaNChecker(BaseEstimator, TransformerMixin):
    def __init__(self, df_name):
        self.df_name = df_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            print(f"{self.df_name} contains {nan_count} NaN values.")
        else:
            print(f"{self.df_name} does not contain any NaN values.")
        return X


class ColumnTypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns, target_type):
        self.columns = columns
        self.target_type = target_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in self.columns:
            X[column] = X[column].astype(self.target_type)
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)


def process_data():
    df_PET, df_CT = from_xls_into_df(1)

    # Define the pipeline for df_PET
    pet_pipeline = Pipeline([
        ('nan_checker', NaNChecker(df_name="df_PET")),
        ('type_converter', ColumnTypeConverter(columns=['Patient_ID', 'Type'], target_type='string')),
        ('column_dropper', ColumnDropper(columns=['Patient_ID']))
    ])

    # Define the pipeline for df_CT
    ct_pipeline = Pipeline([
        ('nan_checker', NaNChecker(df_name="df_CT")),
        ('type_converter', ColumnTypeConverter(columns=['Patient_ID', 'Type'], target_type='string')),
        ('column_dropper', ColumnDropper(columns=['Patient_ID']))
    ])

    # Process the data
    df_PET_processed = pet_pipeline.fit_transform(df_PET)
    df_CT_processed = ct_pipeline.fit_transform(df_CT)

    # Printing the data types for verification
    print("Final data types in df_PET:  ", df_PET_processed.dtypes.value_counts().to_dict())
    print("Final data types in df_CT:   ", df_CT_processed.dtypes.value_counts().to_dict())

    # Placeholder for plotting (uncomment and implement as needed)
    # pair_grid_plot_all(df_PET_processed, "PET")
    # pair_grid_plot_all(df_CT_processed, "CT")

    return df_PET_processed, df_CT_processed

