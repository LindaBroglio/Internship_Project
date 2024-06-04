import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from lung_data.LDA import get_x_y


def build_pipeline(df, y_column, model):
    x, y = get_x_y(df, y_column)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    pre_process_pipe = Pipeline(steps=[
        ("missing steps", SimpleImputer(strategy='mean')),  # also takes care of nan
        ("scaling", MinMaxScaler())
    ])

    numeric_fit_result = pre_process_pipe.fit(df).transform(df).shape

    columns = np.array(x.columns)
    columns = columns[:-1]

    numerical_transformed = pd.DataFrame(data=pre_process_pipe.fit_transform(df[columns]), columns=columns)
    for feature in numerical_transformed.columns:
        plt.figure()
        sns.histplot(data=numerical_transformed, x=feature)

    #pre_process_pipe.fit(x_train, y_train)

    model_pipeline = Pipeline(steps=[
        ("preprocessing", pre_process_pipe),
        ("model", model)]
    )

    #model_pipeline.fit(x_train, y_train)

    model_output = model_pipeline.predict(x_test)
    cm = confusion_matrix(y_test, model_output)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True)
    plt.title('Confusion Matrix - Test Data')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    #print(classification_report(y_test, model_output))
