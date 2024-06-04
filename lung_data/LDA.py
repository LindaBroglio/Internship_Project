import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def get_x_y(df, y_column):
    x = df.loc[:, df.columns != y_column]
    y = df.loc[:, y_column]
    return x, y


def split_data(x, y, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    return x_train, x_test, y_train, y_test


def create_and_fit_classifier(x_train, y_train, x_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    y_pred = lda.predict(x_test)
    return y_pred


def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    report = classification_report(y_test, y_pred)
    print(f'Classification Report:\n{report}')
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix:\n{conf_matrix}')
    return


def LDA_classifier(df, response_column):
    x, y = get_x_y(df, response_column)
    x_train, x_test, y_train, y_test = split_data(x, y)
    y_pred = create_and_fit_classifier(x_train, y_train, x_test)
    evaluate(y_test, y_pred)
    return


def lda_bootstrap(df, response_column, n_iterations):
    x, y = get_x_y(df, response_column)

    accuracies = []
    confusion_matrices = np.zeros((4, 4))
    #    confusion_matrices = np.zeros((len(np.unique(y)), len(np.unique(y))))

    per_class_accuracies = {label: [] for label in np.unique(y)}

    for i in range(n_iterations):

        x_train, x_test, y_train, y_test = split_data(x, y, i)

        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        y_pred = lda.predict(x_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        conf_matrix = confusion_matrix(y_test, y_pred, labels=lda.classes_)
        confusion_matrices += conf_matrix

        for idx, label in enumerate(lda.classes_):
            per_class_accuracies[label].append(conf_matrix[idx, idx] / conf_matrix[idx].sum())

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    combined_confusion_matrix = confusion_matrices/n_iterations

    mean_per_class_accuracy = {label: np.mean(accuracies) for label, accuracies in per_class_accuracies.items()}
    std_per_class_accuracy = {label: np.std(accuracies) for label, accuracies in per_class_accuracies.items()}

    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': combined_confusion_matrix,
        'mean_per_class_accuracy': mean_per_class_accuracy,
        'std_per_class_accuracy': std_per_class_accuracy
    }





