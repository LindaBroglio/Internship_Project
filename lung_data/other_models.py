import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from lung_data.LDA import get_x_y, add_binary_column


def rf_accuracies(df, response_column, binary_col_name, iters=100, smote=False):
    x, y = get_x_y(df, binary_col_name)
    accuracies = []
    for i in range(iters):
        if smote:
            x_resampled, y_resampled = SMOTE().fit_resample(x, y)
            x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=i)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies


def svm_accuracies(df, response_column, binary_col_name, iters=100, smote=False):
    x, y = get_x_y(df, binary_col_name)
    accuracies = []
    for i in range(iters):
        if smote:
            x_resampled, y_resampled = SMOTE().fit_resample(x, y)
            x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=i)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
        clf = SVC()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies


def lda_accuracies(df, response_column, binary_col_name, iters=100, smote=False):
    x, y = get_x_y(df, binary_col_name)
    accuracies = []
    for i in range(iters):
        if smote:
            x_resampled, y_resampled = SMOTE().fit_resample(x, y)
            x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=i)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        y_pred = lda.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies


def plot_boxes(df, response_column, binary_col_name):

    df = add_binary_column(df, response_column, binary_col_name)
    df.drop(response_column, axis=1, inplace=True)

    accuracies_rf_unbalanced = rf_accuracies(df, response_column, binary_col_name)
    accuracies_rf_smote = rf_accuracies(df, response_column, binary_col_name, smote=True)

    accuracies_svm_unbalanced = svm_accuracies(df, response_column, binary_col_name)
    accuracies_svm_smote = svm_accuracies(df, response_column, binary_col_name, smote=True)

    accuracies_lda_unbalanced = lda_accuracies(df, response_column, binary_col_name)
    accuracies_lda_smote = lda_accuracies(df, response_column, binary_col_name, smote=True)

    rf_unbalanced = pd.DataFrame(columns=['accuracies', 'data_balance', 'algorithm'])
    rf_unbalanced['accuracies'] = accuracies_rf_unbalanced
    rf_unbalanced['data_balance'] = 'unbalanced'
    rf_unbalanced['algorithm'] = 'Random Forest'

    rf_smote = pd.DataFrame(columns=['accuracies', 'data_balance', 'algorithm'])
    rf_smote['accuracies'] = accuracies_rf_smote
    rf_smote['data_balance'] = 'smote'
    rf_smote['algorithm'] = 'Random Forest'

    svm_unbalanced = pd.DataFrame(columns=['accuracies', 'data_balance', 'algorithm'])
    svm_unbalanced['accuracies'] = accuracies_svm_unbalanced
    svm_unbalanced['data_balance'] = 'unbalanced'
    svm_unbalanced['algorithm'] = 'Support Vector Machine'

    svm_smote = pd.DataFrame(columns=['accuracies', 'data_balance', 'algorithm'])
    svm_smote['accuracies'] = accuracies_svm_smote
    svm_smote['data_balance'] = 'smote'
    svm_smote['algorithm'] = 'Support Vector Machine'

    lda_unbalanced = pd.DataFrame(columns=['accuracies', 'data_balance', 'algorithm'])
    lda_unbalanced['accuracies'] = accuracies_lda_unbalanced
    lda_unbalanced['data_balance'] = 'unbalanced'
    lda_unbalanced['algorithm'] = 'Linear Discriminant Analysis'

    lda_smote = pd.DataFrame(columns=['accuracies', 'data_balance', 'algorithm'])
    lda_smote['accuracies'] = accuracies_lda_smote
    lda_smote['data_balance'] = 'smote'
    lda_smote['algorithm'] = 'Linear Discriminant Analysis'

    acc_df = pd.concat([rf_unbalanced, rf_smote, svm_unbalanced, svm_smote, lda_unbalanced, lda_smote])

    sns.boxplot(x='algorithm', y='accuracies', hue='data_balance', data=acc_df)
    plt.title('Box Plot')
    plt.legend(loc="lower right")
    plt.savefig("Box_Plot")
    plt.close()

