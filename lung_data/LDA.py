import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE


def get_x_y(df, y_column):
    x = df.loc[:, df.columns != y_column]
    y = df.loc[:, [y_column]]
    return x, y


def split_data(x, y, random_state):
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


def lda_bootstrap_multiclass(df, response_column, n_iterations):
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


def backward_elimination(x_train, y_train, threshold=0.01):
    lda = LinearDiscriminantAnalysis()
    remaining_features = x_train.columns.tolist()
    while True:
        lda.fit(x_train[remaining_features], y_train)
        coeffs = np.abs(lda.coef_).flatten()
        min_coeff_idx = np.argmin(coeffs)
        if coeffs[min_coeff_idx] < threshold:
            feature_to_remove = remaining_features[min_coeff_idx]
            remaining_features.remove(feature_to_remove)
        else:
            break
    return x_train[remaining_features], remaining_features


def lda_bootstrap_binary(df, response_column, binary_col_name, n_iterations, plot_name, method='direct'):
    df = add_binary_column(df, response_column, binary_col_name)
    df.drop(response_column, axis=1, inplace=True)
    x, y = get_x_y(df, binary_col_name)

    (accuracies, cm_accuracies, confusion_matrices, met_count_pred, met_count_test,
     met_count_train, sensitivities, specificities, roc_data, auc_values, cutoffs) = initializer()

    for i in range(n_iterations):
        x_resampled, y_resampled = SMOTE().fit_resample(x, y)
        x_train, x_test, y_train, y_test = split_data(x_resampled, y_resampled, i)

        if method == 'backward':
            x_train, selected_features = backward_elimination(x_train, y_train, threshold=0.01)
            x_test = x_test[selected_features]

        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        y_pred = lda.predict(x_test)

        met_count_train.append(y_train.value_counts()['met'])
        met_count_test.append(y_test.value_counts()['met'])
        met_count_pred.append((y_pred == 'met').sum())
        accuracies.append(accuracy_score(y_test, y_pred))
        conf_matrix = confusion_matrix(y_test, y_pred, labels=lda.classes_)
        specificity, sensitivity, cm_accuracy = get_specificity_sensitivity(conf_matrix)
        specificities.append(specificity)
        sensitivities.append(sensitivity)
        cm_accuracies.append(cm_accuracy)
        confusion_matrices += conf_matrix

        fpr, tpr, roc_auc, thresholds = get_roc_and_auc(y_test, y_pred)
        roc_data.append((fpr, tpr, roc_auc))
        auc_values.append(roc_auc)

        youdens_index = tpr - fpr
        optimal_idx = np.argmax(youdens_index)
        #print(thresholds)
        optimal_threshold = thresholds[optimal_idx]
        cutoffs.append(optimal_threshold)

    mean_met_count_train, std_met_count_train = get_mean_and_std(met_count_train)
    mean_met_count_test, std_met_count_test = get_mean_and_std(met_count_test)
    mean_met_count_pred, std_met_count_pred = get_mean_and_std(met_count_pred)
    mean_cm_accuracy, std_cm_accuracy = get_mean_and_std(cm_accuracies)
    mean_specificity, std_specificity = get_mean_and_std(specificities)
    mean_sensitivity, std_sensitivity = get_mean_and_std(sensitivities)
    mean_accuracy, std_accuracy = get_mean_and_std(accuracies)
    combined_confusion_matrix = confusion_matrices/n_iterations

    #mean_auc, std_auc = plot_auc(auc_values, roc_data, cutoffs, False, plot_name)

    return {
        'tot_train': len(y_train),
        'tot_test': len(y_test),

        'mean_met_train': mean_met_count_train,
        'std_met_train': std_met_count_train,

        'mean_met_test': mean_met_count_test,
        'std_met_test': std_met_count_test,

        'mean_met_pred': mean_met_count_pred,
        'std_met_pred': std_met_count_pred,

        'mean_specificity': mean_specificity,
        'std_specificity': std_specificity,
        'mean_sensitivity': mean_sensitivity,
        'std_sensitivity': std_sensitivity,

        'mean_cm_accuracy': mean_cm_accuracy,
        'std_cm_accuracy': std_cm_accuracy,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,

        'confusion_matrix': combined_confusion_matrix,
        #'mean_auc': mean_auc,
        #'std_auc': std_auc,

        'optimal_cutoff': np.mean(cutoffs),
        'std_cutoff': np.std(cutoffs),
        'accuracies': accuracies
    }


def plot_auc(auc_values, roc_data, cutoffs, plot_all, filename):
    plt.figure(figsize=(10, 8))
    if plot_all:
        for fpr, tpr, roc_auc in roc_data:
            plt.plot(fpr, tpr, alpha=0.3)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for fpr, tpr, _ in roc_data:
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = tprs.std(axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_values)
    plt.plot(mean_fpr, mean_tpr, color='blue', label='Mean ROC curve (AUC = %0.2f ± %0.2f)' % (mean_auc, std_auc), lw=2,
             alpha=1)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    mean_cutoff = np.mean(cutoffs)
    std_cutoff = np.std(cutoffs)
    mean_cutoff_idx = np.abs(mean_fpr - mean_cutoff).argmin()
    cutoff_tpr = mean_tpr[mean_cutoff_idx]
    cutoff_fpr = mean_fpr[mean_cutoff_idx]
    plt.scatter([cutoff_fpr], [cutoff_tpr], color='red', label='Mean Cutoff (± %0.2f)' % std_cutoff)
    plt.errorbar(cutoff_fpr, cutoff_tpr, xerr=std_cutoff, fmt='o', color='red', alpha=0.5, label='Std Dev Cutoff')

    if not plot_all:
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()
    return mean_auc, std_auc


def initializer():
    accuracies = []
    met_count_train = []
    met_count_test = []
    met_count_pred = []
    specificities = []
    sensitivities = []
    cm_accuracies = []
    confusion_matrices = np.zeros((2, 2))
    roc_data = []
    auc_values = []
    cutoffs = []
    return (accuracies, cm_accuracies, confusion_matrices, met_count_pred, met_count_test,
            met_count_train, sensitivities, specificities, roc_data, auc_values, cutoffs)


def add_binary_column(df, response_column, binary_col_name):
    df[binary_col_name] = [binary_value(i) for i in df[response_column]]
    return df


def binary_value(og_string):
    if og_string == "met":
        return "met"
    else:
        return "primary"


def get_specificity_sensitivity(c_m):
    TP = c_m[0][0]
    FP = c_m[0][1]
    FN = c_m[1][0]
    TN = c_m[1][1]
    specificity = TN/(TN+FP)
    sensitivity = TP/(TP+FN)
    accuracy = (TP + TN)/(TP + FP + TN + FN)
    return specificity, sensitivity, accuracy


def get_mean_and_std(x):
    return np.mean(x), np.std(x)


def get_roc_and_auc(y_test, y_pred):
    y_test_bin = [1 if item == 'met' else 0 for item in y_test]
    y_pred_bin = [1 if item == 'met' else 0 for item in y_pred]
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_pred_bin, drop_intermediate=True)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, thresholds


def write_results_to_file(df, results, filename):
    tot_patients = len(df['Binary_Type'])
    tot_met_patients = len(df[df['Binary_Type'] == "met"])

    with open(filename, 'w') as file:
        file.write(f'Total number of patients: {tot_patients}, of which {tot_met_patients} with metastasis\n')
        file.write(f'\nPatients in training set: {results["tot_train"]}\n')
        file.write(f'Patients in test set:      {results["tot_test"]}\n')

        file.write(f'\nMetastasis patients counter in:\n'
                   f' - Training set              {results["mean_met_train"]:.4f} ± {results["std_met_train"]:.4f}\n'
                   f' - Validation set (actual)   {results["mean_met_test"]:.4f} ± {results["std_met_test"]:.4f}\n'
                   f' - Validation set (predicted){results["mean_met_pred"]:.4f} ± {results["std_met_pred"]:.4f}\n'
                   )

        file.write(f'\nConfusion Matrix:\n{results["confusion_matrix"]}\n')
        file.write(f'\nAccuracy: {results["mean_accuracy"]:.4f} ± {results["std_accuracy"]:.4f}\n')
        file.write(f'Cut-off: {results["optimal_cutoff"]:.4f} ± {results["std_cutoff"]:.4f}\n')
        file.write(f'Sensitivity ("met" accuracy): {results["mean_sensitivity"]:.4f} ± {results["std_sensitivity"]:.4f}\n')
        file.write(f'Specificity ("primary" acc.): {results["mean_specificity"]:.4f} ± {results["std_specificity"]:.4f}\n')


def backward_elimination(x_train, y_train, threshold=0.01):
    lda = LinearDiscriminantAnalysis()
    remaining_features = x_train.columns.tolist()
    while True:
        lda.fit(x_train[remaining_features], y_train)
        coeffs = np.abs(lda.coef_).flatten()
        min_coeff_idx = np.argmin(coeffs)
        if coeffs[min_coeff_idx] < threshold:
            feature_to_remove = remaining_features[min_coeff_idx]
            remaining_features.remove(feature_to_remove)
        else:
            break
    return x_train[remaining_features], remaining_features

#(sensitivity*0,2279381443)+(specificity*(1-0,2279381443))
#(0,5798*0,2279381443)+(0,8742*(1-0,2279381443))
