import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, \
    ConfusionMatrixDisplay


def get_x_y(final_df):
    col = 'Response'
    x = final_df.loc[:, final_df.columns != col]
    y = final_df.loc[:, col]
    return x, y


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def create_and_train_classifier(strategy, x_train, y_train, constant_value=1):
    if strategy == 'constant' and constant_value is None:
        raise ValueError("Constant target value has to be specified when the constant strategy is used.")
    if strategy == 'constant':
        classifier = DummyClassifier(strategy=strategy, constant=constant_value)
    else:
        classifier = DummyClassifier(strategy=strategy)
    classifier.fit(x_train, y_train)
    return classifier


def predict_and_evaluate_classifier(classifier, x_test, y_test):
    y_pred = classifier.predict(x_test)
    print(x_test['Patient'])
    print('Actual:  ', y_test.values)
    print('Predicted', y_pred)
    results = []
    for yi, ypi in zip(y_test, y_pred):
        results.append(get_result(yi, ypi))
    print('Results: ', results)
    accuracy = accuracy_score(y_test, y_pred)
    print_confusion_matrix(y_test.values, y_pred)
    print("Accuracy:", accuracy)


"""
def predict_probabilities_dummy(classifier, x_test, y_test):
    if classifier.strategy == 'most_frequent':
        most_frequent_class = classifier.classes_[0]
        probabilities = np.zeros(len(x_test))
        probabilities[y_test == most_frequent_class] = 1
    elif classifier.strategy == 'stratified':
        probabilities = np.random.rand(len(x_test), 2)  # random probabilities for each class
        probabilities /= probabilities.sum(axis=1)[:, np.newaxis]
    elif classifier.strategy == 'uniform':
        probabilities = np.ones((len(x_test), 2)) / 2  # equal probabilities for each class
    elif classifier.strategy == 'constant':
        constant_class = classifier.constant
        probabilities = np.zeros((len(x_test), 2))
        probabilities[:, constant_class] = 1
    else:
        raise ValueError("Invalid strategy")
    return probabilities
"""


def plot_roc_curve(y_test, y_pred_proba):
    y_pred_proba_array = np.array(y_pred_proba)
    print(y_pred_proba_array)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_array[:, 1])  # Extract probabilities for class 1
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def dummy_classifier(final_df):
    x, y = get_x_y(final_df)
    x_train, x_test, y_train, y_test = split_data(x, y)
    classifier = create_and_train_classifier('uniform', x_train, y_train)
    predict_and_evaluate_classifier(classifier, x_test, y_test)


def print_confusion_matrix(true, pred):
    cm = confusion_matrix(true, pred)
    print('\nTrue positive = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True negative = ', cm[1][1])
    print('\nConfusion Matrix:')
    df = pd.DataFrame(cm, columns=['positive', 'negative'], index=['positive', 'negative'])
    print(df)
    return

def get_result(true, pred):
    if pred:
        return 'TP' if true else 'FP'
    return 'FN' if true else 'TN'



"""

metrics: accuracy, confusion matrix, and classification report.

'most_frequent': Predicts the most frequent class in the training data.
'stratified': Predicts random classes according to the training data distribution.
'uniform': Predicts uniformly random classes.
'constant': Predicts a constant class label provided by the constant parameter (default is 0).

"""
