from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def get_x_y(final_df):
    col = 'Response'
    x = final_df.loc[:, final_df.columns != col]
    y = final_df.loc[:, col]
    return x, y


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def create_and_train_classifier(strategy, x_train, y_train):  # 'most_frequent', 'stratified', 'uniform', or 'constant'
    classifier = DummyClassifier(strategy=strategy)
    classifier.fit(x_train, y_train)  # train
    return classifier


def predict_and_evaluate_classifier(classifier, x_test, y_test):
    y_pred = classifier.predict(x_test)
    print(x_test['Patient'], y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


def dummy_classifier(final_df):
    x, y = get_x_y(final_df)
    x_train, x_test, y_train, y_test = split_data(x, y)
    classifier = create_and_train_classifier('uniform', x_train, y_train)
    predict_and_evaluate_classifier(classifier, x_test, y_test)


"""

In this example:

We generate some dummy binary classification data using make_classification.
Split the data into training and testing sets using train_test_split.
Create a DummyClassifier with a specified strategy ('most_frequent', 'stratified', 'uniform', or 'constant').
Train the dummy classifier on the training data.
Make predictions on the test data using predict.
Evaluate the classifier using metrics like accuracy, confusion matrix, and classification report.
You can change the strategy parameter in the DummyClassifier constructor to one of the following:

'most_frequent': Predicts the most frequent class in the training data.
'stratified': Predicts random classes according to the training data distribution.
'uniform': Predicts uniformly random classes.
'constant': Predicts a constant class label provided by the constant parameter (default is 0).




"""
