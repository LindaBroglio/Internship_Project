import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn

from dummy_classifier import get_x_y
from gnb_classification import gnb_outside_loop
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def knn_accuracies(df, iters, test_size=0.2, random_state=None, n_neighbors=5):
    accuracies = []
    for i in range(iters):
        x, y = get_x_y(df)
        #x = x.drop('Patient', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies


def grad_boost_accuracies(df, iters, test_size=0.2, random_state=None):
    accuracies = []
    for i in range(iters):
        x, y = get_x_y(df)
        #x = x.drop('Patient', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        clf = GradientBoostingClassifier()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies


def rf_accuracies(df, iters, test_size=0.2, random_state=None):
    accuracies = []
    for i in range(iters):
        x, y = get_x_y(df)
        #x = x.drop('Patient', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies


def plot_boxes(df_uncleaned, df_cleaned):
    iters = 50

    print("Number of iterations", iters)
    gnb_error_sum_uncleaned, n_of_guesses_sum_uncleaned, accuracies1 = gnb_outside_loop(df_uncleaned, iters)
    gnb_error_sum_uncleaned, n_of_guesses_sum_uncleaned, accuracies2 = gnb_outside_loop(df_cleaned, iters)
    acc_gnb_1_df = pd.DataFrame(columns=['accuracies', 'feature_selection', 'algorithm'])
    acc_gnb_1_df['accuracies'] = accuracies1
    acc_gnb_1_df['feature_selection'] = 'uncleaned'
    acc_gnb_1_df['algorithm'] = 'Naive Bayes'

    acc_gnb_2_df = pd.DataFrame(columns=['accuracies', 'feature_selection', 'algorithm'])
    acc_gnb_2_df['accuracies'] = accuracies2
    acc_gnb_2_df['feature_selection'] = 'cleaned'
    acc_gnb_2_df['algorithm'] = 'Naive Bayes'

    accuracies_rf_1 = rf_accuracies(df_uncleaned, iters)
    acc_rf_1_df = pd.DataFrame(columns=['accuracies', 'feature_selection', 'algorithm'])
    acc_rf_1_df['accuracies'] = accuracies_rf_1
    acc_rf_1_df['feature_selection'] = 'uncleaned'
    acc_rf_1_df['algorithm'] = 'Random Forest'

    accuracies_rf_2 = rf_accuracies(df_cleaned, iters)
    acc_rf_2_df = pd.DataFrame(columns=['accuracies', 'feature_selection', 'algorithm'])
    acc_rf_2_df['accuracies'] = accuracies_rf_2
    acc_rf_2_df['feature_selection'] = 'cleaned'
    acc_rf_2_df['algorithm'] = 'Random Forest'

    accuracies_gb_1 = grad_boost_accuracies(df_uncleaned, iters)
    acc_gb_1_df = pd.DataFrame(columns=['accuracies', 'feature_selection', 'algorithm'])
    acc_gb_1_df['accuracies'] = accuracies_gb_1
    acc_gb_1_df['feature_selection'] = 'uncleaned'
    acc_gb_1_df['algorithm'] = 'Gradient Boosting'

    accuracies_gb_2 = grad_boost_accuracies(df_cleaned, iters)
    acc_gb_2_df = pd.DataFrame(columns=['accuracies', 'feature_selection', 'algorithm'])
    acc_gb_2_df['accuracies'] = accuracies_gb_2
    acc_gb_2_df['feature_selection'] = 'cleaned'
    acc_gb_2_df['algorithm'] = 'Gradient Boosting'

    accuracies_knn_1 = knn_accuracies(df_uncleaned, iters)
    acc_knn_1_df = pd.DataFrame(columns=['accuracies', 'feature_selection', 'algorithm'])
    acc_knn_1_df['accuracies'] = accuracies_knn_1
    acc_knn_1_df['feature_selection'] = 'uncleaned'
    acc_knn_1_df['algorithm'] = 'K-Nearest'

    accuracies_knn_2 = knn_accuracies(df_cleaned, iters)
    acc_knn_2_df = pd.DataFrame(columns=['accuracies', 'feature_selection', 'algorithm'])
    acc_knn_2_df['accuracies'] = accuracies_knn_2
    acc_knn_2_df['feature_selection'] = 'cleaned'
    acc_knn_2_df['algorithm'] = 'K-Nearest'

    acc_df = pd.concat([acc_gnb_1_df, acc_gnb_2_df,
                        acc_rf_1_df, acc_rf_2_df,
                        acc_gb_1_df, acc_gb_2_df,
                        acc_knn_1_df, acc_knn_2_df])

    sn.violinplot(x='algorithm', y='accuracies', hue='feature_selection', data=acc_df)
    plt.show()

    #all_accuracies = [accuracies1, accuracies2, accuracies_rf_1, accuracies_rf_2, accuracies_gb_1, accuracies_gb_2]
