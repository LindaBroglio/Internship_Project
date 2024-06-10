import pandas as pd
from matplotlib import pyplot as plt

from dummy_classifier import get_x_y, split_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from feature_selection import random_feature_selection


def inside_loop(df, i, gnb_error_sum, n_of_guesses_sum):
    #df = df.drop('Patient', axis=1)
    x, y = get_x_y(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i) #18 e 7
    #if i == 1:
        #print("training size", x_train.shape[0])
        #print("test size", x_test.shape[0])
    gnb_error, gnb_n_guesses = get_g_naive_bayes_error(x_train, x_test, y_train, y_test)
    single_accuracy=gnb_error/gnb_n_guesses
    #print("errors:", gnb_error)
    gnb_error_sum = gnb_error_sum + gnb_error
    n_of_guesses_sum = n_of_guesses_sum + gnb_n_guesses
    return gnb_error_sum, n_of_guesses_sum, single_accuracy


def get_g_naive_bayes_error(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    gnb_error, guesses = get_mislabeled_patients(y_test, y_pred)
    return gnb_error, guesses


def get_mislabeled_patients(y_test, y_pred):
    #print("true     ", y_test.values)
    #print("predicted", y_pred)
    return (y_test != y_pred).sum(), y_test.shape[0]


def gnb_outside_loop(df, n, random=False):
    gnb_error_sum = 0
    n_of_guesses_sum = 0
    accuracies = []

    for i in range(1, n + 1):
        # print(f"\nIteration {i}:")
        if random == True:
            df_r = random_feature_selection(df, 25)
            gnb_error_sum, n_of_guesses_sum, single_accuracy = inside_loop(df_r, i, gnb_error_sum, n_of_guesses_sum)
            accuracies.append(single_accuracy)
        else:
            gnb_error_sum, n_of_guesses_sum, single_accuracy = inside_loop(df, i, gnb_error_sum, n_of_guesses_sum)
            accuracies.append(single_accuracy)

    # print("\n Final sum of errors", gnb_error_sum)
    # print("Final sum of guesses", n_of_guesses_sum)
    # print("Ratio", gnb_error_sum/n_of_guesses_sum)
    return gnb_error_sum, n_of_guesses_sum, accuracies


def box_plot(df_uncleaned, df_cleaned):
    iters = 150

    print("Number of iterations", iters)
    gnb_error_sum_uncleaned, n_of_guesses_sum_uncleaned, accuracies1 = gnb_outside_loop(df_uncleaned, iters)
    ratio_uncleaned = gnb_error_sum_uncleaned / n_of_guesses_sum_uncleaned

    gnb_error_sum_cleaned, n_of_guesses_sum_cleaned, accuracies2 = gnb_outside_loop(df_cleaned, iters)
    ratio_cleaned = gnb_error_sum_cleaned / n_of_guesses_sum_cleaned

    gnb_error_sum_random, n_of_guesses_sum_random, accuracies3 = gnb_outside_loop(df_cleaned, iters, random=True)
    ratio_random = gnb_error_sum_random / n_of_guesses_sum_random

    # Combine all accuracies into a single list
    all_accuracies = [accuracies1, accuracies2, accuracies3]

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot box plots for all accuracies
    ax.boxplot(all_accuracies, positions=[1, 2, 3])
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Uncleaned', 'Cleaned', 'Random Guess'])

    # Add labels and title
    ax.set_ylabel('Accuracy')
    ax.set_title('Box Plot of Accuracies')

    # Show plot
    plt.show()




