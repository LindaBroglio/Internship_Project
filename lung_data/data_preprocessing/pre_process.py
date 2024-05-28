import sys


def check_for_NaN(df, name):
    if df.isnull().sum().sum():
        print("Exiting the program, NaN found in", name, "but no actions to fix the issue were given")
        sys.exit(0)
    else:
        print("No NaN found in", name)



