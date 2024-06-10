import pandas as pd
from scipy.stats import f_oneway


def get_anova_test(df, categorical_feature):
    anova_results_dict = {}

    numerical_columns = [col for col in df.columns if col != categorical_feature]
    for numerical_col in numerical_columns:
        grouped_data = [df[df[categorical_feature] == category][numerical_col] for category in
                        df[categorical_feature].unique()]
        anova_results = f_oneway(*grouped_data)
        anova_results_dict[numerical_col] = {
            'F-statistic': anova_results.statistic,
            'p-value': anova_results.pvalue
        }
    anova_results_df = pd.DataFrame(anova_results_dict).T
    anova_results_df['relevance'] = anova_results_df['p-value'].apply(
        lambda p: 'relevant' if p < 0.05 else 'irrelevant')
    #print(anova_results_df)
    anova_results_df.reset_index(inplace=True)
    anova_results_df.rename(columns={'index': 'feature'}, inplace=True)

    return anova_results_df[anova_results_df['relevance'] == 'irrelevant']['feature']


