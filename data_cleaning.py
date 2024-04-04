import pandas as pd
import matplotlib.pyplot as plt



def rows_to_remove(df, columns, missing_type = 'both'):
    """
    Calculate the rows that would be removed if all rows containing a 0 or NaN value in any of the specified columns are removed.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    columns (list): A list of column names to check for 0 or NaN values.
    
    Returns:
    list: A list of indices of rows to be removed.
    """
    rows_to_remove = set()
    for column in columns:

        if missing_type == 'both':
            cond = [0, float('nan'), pd.NA]
        if missing_type == 'zero':
            cond = [0]
        if missing_type == 'nan':
            cond = [float('nan'), pd.NA]

        zero_or_nan_indices = df[df[column].isin(cond)].index
        rows_to_remove.update(zero_or_nan_indices)
    return list(rows_to_remove)

"""
def print_missing(df, columns):
    for column in columns:
        has_zero = (df[column] == 0).any()
        has_nan = df[column].isna().any()
        missing_values = (df[column].isna() | (df[column] == 0)).sum()
        if has_zero or has_nan:
            print(f"Column '{column}' has {'0' if has_zero else ''}{' and ' if (has_zero and has_nan) else ''}{'NaN' if has_nan else ''}. - {missing_values}")
"""


def print_missing(df, columns, missing_type='both'):
    """
    Print information about missing values in specified columns of the DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    columns (list): A list of column names to check for missing values.
    missing_type (str, optional): Specify 'zero' to consider only zeros, 'nan' to consider only NaNs,
                                   'both' (default) to consider both zeros and NaNs.
    
    """
    for column in columns:
        if missing_type == 'zero':
            has_missing = (df[column] == 0).any()
        elif missing_type == 'nan':
            has_missing = df[column].isna().any()
        else:  # 'both'
            has_missing = (df[column].isna() | (df[column] == 0)).any()

        if has_missing:
            missing_values = (df[column].isna() | (df[column] == 0)).sum()
            missing_types = []
            if missing_type == 'zero' or missing_type == 'both':
                if (df[column] == 0).any():
                    missing_types.append('0')
            if missing_type == 'nan' or missing_type == 'both':
                if df[column].isna().any():
                    missing_types.append('NaN')

            print(f"Column '{column}' has {' and '.join(missing_types)}. - {missing_values}")

def display_histogram(df, column_name):
    """
    Display the frequency of values in a categorical variable stored in a column of a DataFrame as a histogram.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column containing the categorical variable.
    """
    # Count occurrences of each category
    category_counts = df[column_name].value_counts()

    # Plot histogram
    plt.figure(figsize=(8, 6))
    category_counts.plot(kind='bar')
    plt.title('Histogram of {}'.format(column_name))
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.show()

def plot_continuous_variable(df, column_name, num_bins=20):
    """
    Plot a histogram to visualize the distribution of a continuous variable stored in a column of a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column containing the continuous variable.
    num_bins (int, optional): Number of bins for the histogram (default is 20).
    """
    plt.figure(figsize=(8, 6))
    plt.hist(df[column_name], bins=num_bins, color='skyblue', edgecolor='black')
    plt.title('Histogram of {}'.format(column_name))
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def print_unique_values(df, column_names):
    """
    Print all unique values found in the specified columns of a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_names (list): A list of column names to print unique values for.
    """
    for column in column_names:
        unique_values = df[column].unique()
        print("Unique values in column '{}':".format(column))
        print(unique_values)
        print()

def scatter_plot(df, x_column, y_column):
    """
    Create a scatter plot between two numeric attributes in a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x_column (str): The name of the column to be plotted on the x-axis.
    y_column (str): The name of the column to be plotted on the y-axis.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_column], df[y_column], alpha=0.5)
    plt.title(f'Scatter Plot: {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
def plot_heatmap(df, x_column, y_column):
    """
    Plot a heatmap based on two columns of a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x_column (str): The name of the column for the x-axis.
    y_column (str): The name of the column for the y-axis.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.crosstab(df[x_column], df[y_column]), cmap='YlGnBu', annot=True, fmt='g')
    plt.title('Heatmap of {} vs {}'.format(x_column, y_column))
    plt.xlabel(y_column)
    plt.ylabel(x_column)
    plt.show()

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    """
    Calculate Cramér's V statistic for categorical-categorical association.
    
    Parameters:
    x (pandas.Series): A categorical variable.
    y (pandas.Series): Another categorical variable.
    
    Returns:
    float: The Cramér's V statistic.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def combine_categorical_columns(df, column_names, new_column_name='combined_column'):
    """
    Combine specified categorical columns into a single column.

    Parameters:
    - df: DataFrame containing the original columns
    - column_names: List of column names storing categorical variables to be combined
    - new_column_name: Name of the new combined column (default is 'combined_column')

    Returns:
    - DataFrame with the new combined column added
    """

    # Combine values of specified columns into a new column
    df[new_column_name] = df[column_names].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    # Drop the original columns
    df.drop(columns=column_names, inplace=True)

    return df