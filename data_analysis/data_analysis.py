import numpy as np
import matplotlib.pyplot as plt
# from scipy import stats
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
import pandas as pd
from sklearn.decomposition import PCA
from varclushi import VarClusHi
from sklearn.preprocessing import StandardScaler

def calculate_nan_statistics(df):
    """
    Calculate count and percentage of NaN values in each column of the DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.

    Returns:
    - nan_count (DataFrame): DataFrame containing count of NaN values in each column.
    - nan_percentage (DataFrame): DataFrame containing percentage of NaN values in each column.
    """
    nan_count = df.isnull().sum()
    nan_percentage = (df.isna().sum() / len(df)) * 100
    return nan_count, nan_percentage


def plot_log_distribution(df):
    
    # Drop non-numeric columns
    data_numeric = df.select_dtypes(include='number')
    
    for column in data_numeric.columns:
        data = data_numeric[column].dropna()  # Remove any missing values
        # Filter out non-positive values
        positive_data = data[data > 0]
        log_data = np.log(data)
        
        # Log-transform the positive data
        log_transformed_data = np.log(positive_data)
        
        # # use scipy.stats to plot against a norm
        # stats.probplot(np.log(data), dist="norm", plot=plt)
        
        # Histogram
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 4, 1)
        plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel(column)
        plt.title('Original Data Histogram')
        
        # Q-Q plot
        plt.subplot(1, 4, 2)
        stats.probplot(data, plot=plt)
        plt.title('Original Data Q-Q Plot')
        
        # Log-Transformed Data Histogram
        plt.subplot(1, 4, 3)
        plt.hist(log_transformed_data, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel(f'Log({column})')
        plt.title('Log-Transformed Data Histogram')
        
        # Q-Q plot for Log-Transformed Data
        plt.subplot(1, 4, 4)
        stats.probplot(log_data, plot=plt)
        plt.title('Q-Q Plot (Log-Transformed Data)')
        
        
        plt.tight_layout()
        plt.show()


def calculate_correlation_matrix(data):
    """
    Calculate the correlation matrix for a DataFrame containing numeric data.

    Parameters:
    ----------
    data : pandas.DataFrame
        A pandas DataFrame containing numeric data for which the correlation matrix
        needs to be computed. Non-numeric columns will be dropped before computation.

    Returns:
    -------
    pandas.DataFrame
        A new DataFrame representing the correlation matrix of the numeric columns
        in the input DataFrame.

    Raises:
    ------
    TypeError
        If the input 'data' is not a pandas DataFrame or does not contain any
        numeric columns.
    
    Notes:
    ------
    - This function drops non-numeric columns before calculating the correlation matrix.

    Example:
    --------
    >>> import pandas as pd
    >>> data = {
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [5, 5, 5, 5, 5],
    ...     'D': [1, 2, 1, 2, 1]
    ... }
    >>> df = pd.DataFrame(data)
    >>> calculate_correlation_matrix(df)
            A         B         C         D
    A  1.000000 -1.000000 -0.500000  0.500000
    B -1.000000  1.000000  0.500000 -0.500000
    C -0.500000  0.500000  1.000000 -1.000000
    D  0.500000 -0.500000 -1.000000  1.000000
    """
    # Drop non-numeric columns
    data_numeric = data.select_dtypes(include='number')
    
    # Calculate correlation matrix
    correlation_matrix = data_numeric.corr()
    
    return correlation_matrix

def calculate_correlation_with_p_value(df):
    """
    Calculate the correlation between the columns of a DataFrame and their p-values.

    Parameters:
        df (DataFrame): Input DataFrame.

    Returns:
        DataFrame: A new DataFrame containing the correlation coefficients and p-values.
                   The index and columns of the new DataFrame will be the column names
                   from the original DataFrame.
    """
    data_numeric = df.select_dtypes(include='number')
    # Calculate the correlation coefficients and p-values
    correlation_matrix = data_numeric.corr()
    p_value_matrix = data_numeric.corr(method=lambda x, y: pearsonr(x, y)[1])

    # Create a new DataFrame with the correlation coefficients and p-values
    result_df = pd.DataFrame(index=data_numeric.columns, columns=data_numeric.columns)
    for col1 in data_numeric.columns:
        for col2 in data_numeric.columns:
            result_df.loc[col1, col2] = (correlation_matrix.loc[col1, col2], p_value_matrix.loc[col1, col2])

    return result_df


# def plot_distribution_and_qq(df):
#     """
#     Plot distribution and QQ plot for each column of the DataFrame.

#     Parameters:
#     - df (DataFrame): The input DataFrame.

#     Returns:
#     None
#     """
#     # Drop non-numeric columns
#     data_numeric = df.select_dtypes(include='number')
    
#     # Set figure size
#     plt.figure(figsize=(12, 6))

#     # Iterate through each column in the DataFrame
#     for col in data_numeric.columns:
#         # Create subplots for distribution and QQ plot
#         plt.subplot(1, 2, 1)
#         sns.histplot(data_numeric[col], kde=True)
#         plt.title(f'Distribution of {col}')

#         plt.subplot(1, 2, 2)
#         stats.probplot(data_numeric[col], dist="norm", plot=plt)
#         plt.title(f'QQ plot of {col}')

#         # Show the plot
#         plt.tight_layout()
#         plt.show()


def plot_distribution_and_qq(df):
    """
    Plot distribution and QQ plot for each column of the DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.

    Returns:
    None
    """
    # Drop non-numeric columns
    data_numeric = df.select_dtypes(include='number')
    
    # Set figure size
    plt.figure(figsize=(12, 6))

    # Iterate through each column in the DataFrame
    for col in data_numeric.columns:
        # Create subplots for distribution and QQ plot
        plt.subplot(1, 2, 1)
        sns.histplot(data_numeric[col], kde=True)
        plt.title(f'Distribution of {col}')

        plt.subplot(1, 2, 2)
        stats.probplot(data_numeric[col], dist="norm", plot=plt)
        plt.title(f'QQ plot of {col}')

        # Adjust QQ plot style to match histogram style
        ax = plt.gca()
        ax.get_lines()[1].set_color('orange')  # Set line color to orange
        ax.spines['top'].set_visible(False)  # Remove top spine
        ax.spines['right'].set_visible(False)  # Remove right spine
        ax.tick_params(axis='both', which='both', length=0)  # Remove ticks
        
        # Show the plot
        plt.tight_layout()
        plt.show()


def perform_pca(df, n_components=None):
    """
    Perform Principal Component Analysis (PCA) on a DataFrame containing numeric values.

    Parameters:
    ----------
    df : pandas.DataFrame
        A pandas DataFrame containing numeric data for PCA analysis.
    n_components : int or None, optional
        The number of components to keep. If None (default), all components are kept.

    Returns:
    -------
    tuple
        A tuple containing the following elements:
        - transformed_df : pandas.DataFrame
            A new DataFrame representing the principal components of the input DataFrame.
        - pca : sklearn.decomposition.PCA
            The fitted PCA object with information about the PCA analysis.
        - columns_used : dict
            A dictionary where keys are the names of principal components (e.g., 'PC1', 'PC2', ...)
            and values are lists of column names from the original DataFrame used in each component.
        - correlations : dict
            A dictionary where keys are the names of principal components (e.g., 'PC1', 'PC2', ...)
            and values are dictionaries containing column names and their respective correlations
            with the corresponding principal component.
        - explained_variance_ratio : numpy.ndarray
            An array containing the explained variance ratio for each principal component.
        - loadings : pandas.DataFrame
            A DataFrame containing the loadings for each column, indicating the contribution
            of each column to each principal component.

    Raises:
    ------
    TypeError
        If 'df' is not a pandas DataFrame, or if 'n_components' is not an integer or None.
    ValueError
        If 'n_components' is not within the valid range or if 'df' contains non-numeric columns.

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [5, 5, 5, 5, 5],
    ...     'D': [1, 2, 1, 2, 1]
    ... }
    >>> df = pd.DataFrame(data)
    >>> transformed_df, pca, columns_used, correlations, explained_variance_ratio, loadings = perform_pca(df)
    >>> print(transformed_df)
        PC1  PC2  PC3  PC4
    0 -2.50 0.00 0.00 0.00
    1 -1.25 0.00 0.00 0.00
    2  0.00 0.00 0.00 0.00
    3  1.25 0.00 0.00 0.00
    4  2.50 0.00 0.00 0.00
    """
    # Select columns with numeric values only
    df_numeric = df.select_dtypes(include='number')
    
    # Drop rows with NaN values
    df_numeric = df_numeric.dropna()

    # sc = StandardScaler()
    # df_std = sc.fit_transform(df_numeric)
    
    # # Convert the standardized array back into a DataFrame
    # df_std = pd.DataFrame(df_std, columns=df_numeric.columns)

    # Instantiate PCA with the specified number of components
    pca = PCA(n_components=n_components)
    
    # Fit PCA on the numeric DataFrame
    pca.fit(df_numeric)
    # pca.fit(df_std)
    # Transform the numeric DataFrame to the principal components
    transformed_df = pca.transform(df_numeric)
    # transformed_df = pca.transform(df_std)
    # Create a new DataFrame with the principal components
    component_names = [f"PC{i+1}" for i in range(pca.n_components_)]
    transformed_df = pd.DataFrame(transformed_df, columns=component_names)
    
    # Get the column names used in each principal component
    columns_used = {}
    for i, component in enumerate(pca.components_):
        used_columns = df_numeric.columns[component != 0]
        columns_used[f"PC{i+1}"] = used_columns.tolist()
        
    # Calculate correlations between columns and principal components
    correlations = {}
    for i, component in enumerate(pca.components_):
        column_correlations = {}
        for j, column_name in enumerate(df_numeric.columns):
            correlation = component[j]
            column_correlations[column_name] = correlation
        correlations[f"PC{i+1}"] = column_correlations
        
    # Calculate explained variance ratio for each principal component
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Calculate loadings for each column
    loadings = pd.DataFrame(pca.components_.T, columns=component_names, index=df_numeric.columns)

    # Plot explained variance ratio for each number of components
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    
    
    return transformed_df, pca, columns_used, correlations, explained_variance_ratio, loadings


def variable_clustering(df, maxeigval2=1, maxclus=None):
    """
    Perform variable clustering on a DataFrame.

    Variable clustering is a technique to group together highly correlated variables into clusters.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing numeric values.
    maxeigval2 : int or None, optional
        Maximum eigenvalue for merging clusters. If None, it is automatically determined.
    maxclus : int or None, optional
        Maximum number of clusters to form. If None, it is automatically determined.

    Returns:
    --------
    tuple
        A tuple containing the following elements:
        - info : pandas.DataFrame
            Information about the clustering analysis.
        - rsquare : pandas.DataFrame
            R-squared values for each variable in each cluster.

    Raises:
    -------
    TypeError
        If the input 'df' is not a pandas DataFrame.

    Example:
    --------
    >>> import pandas as pd
    >>> from varclushi import VarClusHi
    >>> data = {
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [5, 5, 5, 5, 5],
    ...     'D': [1, 2, 1, 2, 1]
    ... }
    >>> df = pd.DataFrame(data)
    >>> info, rsquare = variable_clustering(df)
    >>> print(info)
    >>> print(rsquare)
    """

    # Select columns with numeric values only
    demo1_df = df.select_dtypes(include='number')
    
    # Drop rows with NaN values
    demo1_df = demo1_df.dropna()

    # Calculate the standard deviation for each column
    std_dev = demo1_df.std()

    # Identify columns with a standard deviation of zero
    zero_std_dev_columns = std_dev[std_dev == 0].index

    # Drop columns with zero standard deviation from the DataFrame
    demo1_df = demo1_df.drop(zero_std_dev_columns, axis=1)

    # Perform variable clustering
    demo1_vc = VarClusHi(demo1_df, maxeigval2=maxeigval2, maxclus=maxclus)
    demo1_vc.varclus()

    # Convert info and rsquare to DataFrames
    info_df = pd.DataFrame(demo1_vc.info)
    rsquare_df = pd.DataFrame(demo1_vc.rsquare)

    # Return information about clustering and R-squared values as DataFrames
    return info_df, rsquare_df

def significance_stars(p_value):
    if p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return ''

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr


def data_analysis(df, target_column):
    """
    Perform data analysis on a DataFrame to calculate R-squared, Pearson correlation, 
    Spearman correlation, and their respective p-values for each numeric feature against the target column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data to be analyzed. 
    target_column (str): The name of the target column against which correlations and regressions are computed.

    Returns:
    pandas.DataFrame: A DataFrame containing the feature names, R-squared values, Pearson correlation coefficients and p-values, 
                      Spearman correlation coefficients and p-values, and the number of data points for each feature.
    
    The resulting DataFrame will have the following columns:
    - 'Feature': The name of the feature.
    - 'R-squared': The R-squared value from the linear regression model with the target column.
    - 'Pearson Correlation': The Pearson correlation coefficient with the target column.
    - 'Pearson P-value': The p-value for the Pearson correlation.
    - 'Spearman Correlation': The Spearman correlation coefficient with the target column.
    - 'Spearman P-value': The p-value for the Spearman correlation.
    - 'Num Data Points': The number of data points used in the analysis for the feature.
    """
    
    # Drop non-numeric columns
    data_numeric = df.select_dtypes(include='number')
    
    # List to store analysis results
    feature_r2_corr = []
    
    # Loop through each column in the DataFrame except the target column
    for column in data_numeric.columns:
        if column != target_column:
            
            data = data_numeric[[column, target_column]]
            # Available data
            available_data = (len(data[[column]].dropna())/len(data[[column]]))*100
            
            # Drop NaN values for the current feature and the target column
            data = data_numeric[[column, target_column]].dropna()
    
            # Extract the feature and target variables
            X = data[[column]]
            y = data[target_column]
    
            # Fit a linear regression model
            model = LinearRegression()
            model.fit(X, y)
    
            # Calculate R-squared value and round to 2 decimal points
            r2 = round(r2_score(y, model.predict(X)), 2)
    
            # Calculate Pearson correlation with the target column and p-value
            pearson_corr, pearson_p_value = pearsonr(data[column], data[target_column])
            pearson_corr = round(pearson_corr, 2)
            pearson_p_value = round(pearson_p_value, 4)
            
            # Calculate Spearman correlation with the target column and p-value
            spearman_corr, spearman_p_value = spearmanr(data[column], data[target_column])
            spearman_corr = round(spearman_corr, 2)
            spearman_p_value = round(spearman_p_value, 4)
    
            # Number of data points
            # num_data_points = len(data)
    
            # Append feature name, R-squared value, Pearson correlation, Pearson p-value, Spearman correlation, Spearman p-value, and number of data points to the list
            feature_r2_corr.append({
                'Feature': column,
                'R-squared': r2,
                'Pearson Correlation': pearson_corr,
                'Pearson P-value': pearson_p_value,
                'Spearman Correlation': spearman_corr,
                'Spearman P-value': spearman_p_value,
                # 'Num Data Points': num_data_points, 
                'Data Availability (%)' : available_data
            })
    
    # Create a new DataFrame from the list
    result_df = pd.DataFrame(feature_r2_corr)

    result_df['Correlation Average']=(result_df['Pearson Correlation']+result_df['Spearman Correlation'])/2
    
    return result_df

def analysis_latex_table(df):
    """
    Update the DataFrame to add significance stars to the Pearson and Spearman correlation columns,
    based on their p-values, and then drop the p-value columns.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the correlation and p-value columns.
    
    Returns:
    pandas.DataFrame: The updated DataFrame with significance stars and without p-value columns.
    """
    # Ensure that necessary columns exist
    required_columns = {'Pearson Correlation', 'Pearson P-value', 'Spearman Correlation', 'Spearman P-value'}
    if not required_columns.issubset(df.columns):
        raise ValueError("The DataFrame must contain the columns: 'Pearson Correlation', 'Pearson P-value', 'Spearman Correlation', 'Spearman P-value'")
    
    # Update Pearson Correlation with significance stars
    df['Pearson Correlation'] = df.apply(
        lambda row: f"{row['Pearson Correlation']}{significance_stars(row['Pearson P-value'])}",
        axis=1
    )
    
    # Update Spearman Correlation with significance stars
    df['Spearman Correlation'] = df.apply(
        lambda row: f"{row['Spearman Correlation']}{significance_stars(row['Spearman P-value'])}",
        axis=1
    )
    
    # Drop the p-value columns
    df = df.drop(columns=['Pearson P-value', 'Spearman P-value'])

    # df.insert(loc=1, column='Definition', value='')
    df.insert(loc=6, column='Literature', value='')
    df.insert(loc=1, column='Data Availability (%)', value=df.pop('Data Availability (%)'))
    df.insert(5, 'Correlation Average', df.pop('Correlation Average'))

    
    # df['Num Data Points'] = df['Num Data Points'].map('{:.0f}'.format)
    df['Data Availability (%)'] = df['Data Availability (%)'].map('{:.0f}'.format)
    df['R-squared'] = df['R-squared'].map('{:.2f}'.format)
    df['Correlation Average'] = df['Correlation Average'].map('{:.2f}'.format)

    df = df.sort_values(by='Correlation Average', ascending=False)
    
    return df
# def plot_correlations(df):
#     """
#     Plot the Pearson vs Spearman correlation for each feature in the DataFrame, sorted by the largest absolute Pearson correlation.

#     Parameters:
#     df (pandas.DataFrame): The input DataFrame containing the features and their correlations.
#                            The DataFrame should have columns 'Feature', 'Pearson Correlation', and 'Spearman Correlation'.
#     """
#     # Sort the DataFrame based on the absolute value of Pearson Correlation
#     df_sorted = df.reindex(df['Pearson Correlation'].abs().sort_values(ascending=False).index)
    
#     # Melt the DataFrame to have a long format suitable for seaborn's barplot
#     melted_df = df_sorted.melt(id_vars='Feature', 
#                                value_vars=['Pearson Correlation', 'Spearman Correlation'], 
#                                var_name='Correlation Type', 
#                                value_name='Correlation')
    
#     # Set up the matplotlib figure
#     plt.figure(figsize=(12, 40))
    
#     # Create the bar plot
#     sns.barplot(x='Correlation', y='Feature', hue='Correlation Type', data=melted_df)
    
#     # Add titles and labels
#     plt.title('Pearson vs Spearman Correlation for Each Feature')
#     plt.xlabel('Correlation')
#     plt.ylabel('Feature')
    
#     # Adjust the legend
#     plt.legend(title='Correlation Type')
    
#     # Display the plot
#     plt.show()


# def plot_correlations(df):
#     """
#     Plot the Pearson vs Spearman correlation for each feature in the DataFrame, sorted by the largest absolute Pearson correlation.
#     Includes horizontal guidelines for better readability.

#     Parameters:
#     df (pandas.DataFrame): The input DataFrame containing the features and their correlations.
#                            The DataFrame should have columns 'Feature', 'Pearson Correlation', and 'Spearman Correlation'.
#     """
#     # Sort the DataFrame based on the absolute value of Pearson Correlation
#     df_sorted = df.reindex(df['Pearson Correlation'].abs().sort_values(ascending=False).index)
    
#     # Melt the DataFrame to have a long format suitable for seaborn's barplot
#     melted_df = df_sorted.melt(id_vars='Feature', 
#                                value_vars=['Pearson Correlation', 'Spearman Correlation'], 
#                                var_name='Correlation Type', 
#                                value_name='Correlation')
    
#     # Set up the matplotlib figure
#     plt.figure(figsize=(12, 30))
    
#     # Create the bar plot
#     ax = sns.barplot(x='Correlation', y='Feature', hue='Correlation Type', data=melted_df)
    
#     # Add horizontal guidelines
#     for i in range(len(df_sorted)):
#         plt.axhline(i, color='grey', linestyle='--', linewidth=0.5)
    
#     # Add titles and labels
#     plt.title('Pearson vs Spearman Correlation for Each Feature')
#     plt.xlabel('Correlation')
#     plt.ylabel('Feature')
    
#     # Adjust the legend
#     plt.legend(title='Correlation Type')
    
#     # Display the plot
#     plt.show()

def plot_correlations(df, correlation_type='Pearson'):
    """
    Plot the Pearson vs Spearman correlation for each feature in the DataFrame, sorted by the largest absolute correlation
    of the specified type. Includes horizontal guidelines and vertical lines to indicate degrees of correlation.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features and their correlations.
                           The DataFrame should have columns 'Feature', 'Pearson Correlation', and 'Spearman Correlation'.
    correlation_type (str): The type of correlation to sort by ('Pearson' or 'Spearman'). Default is 'Pearson'.
    """
    # Validate correlation_type input
    if correlation_type not in ['Pearson', 'Spearman']:
        raise ValueError("correlation_type must be either 'Pearson' or 'Spearman'")
    
    # Determine the correlation column to sort by
    correlation_column = f'{correlation_type} Correlation'
    
    # Sort the DataFrame based on the absolute value of the chosen correlation type
    df_sorted = df.reindex(df[correlation_column].abs().sort_values(ascending=False).index)
    
    # Melt the DataFrame to have a long format suitable for seaborn's barplot
    melted_df = df_sorted.melt(id_vars='Feature', 
                               value_vars=['Pearson Correlation', 'Spearman Correlation'], 
                               var_name='Correlation Type', 
                               value_name='Correlation')
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 30))
    
    # Create the bar plot
    ax = sns.barplot(x='Correlation', y='Feature', hue='Correlation Type', data=melted_df)
    
    # Add horizontal guidelines
    for i in range(len(df_sorted)):
        plt.axhline(i, color='grey', linestyle='--', linewidth=0.5)
    
    # Add vertical lines for degrees of correlation
    correlation_degrees = {
        'Perfect': 1,
        'High Degree': 0.5,
        'Moderate Degree': 0.3,
        'Low Degree': 0.1,
        # 'No Correlation': 0
    }
    
    for degree, value in correlation_degrees.items():
        plt.axvline(x=value, color='black', linestyle='--', linewidth=0.7, label=f'{degree} ({value})')
        plt.axvline(x=-value, color='black', linestyle='--', linewidth=0.7)

    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Degree of Correlation')

    # Add titles and labels
    # plt.title(f'{correlation_type} vs Spearman Correlation for Each Feature')
    plt.xlabel('Correlation')
    plt.ylabel('Feature')
    
    # Display the plot
    plt.show()

# def plot_correlations(df):
#     """
#     Plot the Pearson vs Spearman correlation for each feature in the DataFrame, sorted by the largest absolute Pearson correlation.
#     Includes horizontal guidelines and vertical lines to indicate degrees of correlation.

#     Parameters:
#     df (pandas.DataFrame): The input DataFrame containing the features and their correlations.
#                            The DataFrame should have columns 'Feature', 'Pearson Correlation', and 'Spearman Correlation'.
#     """
#     # Sort the DataFrame based on the absolute value of Pearson Correlation
#     df_sorted = df.reindex(df['Pearson Correlation'].abs().sort_values(ascending=False).index)
    
#     # Melt the DataFrame to have a long format suitable for seaborn's barplot
#     melted_df = df_sorted.melt(id_vars='Feature', 
#                                value_vars=['Pearson Correlation', 'Spearman Correlation'], 
#                                var_name='Correlation Type', 
#                                value_name='Correlation')
    
#     # Set up the matplotlib figure
#     plt.figure(figsize=(12, 30))
    
#     # Create the bar plot
#     ax = sns.barplot(x='Correlation', y='Feature', hue='Correlation Type', data=melted_df)
    
#     # Add horizontal guidelines
#     for i in range(len(df_sorted)):
#         plt.axhline(i, color='grey', linestyle='--', linewidth=0.5)
    
#     # Add vertical lines for degrees of correlation
#     correlation_degrees = {
#         'Perfect': 1,
#         'High Degree': 0.5,
#         'Moderate Degree': 0.3,
#         # 'Low Degree': 0.29,
#         # 'No Correlation': 0
#     }
    
#     for degree, value in correlation_degrees.items():
#         plt.axvline(x=value, color='black', linestyle='--', linewidth=0.7, label=f'{degree} ({value})')
#         plt.axvline(x=-value, color='black', linestyle='--', linewidth=0.7)

#     # Remove duplicate labels
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), title='Degree of Correlation')

#     # Add titles and labels
#     plt.title('Pearson vs Spearman Correlation for Each Feature')
#     plt.xlabel('Correlation')
#     plt.ylabel('Feature')
    
#     # Display the plot
#     plt.show()

def print_features_by_correlation(df):
    """
    Print the names of the features sorted by the absolute value of their Pearson correlation
    with 'capacity_mwp' from highest to lowest.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features and their correlations.
                           The DataFrame should have columns 'Feature' and 'Pearson Correlation'.
    """
    # Sort the DataFrame based on the absolute value of Pearson Correlation
    df_sorted = df.reindex(df['Pearson Correlation'].abs().sort_values(ascending=False).index)
    
    # # Print the feature names in the sorted order
    # print("Features sorted by absolute Pearson correlation:")
    # for feature in df_sorted['Feature']:
    #     print(feature)
    print("Features sorted by absolute Pearson correlation:")
    for _, row in df_sorted.iterrows():
        print(f"{row['Feature']}, {row['Pearson Correlation']}")

def print_features_by_spearman_correlation(df):
    """
    Print the names of the features sorted by the absolute value of their Spearman correlation
    with 'capacity_mwp' from highest to lowest.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features and their correlations.
                           The DataFrame should have columns 'Feature' and 'Spearman Correlation'.
    """
    # Sort the DataFrame based on the absolute value of Spearman Correlation
    df_sorted = df.reindex(df['Spearman Correlation'].abs().sort_values(ascending=False).index)
    
    # Print the feature names in the sorted order
    print("Features sorted by absolute Spearman correlation:")
    for feature in df_sorted['Feature']:
        print(feature)
    print("Features sorted by absolute Spearman correlation:")
    for _, row in df_sorted.iterrows():
        print(f"{row['Feature']}, {row['Spearman Correlation']}")
        

def print_features_by_r2(df):
    """
    Print the names of the features and their R-squared values sorted by R-squared with 'capacity_mwp' from highest to lowest.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features and their R-squared values.
                           The DataFrame should have columns 'Feature' and 'R-squared'.
    """
    # Sort the DataFrame based on the R-squared values
    df_sorted = df.reindex(df['R-squared'].sort_values(ascending=False).index)
    
    # Print the feature names and R-squared values in the sorted order
    print("Features sorted by R-squared:")
    for _, row in df_sorted.iterrows():
        print(f"{row['Feature']}, {row['R-squared']}")

def custom_latex_format(x):
    if x < 1000:
        return f"{x:.1f}"
    else:
        exponent = int(f"{x:.1e}".split('e')[1])
        mantissa = float(f"{x:.1e}".split('e')[0])
        return f"${mantissa:.1f}\\times 10^{{{exponent}}}$"

def remove_underscores(column_name):
    column_name_no_underscore = column_name.replace('_', ' ')
    return column_name_no_underscore


def rename_clc_codes(df, csv_file_path="/gb_pv_capacity_model/data_analysis/land categories.csv"):
    """
    Renames columns in a DataFrame based on CLC codes and land type mapping from a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame whose columns need renaming.
        csv_file_path (str, optional): The file path to the CSV file containing 'code' and 'Land Type' columns. 
                                       Defaults to "/gb_pv_capacity_model/data_analysis/land categories.csv".

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    
    # Read the CSV file into a DataFrame
    dfc = pd.read_csv(csv_file_path)
    
    # Extract the old and new column names into separate lists
    old_cols = dfc['code'].tolist()
    new_cols = dfc['Land Type'].tolist()
    
    # Create a dictionary to map old column names to new ones
    col_map = dict(zip(old_cols, new_cols))
    
    # Rename the columns in the DataFrame
    df.rename(columns=col_map, inplace=True)
    
    return df


