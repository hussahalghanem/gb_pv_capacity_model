
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
import mlflow
import pycountry
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import shap 
import matplotlib.pyplot as plt

def drop_columns_with_high_nan(df, threshold=10):
    '''
    Drops columns from the DataFrame where the percentage of NaN values exceeds a specified threshold.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - threshold (float): The threshold percentage of NaN values. Columns with NaN percentage higher than this threshold will be dropped.
                         Defaults to 10.

    Returns:
    - df (DataFrame): The DataFrame with columns dropped based on the threshold.
    '''
    # Calculate the percentage of NaN values for each column
    nan_percentage = (df.isna().sum() / len(df)) * 100
    
    # Select columns with NaN percentage higher than the threshold
    selected_columns = nan_percentage[nan_percentage > threshold].index
    
    # Drop columns where NaN percentage is higher than the threshold
    df = df.drop(columns=selected_columns)
    
    return df



def select_columns_by_correlation(df, target_column, threshold=0.3, include_categorical=False):
    '''
    Selects columns from the DataFrame based on correlation with the target column.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - target_column (str): The name of the target column.
    - threshold (float): The correlation threshold. Columns with correlation coefficient absolute value greater than this threshold
                         with the target column will be selected. Defaults to 0.3.
    - include_categorical (bool): Whether to include categorical columns 'year' and 'country_cd' after selecting columns based on correlation. Defaults to False.

    Returns:
    - selected_columns (DataFrame): The DataFrame with selected columns based on correlation.
    '''
    # Using Pearson Correlation
    numeric_columns = df.select_dtypes(include='number')  # Exclude non-numeric (string) columns
    cor = numeric_columns.corr()
    cor_target = abs(cor[target_column])
    
    # Select relevant features based on correlation with the target column
    relevant_features = cor_target[cor_target > threshold]
    column_labels = relevant_features.index.tolist()
    
    # Select columns from the original DataFrame based on relevant features
    selected_columns = df[column_labels]
    
    # Include categorical columns if specified
    if include_categorical:
        selected_columns = df.loc[:, ['year', 'country_cd']].join(selected_columns)
    
    return selected_columns, column_labels


def select_columns_by_spearman_correlation(df, target_column, threshold=0.3, include_categorical=False):
    '''
    Selects columns from the DataFrame based on Spearman's correlation with the target column.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - target_column (str): The name of the target column.
    - threshold (float): The correlation threshold. Columns with correlation coefficient absolute value greater than this threshold
                         with the target column will be selected. Defaults to 0.3.
    - include_categorical (bool): Whether to include categorical columns 'year' and 'country_cd' after selecting columns based on correlation.
                                   Defaults to False.

    Returns:
    - selected_columns (DataFrame): The DataFrame with selected columns based on correlation.
    - column_labels (list): The list of column labels that were selected based on correlation.
    '''
    # Using Spearman Correlation
    numeric_columns = df.select_dtypes(include='number')  # Exclude non-numeric (string) columns
    cor = numeric_columns.corr(method='spearman')
    cor_target = abs(cor[target_column])
    
    # Select relevant features based on correlation with the target column
    relevant_features = cor_target[cor_target > threshold]
    column_labels = relevant_features.index.tolist()
    
    # Select columns from the original DataFrame based on relevant features
    selected_columns = df[column_labels]
    
    # Include categorical columns if specified
    if include_categorical:
        selected_columns = df.loc[:, ['year', 'country_cd']].join(selected_columns)
    
    return selected_columns, column_labels


def select_columns_by_average_correlation(df, target_column, threshold=0.3, include_categorical=False):
    '''
    Selects columns from the DataFrame based on the average of Pearson and Spearman's correlation with the target column.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - target_column (str): The name of the target column.
    - threshold (float): The correlation threshold. Columns with correlation coefficient absolute value greater than this threshold
                         with the target column will be selected. Defaults to 0.3.
    - include_categorical (bool): Whether to include categorical columns 'year' and 'country_cd' after selecting columns based on correlation.
                                   Defaults to False.

    Returns:
    - selected_columns (DataFrame): The DataFrame with selected columns based on correlation.
    - column_labels (list): The list of column labels that were selected based on correlation.
    '''
    
    numeric_columns = df.select_dtypes(include='number')  # Exclude non-numeric (string) columns
    # Pearson
    pearson_cor = numeric_columns.corr()
    pearson_cor_target = abs(pearson_cor[target_column])
    # Spearman Correlation
    spearman_cor = numeric_columns.corr(method='spearman')
    spearman_cor_target = abs(spearman_cor[target_column])

    average_cor_target = (pearson_cor_target+spearman_cor_target)/2
    
    # Select relevant features based on correlation with the target column
    relevant_features = average_cor_target[average_cor_target > threshold]
    column_labels = relevant_features.index.tolist()
    
    # Select columns from the original DataFrame based on relevant features
    selected_columns = df[column_labels]
    
    # Include categorical columns if specified
    if include_categorical:
        selected_columns = df.loc[:, ['year', 'country_cd']].join(selected_columns)
    
    return selected_columns, column_labels


def select_columns_by_column_name(df, target_column, relevant_features, include_categorical=False):
    '''
    Selects columns from the DataFrame based on explicitly provided column names.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - target_column (str): The name of the target column. (Optional in this version, can be used for future extensions.)
    - relevant_features (list): A list of column names to select.
    - include_categorical (bool): Whether to include categorical columns like 'year' and 'country_cd'.
                                   Defaults to False.

    Returns:
    - selected_columns (DataFrame): The DataFrame with selected columns based on the given column names.
    - column_labels (list): The list of column labels that were selected.
    '''
    # Verify that relevant features exist in the DataFrame
    column_labels = [col for col in relevant_features if col in df.columns]
    if not column_labels:
        raise ValueError("No valid columns from 'relevant_features' found in the DataFrame.")

    # Select the specified columns
    selected_columns = df[column_labels]

    # Optionally include categorical columns
    if include_categorical:
        categorical_columns = ['year', 'country_cd']
        categorical_columns = [col for col in categorical_columns if col in df.columns]
        selected_columns = df[categorical_columns].join(selected_columns)

    return selected_columns, column_labels


def split_data_by_country(df, target_column=None, test_country=None, random_seed=None):
    """
    Splits a DataFrame by country code into training and test sets.

    Parameters:
    - df (DataFrame): Input DataFrame with columns 'country_cd', 'year', and the target column.
    - target_column (str, optional): Name of the target column. Defaults to 'regional_capacity_percentage'.
    - test_country (str, optional): Country code to be used as the test set. If None, a random country is selected.
    - random_seed (int, optional): Seed for reproducible random selection of the test country.

    Returns:
    - X_train (DataFrame): Features of the training set.
    - X_test (DataFrame): Features of the test set.
    - y_train (Series): Target values of the training set.
    - y_test (Series): Target values of the test set.
    - train_data (DataFrame): Training data with columns 'country_cd', 'year', and the target column.
    - test_data (DataFrame): Test data with columns 'country_cd', 'year', and the target column.
    """
    
    if target_column is None:
        target_column = 'regional_capacity_percentage'
    
    if random_seed is not None:
        random.seed(random_seed)
    
    train_data = pd.DataFrame(columns=df.columns)
    test_data = pd.DataFrame(columns=df.columns)

    countries = df['country_cd'].unique()
    if test_country is None:
        test_country = random.choice(countries)

    for country in countries:
        country_data = df[df['country_cd'] == country]
        if country == test_country:
            test_data = pd.concat([test_data, country_data])
        else:
            train_data = pd.concat([train_data, country_data])


    X_train = train_data.drop(columns=[target_column, 'year', 'country_cd', 'capacity_mwp'])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column, 'year', 'country_cd', 'capacity_mwp'])
    y_test = test_data[target_column]

    return X_train, X_test, y_train, y_test, train_data, test_data

def split_data_by_year(df, target_column=None, test_years=None, random_seed=None):
    """
    Splits a DataFrame by year into training and test sets.

    Parameters:
    - df (DataFrame): Input DataFrame with columns 'country_cd', 'year', and the target column.
    - target_column (str, optional): Name of the target column. Defaults to 'regional_capacity_percentage'.
    - test_years (list of int, optional): Years to be used as the test set. If None, a random year is selected.
    - random_seed (int, optional): Seed for reproducible random selection of test years.

    Returns:
    - X_train (DataFrame): Features of the training set.
    - X_test (DataFrame): Features of the test set.
    - y_train (Series): Target values of the training set.
    - y_test (Series): Target values of the test set.
    - train_data (DataFrame): Training data with columns 'country_cd', 'year', and the target column.
    - test_data (DataFrame): Test data with columns 'country_cd', 'year', and the target column.
    """
    if target_column is None:
        target_column = 'regional_capacity_percentage'
    
    if random_seed is not None:
        random.seed(random_seed)
    
    train_data = pd.DataFrame(columns=df.columns)
    test_data = pd.DataFrame(columns=df.columns)

    years = df['year'].unique()
    if test_years is None:
        # Randomly select one or more years if none are provided
        test_years = [random.choice(years)]
    elif not isinstance(test_years, list):
        # Convert single year input to a list
        test_years = [test_years]

    for year in years:
        year_data = df[df['year'] == year]
        if year in test_years:
            test_data = pd.concat([test_data, year_data])
        else:
            train_data = pd.concat([train_data, year_data])

    X_train = train_data.drop(columns=[target_column, 'year', 'country_cd', 'capacity_mwp'])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column, 'year', 'country_cd', 'capacity_mwp'])
    y_test = test_data[target_column]

    return X_train, X_test, y_train, y_test, train_data, test_data



def split_data_by_year_and_country(df, target_column='regional_capacity_percentage', test_size=0.2, random_state=None):
    '''
    Splits the data into training and testing sets based on years and countries.

    Parameters:
    - df (DataFrame): The input DataFrame containing the data.
    - target_column (str): The name of the target column.
    - test_size (float): The proportion of data to be used for testing. Defaults to 0.2.
    - random_state (int or None): The seed for reproducibility. Defaults to None.

    Returns:
    - X_train (DataFrame): The feature variables for the training set.
    - X_test (DataFrame): The feature variables for the testing set.
    - y_train (Series): The target variable for the training set.
    - y_test (Series): The target variable for the testing set.
    - train_data (DataFrame): The concatenated training data.
    - test_data (DataFrame): The concatenated testing data.
    '''
    
    grouped = df.groupby(['year', 'country_cd'], observed=False)
    # Shuffle the groups
    grouped_keys = list(grouped.groups.keys())
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(grouped_keys)
    # Calculate the number of groups for training set based on test_size
    n_train_groups = int((1 - test_size) * len(grouped_keys))
    train_keys = grouped_keys[:n_train_groups]
    test_keys = grouped_keys[n_train_groups:]
    # Concatenate the selected groups into training and test sets
    train_data = pd.concat([grouped.get_group(key) for key in train_keys])
    test_data = pd.concat([grouped.get_group(key) for key in test_keys])
    # Extract features and target variables
    X_train = train_data.drop(columns=[target_column, 'year', 'country_cd', 'capacity_mwp'])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column, 'year', 'country_cd', 'capacity_mwp'])
    y_test = test_data[target_column]
    return X_train, X_test, y_train, y_test, train_data, test_data



def train_xgboost_model(x_train, y_train, train_data, model_name="xgboost_model", use_group_kfold=False):
    """
    Train an XGBoost model using either standard CV or GroupKFold.

    Parameters:
    - x_train (DataFrame): The training features.
    - y_train (Series): The training target values.
    - train_data (DataFrame): The original training data containing 'country_cd'.
    - model_name (str): The name to save the model as.
    - use_group_kfold (bool): Whether to use GroupKFold based on 'country_cd'. Default is False.

    Returns:
    - best_regressor: The trained XGBoost model with the best hyperparameters.
    - cv_results: a DataFrame containing the cross-validation results.
    """
    
    # Create an XGBoost regressor
    regressor = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', tree_method="exact")
    # regressor = xgb.XGBRegressor(objective='reg:tweedie', tree_method="exact")

    # Set up the parameter grid for grid search
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
        "n_estimators": [500, 600, 700, 800, 900, 1000],
        "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }
    # param_grid = {
    #     "max_depth": [3],
    #     "n_estimators": [500],
    #     "learning_rate": [0.01, 1]
    # }
    
    if use_group_kfold:
        # Using GroupKFold based on 'year'
        groups = train_data['year']
        n_splits = len(groups.unique())
        group_kfold = GroupKFold(n_splits=n_splits)
        search = GridSearchCV(regressor, param_grid, cv=group_kfold, n_jobs=-1)

        # Iterate through each fold to inspect the splits (optional for debugging)
        if True:  # Set to False if you want to disable debugging output
            for fold, (train_idx, val_idx) in enumerate(group_kfold.split(x_train, y_train, groups=groups)):
                print(f"Fold {fold + 1}")
                print(f"Training data shape: {x_train.iloc[train_idx].shape}")
                print(f"Validation data shape: {x_train.iloc[val_idx].shape}")
                print(f"Unique training groups: {groups.iloc[train_idx].unique()}")
                print(f"Unique validation groups: {groups.iloc[val_idx].unique()}")
                print("-" * 40)

        search.fit(x_train, y_train, groups=groups)
    
    else:
        # Standard cross-validation
        search = GridSearchCV(regressor, param_grid, cv=10, n_jobs=-1)
        search.fit(x_train, y_train)

    # Access the cross-validation results
    cv_results = pd.DataFrame(search.cv_results_)
    
    print(" Results from Grid Search ")
    print("\n The best estimator across ALL searched params:\n", search.best_estimator_)
    print("\n The best score across ALL searched params:\n", search.best_score_)
    print("\n The best parameters across ALL searched params:\n", search.best_params_)

    # Get the best regressor with the best hyperparameters
    best_regressor = search.best_estimator_

    # Log the model with MLflow
    mlflow.xgboost.log_model(best_regressor, model_name)

    return best_regressor, cv_results


def groupby_validation(df, target_column='regional_capacity_percentage'):
    '''
    Groups the validation data by year and country code and splits it into input features and target variables for model feeding.

    Parameters:
    - df (DataFrame): The DataFrame to split into groups.
    - target_column (str): The name of the target column. Defaults to 'regional_capacity_percentage'.

    Returns:
    - val_data (DataFrame): The concatenated validation data.
    - X_val (DataFrame): The feature variables for the validation set.
    - y_val (Series): The target variable for the validation set.
    '''
    val_grouped = df.groupby(['year', 'country_cd'], observed=False)
    val_grouped_keys = list(val_grouped.groups.keys())
    val_data = pd.concat([val_grouped.get_group(key) for key in val_grouped_keys])
    X_val = val_data.drop(columns=[target_column, 'year', 'country_cd', 'nuts_cd', 'nuts_name', 'national_capacity_mwp', 'capacity_mwp'])
    y_val = val_data[target_column]
    return val_data, X_val, y_val
    
def scale_capacity_to_national(df):
    '''
    Adjusts the predicted regional cumulative capacity so that it sums to the national capacity.

    Parameters:
    - df (DataFrame): The input DataFrame containing the predicted regional cumulative capacity and grouping columns.

    Returns:
    - df (DataFrame): The DataFrame with the predicted regional cumulative capacity adjusted to sum to the national capacity.
    '''
    # Set negative values to 0 
    df['clipped_predicted_capacity_percentage'] = df['predicted_capacity_percentage'].clip(lower=0)
    
    # Calculate the sum of clipped 'Predicted Regional cumulative capacity (% National)' for each group
    group_sums = df.groupby(['year', 'country_cd'], observed=False)['clipped_predicted_capacity_percentage'].transform('sum')

    # Calculate the scaling factor for each row
    df['scaling_factor'] = 100 / group_sums

    # Apply the scaling factor to adjust the values
    df['scaled_predicted_capacity_percentage'] = df['clipped_predicted_capacity_percentage'] * df['scaling_factor']

    # Drop the 'Scaling Factor' column if you don't need it
    df.drop(columns=['scaling_factor', 'clipped_predicted_capacity_percentage'], inplace=True)

    return df



def convert_from_percentage_to_MW(df):
    '''
    Converts percentage regional capacity and scaled percentage regional capacity to regional capacity in MW and scaled regional capacity in MW.

    Parameters:
    - df (DataFrame): The input DataFrame containing percentage regional capacity and scaled percentage regional capacity columns.

    Returns:
    - df (DataFrame): The DataFrame with regional capacity and scaled regional capacity in MW.
    '''
    
    # Check if the necessary columns exist before performing the calculation
    if 'predicted_capacity_percentage' in df.columns and 'national_capacity_mwp' in df.columns:
        df['predicted_capacity_mwp'] = (df['predicted_capacity_percentage'] / 100) * df['national_capacity_mwp']
    
    if 'scaled_predicted_capacity_percentage' in df.columns and 'national_capacity_mwp' in df.columns:
        df['scaled_predicted_capacity_mwp'] = (df['scaled_predicted_capacity_percentage'] / 100) * df['national_capacity_mwp']
    
    return df


def calculate_predicted_national_capacity(df):
    '''
    Calculates the predicted national capacity and predicted scaled national capacity, which is the sum of predicted regional capacity and scaled regionsl capacity.

    Parameters:
    - df (DataFrame): The input DataFrame containing regional capacity data.

    Returns:
    - df (DataFrame): The DataFrame with predicted national capacity calculated and added as new columns.
    '''
    # Group the data by columns 'year' and 'country_cd' and calculate the sum of 'Predicted Regional cumulative capacity (MW)'
    result = df.groupby(['year', 'country_cd'], as_index=False, observed=False)['predicted_capacity_mwp'].sum()
    result.rename(columns={'predicted_capacity_mwp': 'predicted_national_capacity_mwp'}, inplace=True)
    # Save the result back to the original DataFrame
    df = df.merge(result, on=['year', 'country_cd'], how='left')
    
    # Calculate scaled predicted national capacity
    result2 = df.groupby(['year', 'country_cd'], as_index=False, observed=False)['scaled_predicted_capacity_mwp'].sum()
    result2.rename(columns={'scaled_predicted_capacity_mwp': 'scaled_predicted_national_capacity_mwp'}, inplace=True)
    # Save the result back to the original DataFrame
    df = df.merge(result2, on=['year', 'country_cd'], how='left')

    # Group the data by columns 'year' and 'country_cd' and calculate the sum of 'Predicted Regional cumulative capacity (%)'
    result3 = df.groupby(['year', 'country_cd'], as_index=False, observed=False)['predicted_capacity_percentage'].sum()
    result3.rename(columns={'predicted_capacity_percentage': 'predicted_national_capacity_percentage'}, inplace=True)
    # Save the result back to the original DataFrame
    df = df.merge(result3, on=['year', 'country_cd'], how='left')
    
    # Calculate scaled predicted national capacity
    result4 = df.groupby(['year', 'country_cd'], as_index=False, observed=False)['scaled_predicted_capacity_percentage'].sum()
    result4.rename(columns={'scaled_predicted_capacity_percentage': 'scaled_predicted_national_capacity_percentage'}, inplace=True)
    # Save the result back to the original DataFrame
    df = df.merge(result4, on=['year', 'country_cd'], how='left')
    
    return df


def gb_results(df, log_to_mlflow=False, metric_label="gb"):
    '''
    Calculates and prints evaluation metrics for European Union (EU) validation results.

    Parameters:
    - df (DataFrame): The input DataFrame containing the validation results.
    - log_to_mlflow (bool): Whether to log the metrics to MLflow. Defaults to False.
    - metric_label (str): Text to include in the metric name. Defaults to "EU".

    Returns:
    None
    '''
    # gb validation results
    r2_test = r2_score(df['national_capacity_mwp'], df['predicted_national_capacity_mwp'])
    mse = mean_squared_error(df['national_capacity_mwp'], df['predicted_national_capacity_mwp'])
    mae = mean_absolute_error(df['national_capacity_mwp'], df['predicted_national_capacity_mwp'])
    rmse = np.sqrt(mean_squared_error(df['national_capacity_mwp'], df['predicted_national_capacity_mwp']))
    mape = mean_absolute_percentage_error(df['national_capacity_mwp'], df['predicted_national_capacity_mwp'])

    print('R2 test score is ', r2_test)
    print('The MSE is:', mse)
    print('The MAE is:', mae)
    print('RMSE is:', rmse)
    print('MAPE is:', mape)

    # Log metrics to MLflow if log_to_mlflow is True
    if log_to_mlflow:
        mlflow.log_metric(f"R2_{metric_label}", r2_test)
        mlflow.log_metric(f"MSE_{metric_label}", mse)
        mlflow.log_metric(f"MAE_{metric_label}", mae)
        mlflow.log_metric(f"RMSE_{metric_label}", rmse)
        mlflow.log_metric(f"MAPE_{metric_label}", mape)

    return




def gb_scaled_results(df, log_to_mlflow=False, metric_label="gb_scaled"):
    '''
    Calculates and prints evaluation metrics for gb results with scaled predicted national capacity.

    Parameters:
    - df (DataFrame): The input DataFrame containing the validation results with scaled predicted national capacity.
    - log_to_mlflow (bool): Whether to log the metrics to MLflow. Defaults to False.
    - metric_label (str): Text to include in the metric name. Defaults to "Scaled EU".

    Returns:
    None
    '''
    # gb validation results
    r2_test = r2_score(df['national_capacity_mwp'], df['scaled_predicted_national_capacity_mwp'])
    mse = mean_squared_error(df['national_capacity_mwp'], df['scaled_predicted_national_capacity_mwp'])
    mae = mean_absolute_error(df['national_capacity_mwp'], df['scaled_predicted_national_capacity_mwp'])
    rmse = np.sqrt(mean_squared_error(df['national_capacity_mwp'], df['scaled_predicted_national_capacity_mwp']))
    mape = mean_absolute_percentage_error(df['national_capacity_mwp'], df['scaled_predicted_national_capacity_mwp'])

    print('R2 test score is ', r2_test)
    print('The MSE is:', mse)
    print('The MAE is:', mae)
    print('RMSE is:', rmse)
    print('MAPE is:', mape)

    # Log metrics to MLflow if log_to_mlflow is True
    if log_to_mlflow:
        mlflow.log_metric(f"R2_{metric_label}", r2_test)
        mlflow.log_metric(f"MSE_{metric_label}", mse)
        mlflow.log_metric(f"MAE_{metric_label}", mae)
        mlflow.log_metric(f"RMSE_{metric_label}", rmse)
        mlflow.log_metric(f"MAPE_{metric_label}", mape)

    return


def training_scaled_results(combined_data, log_to_mlflow=False):
    '''
    Calculates, prints, and optionally logs scaled evaluation metrics.

    Parameters:
    - combined_data (DataFrame): DataFrame containing combined data.
    - log_to_mlflow (bool): Whether to log metrics to MLflow. Default is False.

    Returns:
    None
    '''
    combined_data = combined_data.dropna(subset=['regional_capacity_percentage', 'scaled_predicted_capacity_percentage'])

    train_data = combined_data[combined_data['set'] == 'train']
    test_data = combined_data[combined_data['set'] == 'test']
    
    r2_scaled = r2_score(combined_data["regional_capacity_percentage"], combined_data["scaled_predicted_capacity_percentage"])
    mse_scaled = mean_squared_error(combined_data["regional_capacity_percentage"], combined_data["scaled_predicted_capacity_percentage"])
    mae_scaled = mean_absolute_error(combined_data["regional_capacity_percentage"], combined_data["scaled_predicted_capacity_percentage"])
    rmse_scaled = np.sqrt(mean_squared_error(combined_data["regional_capacity_percentage"], combined_data["scaled_predicted_capacity_percentage"]))
    mape_scaled = mean_absolute_percentage_error(combined_data["regional_capacity_percentage"], combined_data["scaled_predicted_capacity_percentage"])

    # Print the results
    print('Error metrics for the entire dataset: ')
    print('R2 score is ', r2_scaled)
    print('The MSE is:', mse_scaled)
    print('The MAE is:', mae_scaled)
    print('RMSE is:', rmse_scaled)
    print('MAPE is:', mape_scaled)

    r2_scaled_test = r2_score(test_data["regional_capacity_percentage"], test_data["scaled_predicted_capacity_percentage"])
    mse_scaled_test = mean_squared_error(test_data["regional_capacity_percentage"], test_data["scaled_predicted_capacity_percentage"])
    mae_scaled_test = mean_absolute_error(test_data["regional_capacity_percentage"], test_data["scaled_predicted_capacity_percentage"])
    rmse_scaled_test = np.sqrt(mean_squared_error(test_data["regional_capacity_percentage"], test_data["scaled_predicted_capacity_percentage"]))
    mape_scaled_test = mean_absolute_percentage_error(test_data["regional_capacity_percentage"], test_data["scaled_predicted_capacity_percentage"])

    if log_to_mlflow:
        mlflow.log_metric("R2_scaled", r2_scaled)
        mlflow.log_metric("MSE_scaled", mse_scaled)
        mlflow.log_metric("MAE_scaled", mae_scaled)
        mlflow.log_metric("RMSE_scaled", rmse_scaled)
        mlflow.log_metric("MAPE_scaled", mape_scaled)
        mlflow.log_metric("R2_test_scaled", r2_scaled_test)
        mlflow.log_metric("MSE_test_scaled", mse_scaled_test)
        mlflow.log_metric("MAE_test_scaled", mae_scaled_test)
        mlflow.log_metric("RMSE_test_scaled", rmse_scaled_test)
        mlflow.log_metric("MAPE_test_scaled", mape_scaled_test)

    # Print the results
    print('Error metrics for the test points: ')
    print('R2 test score is ', r2_scaled_test)
    print('The MSE is:', mse_scaled_test)
    print('The MAE is:', mae_scaled_test)
    print('RMSE is:', rmse_scaled_test)
    print('MAPE is:', mape_scaled_test)
    
    return


def training_results(y_train, y_test, y_pred, y_pred_t, log_to_mlflow=False):
    '''
    Calculates, prints, and optionally logs training and test evaluation metrics.

    Parameters:
    - y_train (array-like): True labels for training data.
    - y_test (array-like): True labels for test data.
    - y_pred (array-like): Predicted labels for test data.
    - y_pred_t (array-like): Predicted labels for training data.
    - log_to_mlflow (bool): Whether to log metrics to MLflow. Default is False.

    Returns:
    None
    '''
    r2_train = r2_score(y_train, y_pred_t)
    r2_test = r2_score(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    mape_test = mean_absolute_percentage_error(y_test, y_pred)

    # Print the results
    print('R2 training score is ', r2_train)
    print('R2 test score is ', r2_test)
    print('The MSE is:', mse_test)
    print('The MAE is:', mae_test)
    print('RMSE is:', rmse_test)
    print('MAPE is:', mape_test)


    if log_to_mlflow:
        # Log metrics to MLflow
        mlflow.log_metric("R2_train", r2_train)
        mlflow.log_metric("R2_test", r2_test)
        mlflow.log_metric("MSE_test", mse_test)
        mlflow.log_metric("MAE_test", mae_test)
        mlflow.log_metric("RMSE_test", rmse_test)
        mlflow.log_metric("MAPE_test", mape_test)

    return

def training_results_mw(df):

    train_data = df[df['set'] == 'train']
    # Select rows where the 'set' column equals 'test'
    test_data = df[df['set'] == 'test']

    print('Results for the unscaled data')
    print('R2 training score is ',r2_score(train_data['capacity_mwp'], train_data['predicted_capacity_mwp']))
    print('R2 test score is ', r2_score(test_data['capacity_mwp'], test_data['predicted_capacity_mwp']))
    print('The MSE is:', mean_squared_error(test_data['capacity_mwp'], test_data['predicted_capacity_mwp']))
    print('The MAE is:', mean_absolute_error(test_data['capacity_mwp'], test_data['predicted_capacity_mwp']))
    print('RMSE is:',np.sqrt(mean_squared_error(test_data['capacity_mwp'], test_data['predicted_capacity_mwp'])))
    print('MAPE is:', mean_absolute_percentage_error(test_data['capacity_mwp'], test_data['predicted_capacity_mwp']))

    print('Results for the scaled data')
    print('R2 training score is ',r2_score(train_data['capacity_mwp'], train_data['scaled_predicted_capacity_mwp']))
    print('R2 test score is ', r2_score(test_data['capacity_mwp'], test_data['scaled_predicted_capacity_mwp']))
    print('The MSE is:', mean_squared_error(test_data['capacity_mwp'], test_data['scaled_predicted_capacity_mwp']))
    print('The MAE is:', mean_absolute_error(test_data['capacity_mwp'], test_data['scaled_predicted_capacity_mwp']))
    print('RMSE is:',np.sqrt(mean_squared_error(test_data['capacity_mwp'], test_data['scaled_predicted_capacity_mwp'])))
    print('MAPE is:', mean_absolute_percentage_error(test_data['capacity_mwp'], test_data['scaled_predicted_capacity_mwp']))

    return 

def generate_predictions_training_data(best_regressor, X_train, X_test, train_data, test_data, Training_df):
    """
    Generate predictions and combine data for data included in training.

    Parameters:
    - best_regressor: The trained regressor model.
    - X_train: The features of the training set.
    - X_test: The features of the test set.
    - train_data: DataFrame containing training data.
    - test_data: DataFrame containing test data.
    - Training_df: DataFrame containing the original training data.

    Returns:
    - y_train: Predicted values for the training set.
    - y_test: Predicted values for the test set.
    - y_pred: Predicted values for the combined dataset.
    - y_pred_t: Predicted values for the training set.
    - combined_data: Combined DataFrame with predictions.
    """
    # Predictions
    y_pred = best_regressor.predict(X_test)
    y_pred_t = best_regressor.predict(X_train)

    # Add 'set' column
    train_data["set"] = "train"
    test_data["set"] = "test"

    # Add predicted values to train_data and test_data
    train_data["predicted_capacity_percentage"] = y_pred_t
    test_data["predicted_capacity_percentage"] = y_pred

    # Combine train_data and test_data
    combined_data = pd.concat([train_data, test_data])

    combined_data = Training_df[['nuts_cd', 'nuts_name', 'national_capacity_mwp']].join(combined_data)
    

    # Convert from percentage to MW
    combined_data = convert_from_percentage_to_MW(combined_data)

    return y_pred, y_pred_t, combined_data


def apply_model_to_all_countries(best_regressor, df, column_labels, regional_data):
    """
    Apply the trained regressor model to the entire data set which includes countries whithout regional capacity data.

    Parameters:
    - best_regressor: The trained regressor model.
    - df: DataFrame containing all data.
    - column_labels: Labels of selected columns.
    - regional_data: DataFrame containing the training data with column set which is either train or test.

    Returns:
    - val_data: Processed DataFrame which includes the results from applying the model.
    """
    # Filter rows where 'national_capacity_mwp' is not NaN
    Validation_df = df[df['national_capacity_mwp'].notna()]

    # Select columns based on column_labels
    val_selected_columns = Validation_df[column_labels]

    # Join selected columns with additional columns
    val_selected_columns = Validation_df[['country_cd', 'year', 'nuts_cd', 'nuts_name', 'national_capacity_mwp']].join(val_selected_columns)

    # Group validation data
    val_data, X_val, y_val = groupby_validation(val_selected_columns)

    # Predict using the best regressor
    val_data['predicted_capacity_percentage'] = best_regressor.predict(X_val)

    # Scale capacity to national
    val_data = scale_capacity_to_national(val_data)

    # Convert from percentage to MW
    val_data = convert_from_percentage_to_MW(val_data)

    # Calculate predicted national capacity
    val_data = calculate_predicted_national_capacity(val_data)

    # Calculate residuals
    val_data['national_residual_mwp'] = val_data['national_capacity_mwp'] - val_data['predicted_national_capacity_mwp']
    val_data['scaled_national_residual_mwp'] = val_data['national_capacity_mwp'] - val_data['scaled_predicted_national_capacity_mwp']
    val_data['national_residual_percentage'] = 100 - val_data['predicted_national_capacity_percentage']
    val_data['scaled_national_residual_percentage'] = 100 - val_data['scaled_predicted_national_capacity_percentage']
    val_data['residual_percentage'] = val_data['regional_capacity_percentage'] - val_data['predicted_capacity_percentage']
    val_data['scaled_residual_percentage'] = val_data['regional_capacity_percentage'] - val_data['scaled_predicted_capacity_percentage']
    val_data['residual_mwp'] = val_data['capacity_mwp'] - val_data['predicted_capacity_mwp']
    val_data['scaled_residual_mwp'] = val_data['capacity_mwp'] - val_data['scaled_predicted_capacity_mwp']


    # Replace 'UK' with 'GB'
    # Define a mapping between NUTS country codes and ISO country codes
    country_code_mapping = {
        'UK':'GB',      
    }
    
    # Create a new 'Country' column based on ISO country codes
    val_data['country'] = val_data['country_cd'].apply(lambda x: pycountry.countries.get(alpha_2=country_code_mapping.get(x, x)).name)

    val_data = val_data.merge(regional_data[['nuts_cd', 'year', 'set']], on=['nuts_cd', 'year'], how='left')

    return val_data


def calculate_mape_by_country(df, country_column, true_column, pred_column):
    '''
    Calculate MAPE by country and average MAPE per country across all years.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - country_column (str): The column name for country.
    - true_column (str): The column name for true values.
    - pred_column (str): The column name for predicted values.

    Returns:
    - mape_by_country (DataFrame): A DataFrame with countries, their MAPE values by year, 
      and the average MAPE per country across all years.
    '''
    
    # Function to calculate MAPE for each group
    def safe_mape(group):
        if group[true_column].shape[0] == 0:
            return float('nan')  # Return NaN for empty groups
        return mean_absolute_percentage_error(group[true_column], group[pred_column])
    
    # Calculate MAPE by country and year
    mape_by_year_country = df.groupby([country_column, 'year']).apply(safe_mape).reset_index(name='MAPE')
    
    # Convert MAPE to percentage
    mape_by_year_country['MAPE'] *= 100
    
    # Calculate average MAPE per country across all years
    mape_by_country = mape_by_year_country.groupby(country_column)['MAPE'].mean().reset_index(name='Average_MAPE')

    # Sort DataFrame by 'Average MAPE' in ascending order
    mape_by_country = mape_by_country.sort_values(by='Average_MAPE', ascending=True)
    # Round 'Average MAPE' to one decimal point
    mape_by_country['Average_MAPE'] = mape_by_country['Average_MAPE'].round(1)
    
    return mape_by_year_country, mape_by_country



def calculate_shap_contributions(regional_data, shap_values, x_train_test, desired_nuts_cd="UKE31", start_year=2010, end_year=2023):
    """
    Calculate the SHAP (Shapley Additive Explanations) contributions for factors influencing solar PV capacity at the regional level.

    This function filters the regional data for the specified NUTS code and years, computes the mean SHAP values for each factor 
    within the selected data, and calculates the percentage contribution of each factor. It then visualizes the results in a bar chart 
    and prints a DataFrame with the contributions.

    Parameters:
    -----------
    regional_data : pd.DataFrame
        A DataFrame containing regional data, including 'nuts_cd' (NUTS code) and 'year' columns.
        
    shap_values : numpy.ndarray
        An array of SHAP values corresponding to the training/testing data, where each element represents the contribution of a factor.
        
    x_train_test : pd.DataFrame
        The feature matrix (training or testing data) used to compute the SHAP values, with columns representing the different factors.

    desired_nuts_cd : str, optional, default="UKE31"
        The NUTS code for the region of interest. This filters the regional data to focus on the specified region.

    start_year : int, optional, default=2010
        The starting year for filtering the regional data. Only data from this year and onward will be included.

    end_year : int, optional, default=2023
        The ending year for filtering the regional data. Only data up to this year will be included.

    Raises:
    -------
    ValueError
        If no matching data is found for the given region and year range, an error is raised.
    
    Returns:
    --------
    None
        The function outputs the percentage contribution of each factor as a bar plot and a printed DataFrame.
    """
    # Ensure the 'year' column is numeric
    regional_data['year'] = pd.to_numeric(regional_data['year'], errors='coerce')

    # Filter data for the specified region and years
    filtered_data = regional_data[
        (regional_data['nuts_cd'] == desired_nuts_cd) &
        (regional_data['year'].between(start_year, end_year))
    ]

    # Ensure there are matching rows
    if filtered_data.empty:
        raise ValueError(f"No matching data found for {desired_nuts_cd} between {start_year} and {end_year}")

    # Get the indices of the matching rows
    filtered_indices = filtered_data.index

    # Get the SHAP values for the filtered test points
    test_positions = [x_train_test.index.get_loc(idx) for idx in filtered_indices]

    # Extract SHAP values for the selected rows and compute the mean
    shap_values_subset = [shap_values[pos] for pos in test_positions]
    mean_shap_values = np.mean(shap_values_subset, axis=0)

    # Calculate total absolute contribution
    total_contribution = np.sum(np.abs(mean_shap_values))

    # Calculate percentage contribution for each factor
    percentage_contributions = (mean_shap_values / total_contribution) * 100

    # Create a DataFrame for better visualization
    percentage_df = pd.DataFrame({
        'Factor': x_train_test.columns.tolist(),
        'Mean SHAP Value': mean_shap_values,
        'Percentage Contribution (%)': percentage_contributions
    })

    # Sort by percentage contribution for better readability
    percentage_df = percentage_df.sort_values(by='Percentage Contribution (%)', ascending=False)

    # Display the table
    print(percentage_df)

    # Plot the percentage contributions
    plt.figure(figsize=(10, 6))
    plt.barh(percentage_df['Factor'], percentage_df['Percentage Contribution (%)'], color='skyblue')
    plt.xlabel('Percentage Contribution (%)')
    plt.ylabel('Factors')
    plt.title(f'Percentage Contribution of Each Factor ({desired_nuts_cd}, {start_year}-{end_year})')
    plt.gca().invert_yaxis()  # Largest contributor at the top
    plt.show()


def calculate_shap_contributions_by_prefix(shap_values, x_train_test, feature_prefix="2"):
    """
    Calculate the SHAP contributions for features whose names start with a given prefix.
    
    This function filters the features in the input data based on the given prefix, computes the 
    mean SHAP values for these features, and calculates their total and percentage contribution 
    relative to all features.

    Parameters:
    -----------
    shap_values : numpy.ndarray
        An array of SHAP values corresponding to the training/testing data, where each element represents the contribution of a feature.
        
    x_train_test : pd.DataFrame
        The feature matrix (training or testing data) used to compute the SHAP values, with columns representing the different features.
        
    feature_prefix : str, optional, default="2"
        The prefix for filtering feature names. Only features whose names start with this prefix will be considered.

    Returns:
    --------
    None
        The function prints the total contribution of the filtered features, their percentage contribution 
        relative to all features, and a DataFrame with the mean SHAP values for these features.
    """
    # 1. Filter the features that start with the given prefix
    features_starting_with_prefix = [col for col in x_train_test.columns if col.startswith(feature_prefix)]

    # 2. Get the SHAP values corresponding to these filtered features
    shap_values_filtered = shap_values[:, x_train_test.columns.isin(features_starting_with_prefix)]

    # 3. Compute the mean absolute SHAP values for the filtered features
    mean_shap_values_filtered = np.abs(shap_values_filtered).mean(axis=0)

    # 4. Compute the total contribution of these features
    total_contribution_filtered = mean_shap_values_filtered.sum()

    # 5. Compute the total contribution of all features
    mean_shap_values_all = np.abs(shap_values).mean(axis=0)
    total_contribution_all = mean_shap_values_all.sum()

    # 6. Calculate the percentage contribution
    percentage_contribution = (total_contribution_filtered / total_contribution_all) * 100

    # Print the results
    print(f"Total contribution of factors starting with '{feature_prefix}': {total_contribution_filtered}")
    print(f"Percentage contribution of factors starting with '{feature_prefix}': {percentage_contribution:.0f}%")

    # Optional: Display contribution by each factor starting with the specified prefix
    shap_summary_df_filtered = pd.DataFrame({
        'Feature': features_starting_with_prefix,
        'Mean SHAP Value': mean_shap_values_filtered
    }).sort_values(by='Mean SHAP Value', ascending=False)

    print(shap_summary_df_filtered)



def plot_shap_waterfall_by_nuts(regional_data, shap_values, x_train_test, explainer, desired_nuts_cd="UKE31", start_year=2010, end_year=2023, save_figure=False, file_path="shap_waterfall_plot.eps"):
    """
    Generate and optionally save a SHAP waterfall plot for the mean SHAP values of a given region and year range.

    This function filters the regional data for a specific NUTS code and year range, calculates the mean SHAP values
    for the corresponding data points, and generates a SHAP waterfall plot. The plot can be saved as an image file.

    Parameters:
    -----------
    regional_data : pd.DataFrame
        The regional data containing columns like 'nuts_cd' and 'year'.
        
    shap_values : numpy.ndarray
        An array of SHAP values corresponding to the training/testing data.

    x_train_test : pd.DataFrame
        The feature matrix (training or testing data) used to compute the SHAP values.

    explainer : shap.Explainer
        The SHAP explainer object used to generate SHAP values.

    desired_nuts_cd : str, optional, default="UKE31"
        The NUTS code for the region of interest. This filters the regional data to focus on the specified region.

    start_year : int, optional, default=2010
        The starting year for filtering the regional data. Only data from this year and onward will be included.

    end_year : int, optional, default=2023
        The ending year for filtering the regional data. Only data up to this year will be included.

    save_figure : bool, optional, default=False
        If True, the generated plot will be saved as an image file.

    file_path : str, optional, default="shap_waterfall_plot.eps"
        The file path where the plot will be saved if `save_figure` is True.

    Returns:
    --------
    None
        The function generates a SHAP waterfall plot and optionally saves it as an image file.
    """
    # Convert 'year' to numeric
    regional_data['year'] = pd.to_numeric(regional_data['year'], errors='coerce')

    # Filter regional_data for the desired NUTS code and year range
    filtered_data = regional_data[
        (regional_data['nuts_cd'] == desired_nuts_cd) &
        (regional_data['year'].between(start_year, end_year))
    ]

    # Ensure there are matching rows
    if filtered_data.empty:
        raise ValueError(f"No matching data found for {desired_nuts_cd} between {start_year} and {end_year}")

    # Get the indices of the matching rows
    filtered_indices = filtered_data.index

    # Get the SHAP values for the filtered test points
    test_positions = [x_train_test.index.get_loc(idx) for idx in filtered_indices]

    # Extract SHAP values for the selected rows and compute the mean
    shap_values_subset = [shap_values[pos] for pos in test_positions]
    mean_shap_values = np.mean(shap_values_subset, axis=0)

    # Create an Explanation object for the mean SHAP values
    mean_explanation = shap.Explanation(
        values=mean_shap_values,
        base_values=np.mean([explainer.expected_value] * len(test_positions)),  
        data=x_train_test.iloc[filtered_indices].mean(axis=0),  # Mean feature values for the filtered points
        feature_names=x_train_test.columns.tolist()  # Feature names
    )

    # Generate a SHAP waterfall plot for the mean contributions
    shap.plots.waterfall(mean_explanation, show=False)
    
    # Optionally save the figure and display the plot
    if save_figure:
        plt.savefig(file_path, format="eps", bbox_inches="tight")
    
    # Display the plot, regardless of whether it was saved
    plt.show()
    
    # Close the plot after showing it
    plt.close()

    
# -----------------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------------------------------



def plot_actual_vs_predicted_regional_capacity(df, capacity_unit='percentage', scaled=False):
    '''
    Plot actual vs. predicted regional capacity.

    Parameters:
        df (DataFrame): The dataframe containing the data.
        capacity_unit (str, optional): The unit of capacity to be used for plotting. Can be 'percentage' or 'MW'. Default is 'percentage'.
        scaled (bool, optional): Indicates whether to plot scaled predicted capacity or not. Default is False.

    Returns:
        None

    Example:
        plot_actual_vs_predicted(df, capacity_unit='MW', scaled=True)
    '''
    if capacity_unit == 'percentage':
        x_col = 'regional_capacity_percentage'
        y_col = 'predicted_capacity_percentage'
        x_label = 'Actual Regional Capacity (%)'
        y_label = 'Predicted Regional Capacity (%)'
    elif capacity_unit == 'MW':
        x_col = 'capacity_mwp'
        y_col = 'predicted_capacity_mwp'
        x_label = 'Actual Regional Capacity (MW)'
        y_label = 'Predicted Regional Capacity (MW)'
    else:
        raise ValueError("Invalid capacity_unit. Use 'percentage' or 'MW'.")

    if scaled:
        y_col = 'scaled_' + y_col

    fig = px.scatter(df, x=x_col, y=y_col, hover_data=["year", "country_cd"])
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label
    )

    x_min = min(df[x_col].min(), df[y_col].min())
    x_max = max(df[x_col].max(), df[y_col].max())

    fig.add_shape(
        type='line',
        x0=x_min,
        y0=x_min,
        x1=x_max,
        y1=x_max,
        line=dict(
            color='darkorange',
            dash='dash',
            width=2
        )
    )
    fig.update_layout(autosize=False, width=700, height=700)
    fig.show()



def plot_set(df, set_value, capacity_unit):
    set_df = df[df['set'] == set_value]
    return plot_actual_vs_predicted_regional_capacity(set_df, capacity_unit)





def plot_actual_vs_predicted_national_capacity(df, by_factor=None, factor=None, scaled=False, unit='MW'):
    '''
    Plot actual vs. predicted national capacity.

    Parameters:
        df (DataFrame): The dataframe containing the data.
        by_factor (bool, optional): Indicates whether to plot by a factor or not. If True, the plot will be grouped by the specified factor. Default is None.
        factor (str, optional): The factor by which the data should be grouped if by_factor is True (e.g., 'country', 'year'). Default is None.
        scaled (bool, optional): Indicates whether to plot scaled predicted capacity or not. Default is False.
        unit (str, optional): The unit of capacity to plot. Either 'MW' or '%'. Default is 'MW'.

    Returns:
        None

    Example:
        plot_actual_vs_predicted_national_capacity(val_data, by_factor=True, factor='country', scaled=True, unit='%')
    '''

    # Set column names based on the unit
    if unit == 'MW':
        actual_col = 'national_capacity_mwp'
        if scaled:
            y_col = 'scaled_predicted_national_capacity_mwp'
        else:
            y_col = 'predicted_national_capacity_mwp'
        x_label = 'Actual National Capacity (MW)'
        y_label = 'Predicted National Capacity (MW)' if not scaled else 'Scaled Predicted National Capacity (MW)'
    elif unit == '%':
        actual_col = 'national_capacity_percentage'
        
        # Check if the column exists, if not create it with 100 as the default value
        if actual_col not in df.columns:
            df[actual_col] = 100
        
        if scaled:
            y_col = 'scaled_predicted_national_capacity_percentage'
        else:
            y_col = 'predicted_national_capacity_percentage'
        x_label = 'Actual National Capacity (%)'
        y_label = 'Predicted National Capacity (%)' if not scaled else 'Scaled Predicted National Capacity (%)'
    else:
        raise ValueError("Invalid unit. Please choose 'MW' or '%'.")

    if by_factor:
        for name, group in df.groupby(factor):
            if group.empty:
                x_min = 0
                x_max = 0
                y_min = 0
                y_max = 0
            else:
                x_min = min(group[actual_col])
                x_max = max(group[actual_col])
                y_min = min(group[y_col])
                y_max = max(group[y_col])

            fig = px.scatter(group, x=actual_col, y=y_col, title=name, hover_data=["year", actual_col, y_col])
            fig.update_traces(mode='markers')
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)

            # Find the maximum and minimum values to set the line's endpoints
            p1 = max(x_max, y_max)
            p2 = min(x_min, y_min)

            fig.add_shape(
                type='line',
                x0=p2,
                y0=p2,
                x1=p1,
                y1=p1,
                line=dict(
                    color='darkorange',
                    dash='dash',
                    width=2)
            )

            fig.update_layout(
                autosize=False,
                width=500,
                height=500
            )

            fig.show()
    else:
        fig = px.scatter(df, x=actual_col, y=y_col, hover_data=["year", "country_cd", actual_col, y_col])
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label
        )

        x_min = min(min(df[actual_col]), min(df[y_col]))
        x_max = max(max(df[actual_col]), max(df[y_col]))

        fig.add_shape(
            type='line',
            x0=x_min,
            y0=x_min,
            x1=x_max,
            y1=x_max,
            line=dict(
                color='darkorange',
                dash='dash',
                width=2
            )
        )
        fig.update_layout(autosize=False, width=700, height=700)
        fig.show()


def plot_actual_vs_predicted_regional_capacity_by_factor(df, by_factor=None, factor=None, scaled=False, unit='MW'):
    '''
    Plot actual vs. predicted national capacity.

    Parameters:
        df (DataFrame): The dataframe containing the data.
        by_factor (bool, optional): Indicates whether to plot by a factor or not. If True, the plot will be grouped by the specified factor. Default is None.
        factor (str, optional): The factor by which the data should be grouped if by_factor is True (e.g., 'nuts_cd', 'year'). Default is None.
        scaled (bool, optional): Indicates whether to plot scaled predicted capacity or not. Default is False.
        unit (str, optional): The unit of capacity to plot. Either 'MW' or '%'. Default is 'MW'.

    Returns:
        None

    Example:
        plot_actual_vs_predicted_national_capacity(val_data, by_factor=True, factor='country', scaled=True, unit='%')
    '''

    # Set column names based on the unit
    if unit == 'MW':
        actual_col = 'capacity_mwp'
        if scaled:
            y_col = 'scaled_predicted_capacity_mwp'
        else:
            y_col = 'predicted_capacity_mwp'
        x_label = 'Actual Regional Capacity (MW)'
        y_label = 'Predicted Regional Capacity (MW)' if not scaled else 'Scaled Predicted Regional Capacity (MW)'
    elif unit == '%':
        actual_col = 'regional_capacity_percentage'
        
        # Check if the column exists, if not create it with 100 as the default value
        if actual_col not in df.columns:
            df[actual_col] = 100
        
        if scaled:
            y_col = 'scaled_predicted_capacity_percentage'
        else:
            y_col = 'predicted_capacity_percentage'
        x_label = 'Actual Regional Capacity (%)'
        y_label = 'Predicted Regional Capacity (%)' if not scaled else 'Scaled Predicted Regional Capacity (%)'
    else:
        raise ValueError("Invalid unit. Please choose 'MW' or '%'.")

    if by_factor:
        for name, group in df.groupby(factor):
            if group.empty:
                x_min = 0
                x_max = 0
                y_min = 0
                y_max = 0
            else:
                x_min = min(group[actual_col])
                x_max = max(group[actual_col])
                y_min = min(group[y_col])
                y_max = max(group[y_col])

            fig = px.scatter(group, x=actual_col, y=y_col, title=name, hover_data=["year", actual_col, y_col])
            fig.update_traces(mode='markers')
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)

            # Find the maximum and minimum values to set the line's endpoints
            p1 = max(x_max, y_max)
            p2 = min(x_min, y_min)

            fig.add_shape(
                type='line',
                x0=p2,
                y0=p2,
                x1=p1,
                y1=p1,
                line=dict(
                    color='darkorange',
                    dash='dash',
                    width=2)
            )

            fig.update_layout(
                autosize=False,
                width=500,
                height=500
            )

            fig.show()
    else:
        fig = px.scatter(df, x=actual_col, y=y_col, hover_data=["year", "nuts_cd", actual_col, y_col])
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label
        )

        x_min = min(min(df[actual_col]), min(df[y_col]))
        x_max = max(max(df[actual_col]), max(df[y_col]))

        fig.add_shape(
            type='line',
            x0=x_min,
            y0=x_min,
            x1=x_max,
            y1=x_max,
            line=dict(
                color='darkorange',
                dash='dash',
                width=2
            )
        )
        fig.update_layout(autosize=False, width=700, height=700)
        fig.show()

