import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import geopandas as gpd
import matplotlib.colors as mcolors
import math

# Global counter for figure naming
figure_counter = 1

def plot_actual_vs_predicted_regional_capacity(df, capacity_unit='percentage', scaled=False, split_column=None, legend=False, save_figure=False):
    '''
    Plot actual vs. predicted regional capacity.

    Parameters:
        df (DataFrame): The dataframe containing the data.
        capacity_unit (str, optional): The unit of capacity to be used for plotting. Can be 'percentage' or 'MW'. Default is 'percentage'.
        scaled (bool, optional): Indicates whether to plot scaled predicted capacity or not. Default is False.
        split_column (str, optional): The column name indicating whether a point belongs to the training or test set. Default is None.
        legend (bool, optional): If True, display the legend. Default is False.
        save_figure (bool, optional): If True, saves the figure with the specified file name. Default is False.
    Returns:
        None

    Example:
        plot_actual_vs_predicted(df, capacity_unit='MW', scaled=True, split_column='set_type')
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

    plt.figure(figsize=(10, 10))
    
    # Filter the dataframe to only include points that will be plotted
    if split_column:
        plot_df = df[df[split_column].notnull()]  # Assuming that split_column defines points to plot
    else:
        plot_df = df

    sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=split_column, palette="deep", legend=legend)

    plt.xlabel(x_label, fontsize=16, fontweight='bold')
    plt.ylabel(y_label, fontsize=16, fontweight='bold')

    # Increase the size of tick labels
    plt.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size to 14

    # Calculate min and max based only on the filtered data
    x_min = min(plot_df[x_col].min(), plot_df[y_col].min())
    x_max = max(plot_df[x_col].max(), plot_df[y_col].max())

    plt.plot([x_min, x_max], [x_min, x_max], color='black', linestyle='--', linewidth=1)

    global figure_counter
    if save_figure:
        plt.savefig(f'FIG{figure_counter}.eps', format='eps', bbox_inches="tight")
        figure_counter += 1
    
    return plt.show()



def plot_actual_vs_predicted_national_capacity(df, by_factor=None, factor=None, scaled=False, save_figure=False):
    '''
    Plot actual vs. predicted national capacity.

    Parameters:
        df (DataFrame): The dataframe containing the data.
        by_factor (bool, optional): Indicates whether to plot by a factor or not. If True, the plot will be grouped by the specified factor. Default is None.
        factor (str, optional): The factor by which the data should be grouped if by_factor is True (e.g., 'country', 'year'). Default is None.
        scaled (bool, optional): Indicates whether to plot scaled predicted capacity or not. Default is False.

    Returns:
        None

    Example:
        plot_actual_vs_predicted_national_capacity(val_data, by_factor=True, factor='country', scaled=True)
    '''
    if scaled:
        y_col = 'scaled_predicted_national_capacity_mwp'
        y_label = 'Scaled Predicted National Capacity (MW)'
    else:
        y_col = 'predicted_national_capacity_mwp'
        y_label = 'Predicted National Capacity (MW)'

    if by_factor:
        unique_factors = df[factor].unique()
        num_factors = len(unique_factors)
        fig, axes = plt.subplots(nrows=num_factors, figsize=(10, 5 * num_factors))

        if num_factors == 1:
            axes = [axes]

        for ax, (name, group) in zip(axes, df.groupby(factor)):
            sns.scatterplot(data=group, x='national_capacity_mwp', y=y_col, ax=ax)
            ax.set_title(f"{name}")
            ax.set_xlabel('Actual National Capacity (MW)')
            ax.set_ylabel(y_label)

            # Find the maximum and minimum values to set the line's endpoints
            x_min, x_max = group['national_capacity_mwp'].min(), group['national_capacity_mwp'].max()
            y_min, y_max = group[y_col].min(), group[y_col].max()
            p1 = max(x_max, y_max)
            p2 = min(x_min, y_min)

            ax.plot([p2, p1], [p2, p1], 'black', linestyle='dashed', linewidth=1)
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(data=df, x='national_capacity_mwp', y=y_col, ax=ax)
        ax.set_xlabel('Actual National Capacity (MW)')
        ax.set_ylabel(y_label)

        x_min, x_max = df['national_capacity_mwp'].min(), df['national_capacity_mwp'].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        p1 = max(x_max, y_max)
        p2 = min(x_min, y_min)

        ax.plot([p2, p1], [p2, p1], 'black', linestyle='dashed', linewidth=1)

    plt.tight_layout()
    global figure_counter
    if save_figure:
            plt.savefig(f'FIG{figure_counter}.eps', format='eps', bbox_inches="tight")
            figure_counter += 1
    
    return plt.show()


# def plot_country_residual(df, unit='MW', save_figure=False):
#     """
#     Plot the national residual per country as a boxplot.

#     Parameters:
#     - df (DataFrame): The dataframe containing the validation data.
#     - unit (str): The unit of measurement for the residual ('MW' or '%').

#     Returns:
#     None: Displays a boxplot of the national residual per country.
#     """
#     if unit == 'MW':
#         plt.figure(figsize=(14, 8))
#         sns.boxplot(data=df, x="country", y="National Residual (MW)")
#         # plt.title("National Residual per Country")
#         plt.ylabel("National Residual (MW)")
#         plt.xlabel("")
#         plt.xticks(rotation=90)  # Rotate x-axis labels if needed
#         plt.tight_layout()
        
#     else:
#         group_sums = df.groupby(['year', 'country_cd'], observed=False)['Predicted Regional cumulative capacity (% National)'].transform('sum')
#         # group_sums = df.groupby(['year', 'country_cd'], observed=False)['Scaled Predicted Regional cumulative capacity (% National)'].transform('sum')
#         # This is equal to MAPE
#         df['National Residual (%)'] = 100 - group_sums
#         plt.figure(figsize=(14, 8))
#         sns.boxplot(data=df, x="Country", y="National Residual (%)")
#         # plt.title("National Residual per Country")
#         plt.ylabel("National Residual (%)")
#         plt.xlabel("")
#         plt.xticks(rotation=90)  # Rotate x-axis labels if needed
#         plt.tight_layout()
        
#     global figure_counter
#     if save_figure:
#             plt.savefig(f'FIG{figure_counter}.eps', format='eps', bbox_inches="tight")
#             figure_counter += 1
    
#     return plt.show()



def plot_yearly_residual(df, unit='MW', save_figure=False):
    """
    Plot the national residual per year as a boxplot.

    Parameters:
    - df (DataFrame): The dataframe containing the validation data.
    - unit (str): The unit of measurement for the residual ('MW' or '%').
    - save_figure (bool): Whether to save the figure as an SVG file.

    Returns:
    None: Displays a boxplot of the national residual per year.
    """
    if unit == 'MW':
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x="year", y="national_residual_mwp")
        # plt.title("National Residual per Year")
        plt.xlabel("Year")
        plt.ylabel("National Residual (MW)")
        plt.xticks(rotation=45)  # Rotate x-axis labels if needed for better readability
        plt.tight_layout()
        
    else:
        group_sums = df.groupby(['year', 'country_cd'], observed=False)['predicted_capacity_percentage'].transform('sum')
        # group_sums = df.groupby(['year', 'country_cd'], observed=False)['Scaled Predicted Regional cumulative capacity (% National)'].transform('sum')

        df['National Residual (%)'] = 100 - group_sums
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x="year", y="National Residual (%)")
        # plt.title("National Residual per Year")
        plt.xlabel("Year")
        plt.ylabel("National Residual (%)")
        plt.xticks(rotation=45)  # Rotate x-axis labels if needed for better readability
        plt.tight_layout()
    global figure_counter
    if save_figure:
            plt.savefig(f'FIG{figure_counter}.eps', format='eps', bbox_inches="tight")
            figure_counter += 1
    
    return plt.show()

# def create_nuts3_geodf(norm_eu, year=None):
#     '''
#     Create a GeoDataFrame for NUTS 3 regions, optionally filtered by year.

#     Parameters:
#     - norm_eu (DataFrame): DataFrame with NUTS 3 level data.
#     - year (int, optional): The year to filter by. If None, returns all data.

#     Returns:
#     - GeoDataFrame with NUTS 2 regions, optionally filtered by year.
#     '''
#     # Load the shapefile for NUTS regions
#     geo_data = gpd.read_file(r"/gb_pv_capacity_model/data/nuts.shp.zip")
    
#     # Filter for NUTS 3 regions
#     geo_data = geo_data[geo_data["LEVL_CODE"] == 3]
    
#     # Merge with the provided dataset on the NUTS ID
#     geo_data = geo_data.merge(norm_eu, left_on='NUTS_ID', right_on='nuts_cd', how='left')
    
#     # Filter by the specified year if provided
#     if year is not None:
#         geo_data = geo_data[geo_data['year'] == year]
    
#     return geo_data

def create_nuts3_geodf(norm_eu, year=None):
    '''
    Create a GeoDataFrame for NUTS 3 regions, optionally filtered by a single year or a range of years.

    Parameters:
    - norm_eu (DataFrame): DataFrame with NUTS 3 level data.
    - year (int, tuple, list, optional): The year or range of years to filter by. 
      If None, returns all data. A tuple or list of years can be provided for a range.

    Returns:
    - GeoDataFrame with NUTS 3 regions, optionally filtered by year or range of years.
    '''
    # Load the shapefile for NUTS regions
    geo_data = gpd.read_file(r"/gb_pv_capacity_model/data/nuts.shp.zip")
    
    # Filter for NUTS 3 regions
    geo_data = geo_data[geo_data["LEVL_CODE"] == 3]
    
    # Merge with the provided dataset on the NUTS ID
    geo_data = geo_data.merge(norm_eu, left_on='NUTS_ID', right_on='nuts_cd', how='left')
    
    # Filter by the specified year(s) if provided
    if year is not None:
        if isinstance(year, (tuple, list)):  # If year is a range (tuple or list)
            geo_data = geo_data[geo_data['year'].between(year[0], year[1])]
        else:  # If year is a single value
            geo_data = geo_data[geo_data['year'] == year]
    
    return geo_data



def plot_geodata_variable(geo_data, variable_name, colorbar_label, vmin=None, vmax=None, save_figure=False):
    """
    Plot a variable from the geo_data dataframe and include a colorbar.
    Handle NaN values by assigning a specific color.

    Parameters:
    - geo_data: DataFrame containing the geographic data.
    - variable_name: The name of the column to plot.
    - colorbar_label: Label for the colorbar.
    - vmin: Minimum value for color normalization (optional).
    - vmax: Maximum value for color normalization (optional).
    """
    # Define overseas regions based on the NUTS_ID column
    overseas_regions = ['FRY1', 'FRY2', 'FRY3', 'FRY4', 'FRY5', 'ES70', 'PT20', 'PT30', 'NO0B']
    
    # Filter mainland regions by removing overseas regions
    mainland = geo_data[~geo_data['NUTS_ID'].isin(overseas_regions)]
    
    # Calculate vmin and vmax if not provided
    if vmin is None:
        vmin = mainland[variable_name].min()
    if vmax is None:
        vmax = mainland[variable_name].max()
    
    # Create a shared color normalization
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Create the figure and axis for mainland regions
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot mainland regions on the map
    mainland.plot(
        column=variable_name, 
        cmap='Blues', 
        linewidth=0.8, 
        ax=ax1, 
        edgecolor='0.8', 
        norm=norm,
        missing_kwds={
            "color": "lightgrey",      # Color for NaN values
            "edgecolor": "darkgrey",        # Edge color for NaN regions (optional)
            "label": "Missing values"  # Label in the legend for NaN values
        }
    )
    ax1.set_axis_off()
    
    # Add a colorbar below the map
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Blues'), ax=ax1, orientation="vertical", fraction=0.05, pad=0.05, shrink=0.8)
    # cbar.set_label(colorbar_label)
    # cbar.set_label(colorbar_label, fontsize=20, fontweight='bold')

    # Set the label on top of the colorbar
    cbar.ax.set_title(colorbar_label, fontsize=20, fontweight='bold', pad=20)  # Adjust 'pad' for spacing


    # Increase the tick label size
    cbar.ax.tick_params(labelsize=20)  # Adjust 'labelsize' as needed
    
    # Show the plot
    plt.tight_layout()

    global figure_counter
    if save_figure:
            plt.savefig(f'FIG{figure_counter}.eps', format='eps', bbox_inches="tight")
            figure_counter += 1
        
    return plt.show()

def calculate_national_capacity_difference(norm_eu, year=None):
    """
    Calculate the difference between National cumulative capacity and 
    the sum of Regional cumulative capacity per country and year.

    Parameters:
    - norm_eu (DataFrame): DataFrame containing 'National cumulative capacity (MW)', 
      'country_cd', 'year', 'Regional cumulative capacity (MW)', and 'nuts_cd'.

    Returns:
    - DataFrame: A DataFrame with the capacity difference per country and year.
    """
    # Step 1: Group by country and year, then sum the regional capacities
    regional_sum = norm_eu.groupby(['country_cd', 'year'])['capacity_mwp'].sum().reset_index()

    # Step 2: Merge the sum of regional capacities with the national capacities
    merged_df = norm_eu[['country_cd', 'year', 'national_capacity_mwp']].drop_duplicates()
    merged_df = pd.merge(merged_df, regional_sum, on=['country_cd', 'year'], how='left')

    # Step 3: Calculate the difference between national and regional cumulative capacities
    merged_df['Capacity Difference (MW)'] = merged_df['national_capacity_mwp'] - merged_df['capacity_mwp']

    # Filter by the specified year if provided
    if year is not None:
        merged_df = merged_df[merged_df['year'] == year]
        
    return merged_df


def plot_geodata_residual(geo_data, variable_name, colorbar_label, vmin=None, vmax=None, vcenter=0, show_labels=False, label_regions=None, show_annotations=False, annotation_regions=None, annotation_x=0, annotation_y=0, save_figure=False):
    """
    Plot a variable from the geo_data dataframe and include a colorbar.

    Parameters:
    - geo_data: DataFrame containing the geographic data.
    - variable_name: The name of the column to plot.
    - colorbar_label: Label for the colorbar.
    - vmin: Minimum value for color normalization (optional).
    - vmax: Maximum value for color normalization (optional).
    - vcenter: Center value for the color normalization (default is 0).
    - show_labels: Boolean, whether to show labels for regions.
    - label_regions: List of region codes (NUTS_IDs) to show labels for (optional).
    - save_figure: Boolean, whether to save the figure as an EPS file.
    """
    # Define overseas regions based on the NUTS_ID column
    overseas_regions = ['FRY1', 'FRY2', 'FRY3', 'FRY4', 'FRY5', 'ES70', 'PT20', 'PT30', 'NO0B']
    
    # Filter mainland regions by removing overseas regions
    mainland = geo_data[~geo_data['NUTS_ID'].isin(overseas_regions)]
    
    # Calculate vmin and vmax if not provided
    if vmin is None or vmax is None:
        min_val = mainland[variable_name].min()
        max_val = mainland[variable_name].max()
        # Determine the absolute max value to ensure zero is centered
        abs_max = max(abs(min_val), abs(max_val))
        vmin = -abs_max
        vmax = abs_max

    # Create a TwoSlopeNorm instance with the given vmin, vcenter, and vmax
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Create the figure and axis for mainland regions
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot mainland regions on the map
    mainland.plot(
        column=variable_name, 
        cmap='RdBu', 
        linewidth=0.8, 
        ax=ax1, 
        edgecolor='0.8', 
        norm=divnorm,
    )
    ax1.set_axis_off()

    # Filter out rows where the variable is NaN, since they will not be plotted
    valid_data = mainland[~mainland[variable_name].isna()]
    
    # Optionally add labels to specific regions
    if show_labels and label_regions:
        for idx, row in valid_data.iterrows():
            if row['NUTS_ID'] in label_regions:
                # Get the centroid of the geometry for placing the label
                centroid = row['geometry'].centroid
                # Place the NUTS_ID as text at the centroid position
                ax1.text(centroid.x, centroid.y, row['NUTS_ID'], fontsize=16, ha='center', color='black')
                # Add a small dot below the text (adjust y-coordinate for placement)
                ax1.plot(centroid.x, centroid.y - 10000, 'o', color='black', markersize=5)  # Adjust y-offset and size
            #     ax1.annotate(row['NUTS_ID'], xy=(centroid.x, centroid.y), xytext=(4893581, centroid.y),
            # arrowprops=dict(arrowstyle="-", facecolor='black'), fontsize=12
            # )
                print(centroid.x, centroid.y)

    # Optionally add labels to specific regions
    if show_annotations and annotation_regions:
        for idx, row in valid_data.iterrows():
            if row['NUTS_ID'] in annotation_regions:
                # Get the centroid of the geometry for placing the label
                centroid = row['geometry'].representative_point()
                # Place the NUTS_ID as text at the centroid position
                # ax1.text(centroid.x, centroid.y, row['NUTS_ID'], fontsize=10, ha='center', color='black')
                ax1.annotate(row['NUTS_ID'], xy=(centroid.x, centroid.y), xytext=(centroid.x+annotation_x, centroid.y+annotation_y),
            arrowprops=dict(arrowstyle="-", facecolor='black'), fontsize=16
            )
                print(centroid.x, centroid.y)
    # Add a colorbar below the map
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=divnorm, cmap='RdBu'), ax=ax1, orientation="vertical", fraction=0.05, pad=0.05, shrink=0.8)

    # cbar.set_label(colorbar_label, fontsize=20, fontweight='bold')
    # Set the label on top of the colorbar
    cbar.ax.set_title(colorbar_label, fontsize=20, fontweight='bold', pad=20)
    # Increase the tick label size
    cbar.ax.tick_params(labelsize=20)  
    # Show the plot
    plt.tight_layout()

    global figure_counter
    if save_figure:
        plt.savefig(f'FIG{figure_counter}.eps', format='eps', bbox_inches="tight")
        figure_counter += 1
        
    plt.show()
    
    return 


def calculate_regional_residuals(df):
    # Define the required columns and their corresponding residual columns
    column_pairs = {
        "capacity_mwp": "predicted_capacity_mwp",
        "regional_capacity_percentage": "predicted_capacity_percentage"
    }
    
    # Initialize a dictionary to hold the residual columns
    residuals = {}

    # Check each pair of columns
    for actual_col, predicted_col in column_pairs.items():
        if actual_col in df.columns and predicted_col in df.columns:
            residual_col = f'Residual {actual_col}'
            residuals[residual_col] = df[actual_col] - df[predicted_col]

    # Add residual columns to the DataFrame
    if residuals:
        for res_col, res_values in residuals.items():
            df[res_col] = res_values

    return df


def plot_unallocated_capacity(result_df, save_figure=False):
    """
    Plots the unallocated capacity vs year.
    
    Parameters:
        result_df (pd.DataFrame): A DataFrame containing columns 'year' and 'Capacity Difference (MW)'.
        save_figure (bool): If True, saves the figure as an EPS file with an incremental name.
    """
    global figure_counter  # Access the global figure counter
    
    plt.figure(figsize=(10, 6))
    plt.plot(result_df['year'], result_df['Capacity Difference (MW)'], marker='o', label='Unallocated Capacity')
    plt.xlabel('Year', fontsize=16, fontweight='bold')
    plt.ylabel('Unallocated Capacity (MW)', fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    if save_figure:
        plt.savefig(f'FIG{figure_counter}.eps', format='eps', bbox_inches="tight")
        figure_counter += 1  # Increment the figure counter

    # Show the plot
    plt.show()


def calculate_and_plot_capacity_difference(df, start_year, end_year, save_figure=False):
    """
    Calculate and visualize the difference between actual and predicted PV capacity for regions.

    This function filters the input DataFrame for a specified year range, computes the capacity difference 
    between actual and predicted PV capacity for each region, and visualizes the results as a horizontal bar chart 
    and a geographic plot. The geographic plot uses the `plot_geodata_residual` function for visualization.

    Parameters:
    - df (DataFrame): The input DataFrame containing data for regions. Must include columns:
                      'nuts_cd', 'year', 'capacity_mwp', and 'predicted_capacity_mwp'.
    - start_year (int): The starting year of the range for filtering the data.
    - end_year (int): The ending year of the range for filtering the data.
    - save_figure (bool, optional): Whether to save the generated plots as EPS files. Default is False.

    Returns:
    - sum_by_region (DataFrame): A DataFrame containing the summed capacity differences for each region.
    
    Side Effects:
    - Generates a horizontal bar chart showing the capacity differences for each region.
    - Generates a geographic plot of the capacity differences.

    Notes:
    - The function uses the `plot_geodata_residual` function to create the geographic plot.
    - Geographic data for regions (with geometries) must be present in the input DataFrame `df`.
    """
    global figure_counter

    # Debug: Check the input DataFrame
    print(f"Initial DataFrame shape: {df.shape}")
    print(f"Unique years in the input data: {df['year'].unique()}")

    # Filter data for the selected year range
    filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

    # Debug: Check the filtered DataFrame
    print(f"Filtered DataFrame shape: {filtered_df.shape}")
    if filtered_df.empty:
        print("Filtered DataFrame is empty. Check the year range and data.")
        return

    # Calculate capacity difference for each row
    filtered_df['Capacity_Difference'] = (
        filtered_df['capacity_mwp'] - filtered_df['predicted_capacity_mwp']
    )

    # Debug: Check if Capacity_Difference was added
    print("Sample Capacity_Difference values:")
    print(filtered_df[['nuts_cd', 'year', 'Capacity_Difference']].head())

    # Group by region and calculate the sum of capacity differences
    sum_by_region = filtered_df.groupby('nuts_cd')['Capacity_Difference'].sum()

    # Debug: Check the grouped data
    print(f"Sum by region:\n{sum_by_region}")
    if sum_by_region.empty:
        print("sum_by_region is empty. No data to plot.")
        return
    # Create a mapping of region codes (nuts_cd) to region names (NUTS_NAME)
    region_mapping = filtered_df.set_index('nuts_cd')['NUTS_NAME'].to_dict()
    
    # Print region code, region name, and capacity difference
    if isinstance(sum_by_region, pd.Series):
        for region, diff in sum_by_region.items():
            region_name = region_mapping.get(region, "Unknown")  # Get region name or "Unknown" if not found
            print(f"Region Code: {region}, Region Name: {region_name}, Capacity Difference: {diff:.1f} MW")
    elif isinstance(sum_by_region, pd.DataFrame):
        for region, diff in zip(sum_by_region.index, sum_by_region['Capacity_Difference']):
            region_name = region_mapping.get(region, "Unknown")  # Get region name or "Unknown" if not found
            print(f"Region Code: {region}, Region Name: {region_name}, Capacity Difference: {diff:.1f} MW")

    # Plot the data
    plt.figure(figsize=(8, 20))
    sum_by_region.sort_values().plot(kind='barh', legend=None)
    plt.xlabel('SPVDI (MW)')
    plt.ylabel('')
    plt.tight_layout()

    # Prepare geographic data for plotting
    sum_by_region = sum_by_region.reset_index()  # Ensure it's a DataFrame
    sum_by_region.rename(columns={'index': 'nuts_cd'}, inplace=True)  # Match the key column name
    geo_data_with_capacity = df.merge(sum_by_region, on='nuts_cd', how='left')

    plot_geodata_residual(
        geo_data=geo_data_with_capacity, 
        variable_name='Capacity_Difference',  # The column you want to plot
        colorbar_label='SPVDI (MW)',  # Adjust label as needed
        save_figure=False  # Set to True to save the plot
    )

    # Save the figure if save_figure is True
    if save_figure:
        plt.savefig(f'FIG{figure_counter}.eps', format='eps', bbox_inches="tight")
        figure_counter += 1

    plt.show()
    return geo_data_with_capacity

# def calculate_capacity_difference(df, start_year, end_year, country):
#     """
#     Calculate the capacity difference for a specified country and year range.
#     """
#     # # Check the input data
#     # print("DataFrame head before any processing:")
#     # print(df.head())

#     # Ensure the 'year' column is handled correctly
#     print("Original 'year' column type:", df['year'].dtype)

#     # If 'year' is numeric, ensure it's correctly recognized
#     if pd.api.types.is_numeric_dtype(df['year']):
#         print("Unique years before any processing:", df['year'].unique())
#     else:
#         print("Year column contains non-numeric data. Attempting to convert.")
#         df['year'] = pd.to_numeric(df['year'], errors='coerce')
#         print("Unique years after conversion:", df['year'].unique())

#     # Ensure 'nuts_cd' is a string
#     df['nuts_cd'] = df['nuts_cd'].astype(str)

#     # Filter the DataFrame based on the chosen year range and country
#     filtered_df = df[
#         (df['year'] >= start_year) & 
#         (df['year'] <= end_year) & 
#         (df['nuts_cd'] == country)
#     ]

#     # Debug: Check the filtered DataFrame
#     if filtered_df.empty:
#         print(f"No data found for country '{country}' between {start_year} and {end_year}.")
#         return None

#     # Calculate the sum of capacities
#     sum_capacity = filtered_df[['predicted_capacity_mwp', 'capacity_mwp']].sum()

#     # Debug: Check sums
#     print(f"Sum of predicted_capacity_mwp: {sum_capacity['predicted_capacity_mwp']:.2f} MW")
#     print(f"Sum of actual capacity_mwp: {sum_capacity['capacity_mwp']:.2f} MW")

#     # Calculate the capacity difference
#     capacity_difference = sum_capacity['capacity_mwp'] - sum_capacity['predicted_capacity_mwp']

#     # Debug: Print results
#     print(f"Capacity difference for {country}: {capacity_difference:.2f} MW")
#     # print(f"Capacity difference for {country}: {capacity_difference / 1000:.2f} GW")

#     return 


def print_spvdi_mw(df, residual_column, n_regions=10):
    """
    Prints the top 'n_regions' and bottom 'n_regions' regions based on the residual values.
    It prints the region name, code, and residual value formatted to 2 decimal places.

    Parameters:
    - df (DataFrame): The input DataFrame with regional data.
    - residual_column (str): The name of the column containing residual values.
    - n_regions (int): The number of top and bottom regions to display. Default is 10.
    """
    # Drop rows where the residual column is NaN
    df_clean = df.dropna(subset=[residual_column])

    # Separate positive and negative residuals
    positive_residuals = df_clean[df_clean[residual_column] > 0]
    negative_residuals = df_clean[df_clean[residual_column] < 0]

    # Sort positive residuals in descending order
    positive_residuals_sorted = positive_residuals.sort_values(by=residual_column, ascending=False)

    # Sort negative residuals in ascending order
    negative_residuals_sorted = negative_residuals.sort_values(by=residual_column, ascending=True)

    # Select the top 'n_regions' max (from positive) and top 'n_regions' min (from negative)
    top_n_max = positive_residuals_sorted.head(n_regions)
    top_n_min = negative_residuals_sorted.head(n_regions)

    # Combine the two into a new DataFrame
    top_n_max_min = pd.concat([top_n_max, top_n_min])

    # Reset the index of the new DataFrame
    top_n_max_min.reset_index(drop=True, inplace=True)

    # Print the result
    print(f"Top {n_regions} Regions with Positive Residuals and Bottom {n_regions} Regions with Negative Residuals:")
    
    for index, row in top_n_max_min.iterrows():
        # Print the values with the residual formatted to 1 decimal place
        print(f"Region Name: {row['NUTS_NAME']}, Region Code: {row['nuts_cd']}, Residual: {row[residual_column]:.0f} MW")


def calculate_percentage_error(df, n_regions=10):
    """
    Prints the top 'n_regions' and bottom 'n_regions' regions based on the percentage error values.
    It prints the region name, code, and percentage error value formatted to 2 decimal places.

    Parameters:
    - df (DataFrame): The input DataFrame with regional data.
    - n_regions (int): The number of top and bottom regions to display. Default is 10.
    """
    df["regional_percentage_error"]=(df["residual_percentage"]/df["regional_capacity_percentage"])*100
    # df["regional_percentage_error_mw"]=(df["residual_mwp"]/df["capacity_mwp"])*100
    residual_column = "regional_percentage_error"

    # df_clean = df[np.isfinite(df[residual_column])]
    # Drop rows where the residual column is NaN
    df_clean = df.dropna(subset=[residual_column])
    
    # Separate positive and negative residuals
    positive_residuals = df_clean[df_clean[residual_column] > 0]
    negative_residuals = df_clean[df_clean[residual_column] < 0]

    # Sort positive residuals in descending order
    positive_residuals_sorted = positive_residuals.sort_values(by=residual_column, ascending=False)

    # Sort negative residuals in ascending order
    negative_residuals_sorted = negative_residuals.sort_values(by=residual_column, ascending=True)

    # Select the top 'n_regions' max (from positive) and top 'n_regions' min (from negative)
    top_n_max = positive_residuals_sorted.head(n_regions)
    top_n_min = negative_residuals_sorted.head(n_regions)

    # Combine the two into a new DataFrame
    top_n_max_min = pd.concat([top_n_max, top_n_min])

    # Reset the index of the new DataFrame
    top_n_max_min.reset_index(drop=True, inplace=True)

    # Print the result
    print(f"Top {n_regions} Regions with Positive percentage error and Bottom {n_regions} Regions with Negative percentage error:")
    
    for index, row in top_n_max_min.iterrows():
        # Print the values with the residual formatted to 1 decimal place
        print(f"Region Code: {row['nuts_cd']}, Region Name: {row['NUTS_NAME']},  Pdercentage Error: {row[residual_column]:.0f} %")


def calculate_and_print_mape_per_region(data, actual_col, predicted_col):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) for each region and print the results.
    
    The function assumes the columns 'nuts_cd' and 'nuts_name' exist in the data.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame containing region, actual, and predicted values.
        actual_col (str): The column name for actual values (e.g., 'regional_capacity_percentage').
        predicted_col (str): The column name for predicted values (e.g., 'predicted_capacity_percentage').

    Returns:
        pd.DataFrame: A DataFrame with region name, region code, and their corresponding MAPE values.
    """
    # Group by region and calculate MAPE for each group
    mape_results = (
        data
        .groupby(['nuts_cd', 'nuts_name'])
        .apply(lambda group: (
            abs((group[actual_col] - group[predicted_col]) / group[actual_col])
            .mean() * 100
        ))
        .reset_index(name='MAPE')
    )
    
    # Print results for each region
    for _, row in mape_results.iterrows():
        print(f"Region Code: {row['nuts_cd']}, Region: {row['nuts_name']}, MAPE: {row['MAPE']:.0f}%")
    
    return mape_results


def plot_timeseries_by_region(df, selected_regions=None, mode='MW', columns=2, legend=True, save_figure=False):
    """
    Create time series plots of PV capacity (MW or percentages) and predicted values by region.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the dataset.
    - selected_regions (list, optional): A list of region names or codes to plot. If None, all unique regions in
      the dataset will be plotted.
    - mode (str, optional): Mode for plotting. 'MW' for absolute capacities and 'percentage' for relative percentages.
      Default is 'MW'.
    - columns (int, optional): Number of columns in the subplot grid.
    - legend (bool, optional): If True, display the legend. Default is True.
    - save_figure (bool, optional): If True, saves the figure with a unique file name. Default is False.

    Returns:
    None
    """
    # Ensure the 'year' column is in datetime format
    df['year'] = pd.to_datetime(df['year'], format='%Y')

    # Use all countries if none are specified
    if selected_regions is None:
        selected_regions = df['nuts_name'].unique()

    # Determine the number of rows for subplots
    rows = math.ceil(len(selected_regions) / columns)

    # Create subplots
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows), sharex=False, sharey=False)
    axes = axes.flatten()  # Flatten axes for easier indexing

    # Select y-axis labels based on mode
    y_col_actual = 'capacity_mwp' if mode == 'MW' else 'regional_capacity_percentage'
    y_col_predicted = 'predicted_capacity_mwp' if mode == 'MW' else 'predicted_capacity_percentage'
    y_label = 'Capacity (MW)' if mode == 'MW' else 'Capacity (%)'

    for i, country in enumerate(selected_regions):
        # Filter data for the current country
        country_data = df[(df['nuts_name'] == country) | (df['nuts_cd'] == country)]

        # Select the current axis
        ax = axes[i]

        # Plot actual and predicted capacity time series
        sns.lineplot(x='year', y=y_col_actual, data=country_data, label='Actual', ax=ax)
        sns.scatterplot(x='year', y=y_col_actual, data=country_data, ax=ax)
        sns.lineplot(x='year', y=y_col_predicted, data=country_data, label='Predicted', ax=ax)
        sns.scatterplot(x='year', y=y_col_predicted, data=country_data, ax=ax)

        # Set axis labels and titles
        ax.set_xlabel('Year')
        ax.set_ylabel(y_label)
        ax.set_title(f'{country}')

        # Configure legend
        if legend:
            ax.legend(loc='upper left')
        else:
            ax.get_legend().remove()

    # Hide unused subplots if the number of countries doesn't fill all axes
    for j in range(len(selected_regions), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout for better appearance
    plt.tight_layout()

    # Save figure if required
    global figure_counter
    if save_figure:
        plt.savefig(f'FIG{figure_counter}.eps', format='eps', bbox_inches="tight")
        figure_counter += 1

    # Show the plot
    plt.show()








