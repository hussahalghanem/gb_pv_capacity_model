

import os
import logging
# import glob
from urllib import request
import json
import pandas as pd
import geopandas as gpd
import openpyxl
# import pycountry
import datetime
import requests
import gzip
import numpy as np
from io import BytesIO
from geocode import Geocoder
import re

def download_engwales_lsoa_boundaries(outfile):
    ons_url = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Lower_Layer_Super_Output_Areas_Dec_2011_Boundaries_Full_Extent_BFE_EW_V3_2022/FeatureServer/0/query"
    
    # Parameters for the initial request
    params = {
        "outFields": "*",
        "where": "1=1",
        "f": "geojson",
        "resultOffset": 0,  # Start from the first record
        "resultRecordCount": 2000  # Number of records per request (assuming 2000 as the limit)
    }

    # Initialize an empty list to store the chunks of data
    all_features = []

    # Loop until all records are fetched
    while True:
        # Send a GET request
        response = requests.get(ons_url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Extract GeoJSON data from the response
            geojson_data = response.json()
            
            # Append features to the list
            all_features.extend(geojson_data['features'])
            
            # Check if there are more records
            if len(geojson_data['features']) < params['resultRecordCount']:
                # No more records to fetch, break out of the loop
                break
            
            # Update the offset for the next request
            params['resultOffset'] += params['resultRecordCount']
        else:
            # Handle errors
            print("Error occurred:", response.text)
            break

    # Write all features to the output file
    with open(outfile, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": all_features}, f)

def download_scotland_lsoa_boundaries(outfile):
    nrs_url = "https://www.nrscotland.gov.uk/files/geography/output-area-2011-eor.zip"
    opener = request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    request.install_opener(opener)
    request.urlretrieve(nrs_url, outfile)

def load_lsoa_boundaries(engwales_file, scotland_file, download=True):
    """Load the LSOA boundaries for GB as GeoDataFrames."""
    logging.debug(f"Loading LSOA boundaries from '{engwales_file}' and '{scotland_file}'")
    if not os.path.isfile(engwales_file):
        if download:
            logging.debug("LSOA boundary file not found for England & Wales, downloading from ONS")
            download_engwales_lsoa_boundaries(engwales_file)
        else:
            raise Exception("LSOA boundaries file for England and Wales not found at "
                            f"'{engwales_file}'")
    if not os.path.isfile(scotland_file):
        if download:
            logging.debug("LSOA boundary file not found for Scotland, downloading from ONS")
            download_scotland_lsoa_boundaries(scotland_file)
        else:
            raise Exception(f"LSOA boundaries file for Scotland not found at '{scotland_file}'")
    engwales = gpd.read_file(engwales_file)[["LSOA11CD", "geometry"]].to_crs("EPSG:27700")
    scotland = gpd.read_file(scotland_file)[["code", "geometry"]]\
                  .rename(columns={"code": "LSOA11CD"}).to_crs("EPSG:27700")
    lsoa = pd.concat([engwales, scotland], ignore_index=True)
    logging.debug("Finished loading LSOA boundaries")
    return lsoa

def download_nuts_boundaries(outfile):
    nuts_url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/shp/NUTS_RG_01M_2021_4326.shp.zip"
    request.urlretrieve(nuts_url, outfile)

def load_nuts_boundaries(nuts_file, download=True):
    logging.debug(f"Loading NUTS boundaries from GISCO API")
    if not os.path.isfile(nuts_file):
        if download:
            logging.debug("NUTS boundary file not found, downloading from GISCO API")
            download_nuts_boundaries(nuts_file)
        else:
            raise Exception(f"NUTS boundaries file not found at '{nuts_file}'")
    nuts_regions = gpd.read_file(nuts_file)
    logging.debug("Finished loading NUTS boundaries")
    return nuts_regions



def save_data(data, subdirectory='', base_path='/gb_pv_capacity_model/data', date=None, data_type='processed'):
    """
    Save data to the specified subdirectory with the given or current date as a subdirectory.

    Args:
        data (Union[pd.DataFrame, dict]): Data to be saved. It can be a single DataFrame or a 
            dictionary of DataFrames.
        subdirectory (str, optional): Additional subdirectory within 'raw' or 'processed'. Defaults 
            to an empty string.
        base_path (str, optional): Base path for the directory. Defaults to 
            '/gb_pv_capacity_model/data'.
        date (Union[str, datetime.date], optional): The date to use for the subdirectory. If None, 
            the current date will be used. Defaults to None.
        data_type (str, optional): Type of data, either 'raw' or 'processed'. Defaults to 'processed'.

    Returns:
        None
    """
    # Use the specified date or get the current date
    if date is None:
        today = datetime.date.today()
    else:
        today = datetime.datetime.strptime(str(date), '%Y%m%d').date()

    # Format the date as a string with the desired format (YYYYMMDD)
    date_str = today.strftime('%Y%m%d')

    # Determine the subdirectory based on the data type
    data_type_directory = f'{date_str}/{data_type}'

    # Create the directory if it doesn't already exist
    directory_path = os.path.join(base_path, data_type_directory, subdirectory)
    os.makedirs(directory_path, exist_ok=True)
        
    # Save the data to the directory
    if isinstance(data, pd.DataFrame):
        file_path = os.path.join(directory_path, f'{data.name}.csv') if hasattr(data, 'name') else os.path.join(directory_path, 'data.csv')
        data.to_csv(file_path, index=False)
    elif isinstance(data, dict):
        for key, df in data.items():
            file_path = os.path.join(directory_path, f'{key}.csv')
            df.to_csv(file_path, index=False)
    else:
        raise ValueError("Invalid input. 'data' must be either a DataFrame or a dictionary of DataFrames.")

def create_region_year_df(start_year, end_year, region_df):
    '''
    Create a DataFrame that contains regions and years for later data population.

    This function creates a DataFrame that combines a range of years with regions from the provided DataFrame
    to create a comprehensive dataset with all possible combinations of regions and years. This DataFrame can 
    be used as a foundation for populating data for each region and year.

    Args:
        start_year (str): The starting year of the range.
        end_year (str): The ending year of the range (inclusive).
        region_df (pandas.DataFrame): DataFrame containing region information with at least one column representing the regions.

    Returns:
        pandas.DataFrame: A DataFrame with all possible combinations of regions and years.

    Example:
        start_year = '2010'
        end_year = '2023'
        region_df = pd.DataFrame({
            'NUTS_CD': ['NUTS1-A', 'NUTS1-B', 'NUTS2-C'],
            'Region_Name': ['Region A', 'Region B', 'Region C']
        })
        result_df = create_region_year_df(start_year, end_year, region_df)

        The resulting DataFrame will contain all combinations of 'NUTS_CD' and years from 2010 to 2023.
    '''
    # Create a range of years from 2010 to 2023
    years = pd.date_range(start=start_year, end=end_year, freq='YS').year.tolist()

    # Create a DataFrame with the years
    year_df = pd.DataFrame({'year': years})

    # Merge the existing DataFrame with the year_df based on the 'NUTS_CD' column
    merged_df = region_df.merge(year_df, how='cross')
    merged_df['date']= pd.to_datetime(merged_df['year'], format='%Y')
    merged_df['date'] = merged_df['date'] + pd.offsets.YearEnd(0)

    return merged_df

def aggregate_capacity_by_year(PV_capacity_data: pd.DataFrame, level: str, capacity_column: str, calculate_cumulative: bool = False) -> pd.DataFrame:
    '''
    Aggregate PV capacity data to any level.

    Args:
        PV_capacity_data (pandas.DataFrame): DataFrame containing PV capacity additions data and column with region id.
        level (str): Column name indicating the llsoa or NUTS level. Could be 'NUTS1_CD', 'NUTS2_CD', or 'NUTS3_CD'.
        capacity_column (str): Column name for the capacity data (e.g., 'dc_capacity_mwp').
        calculate_cumulative (bool): If True, calculate cumulative capacity; if False, calculate capacity additions per year.

    Returns:
        pandas.DataFrame: Aggregated DataFrame with capacity additions per year or cumulative capacity.
    '''
    # Calculate capacity additions or cumulative capacity for each nuts region per year
    df = PV_capacity_data.groupby(['year', level]).agg({capacity_column: 'sum'}).reset_index()
    df = df.rename(columns={capacity_column: 'capacity_mwp'})

    if calculate_cumulative:
        df['capacity_mwp'] = df.groupby([level])['capacity_mwp'].cumsum()

    df['date'] = pd.to_datetime(df['year'], format='%Y')
    df['date'] = df['date'] + pd.offsets.YearEnd(0)

    # Drop unwanted columns
    df = df.drop(['year'], axis=1)
    
    return df

def merge_and_fill_capacity(capacity_data_df, region_year_df, capacity_type, region):
    '''
    Merge capacity data with region-year DataFrame and fill missing capacity values based on capacity type.

    This function merges the capacity data DataFrame with the region-year DataFrame based on the specified 'region' and 'date' columns.
    It then fills missing capacity values based on the provided capacity type: 'added' or 'cumulative'.

    Args:
        capacity_data_df (pandas.DataFrame): DataFrame containing capacity data with 'region', 'date', and 'capacity_MW' columns.
        region_year_df (pandas.DataFrame): DataFrame created using the create_region_year_df function with 'region' and 'date' columns.
        capacity_type (str): Capacity type, either 'added' or 'cumulative'.
        region (str): The column name representing the regions in both DataFrames.

    Returns:
        pandas.DataFrame: Merged DataFrame with filled capacity values.

    Example:
        capacity_data_df:
            region   |   date    | capacity_MW
            -----------------------------------
            Region A | 2010-01-01| 10
            Region A | 2013-01-01| 20
            Region B | 2010-01-01| 15

        region_year_df:
            region   |   date
            -------------------
            Region A | 2010-01-01
            Region A | 2013-01-01
            Region A | 2016-01-01
            Region B | 2010-01-01
            Region B | 2013-01-01
            Region B | 2016-01-01

        merged_and_filled_df = merge_and_fill_capacity(capacity_data_df, region_year_df, capacity_type='cumulative', region='region')

        The resulting DataFrame will have all combinations of regions and dates from 'region_year_df' merged with the capacity data
        from 'capacity_data_df' based on 'region' and 'date'. The missing capacity values will be filled either by forward filling
        for 'cumulative' capacity or with zeros for 'added' capacity.
    '''
    
    # Merge capacity data with region-year DataFrame based on 'region' and 'year' columns
    merged_df = region_year_df.merge(capacity_data_df, on=[region, 'date'], how='left')

     # Sort the merged DataFrame by 'region' and 'date' to ensure correct order for forward filling
    merged_df.sort_values(by=[region, 'date'], inplace=True)
    
    # Fill missing capacity values based on capacity type
    if capacity_type == 'added':
        merged_df['capacity_mwp'].fillna(0, inplace=True)
    elif capacity_type == 'cumulative':
        # Forward fill the missing values within each 'region'
        merged_df['capacity_mwp'] = merged_df.groupby(region)['capacity_mwp'].fillna(method='ffill')
        # Fill any remaining NaN values with zero
        merged_df['capacity_mwp'].fillna(0, inplace=True)

        
    return merged_df


def process_uk_pv_capacity(url):
    """
    Process UK PV capacity data from Sheffield solar URL.

    Parameters:
    url (str): The URL to fetch the data from.

    Returns:
    pandas.DataFrame: DataFrame containing processed UK PV capacity data.
    """
    # Fetch data from the URL
    response = requests.get(url, headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip, deflate',
})
    
    # Check if the request was successful
    if response.status_code == 200:
        # Read the content of the response and decompress it
        with gzip.open(BytesIO(response.content), 'rt') as f:
            # Load the data into a DataFrame
            df = pd.read_csv(f)
            # Print first few rows for verification
            print(df.head()) 
    else:
        print("Failed to fetch data from the URL")

    # Convert 'install_month' column to datetime and extract year
    df['install_month'] = pd.to_datetime(df['install_month'])
    df['year']=df['install_month'].dt.year

    # Load LSOA boundaries data
    # might want to change where the data is saved
    lsoa = load_lsoa_boundaries("/gb_pv_capacity_model/data/engwales", "/gb_pv_capacity_model/data/scotland.zip", download=True)
    print(lsoa.crs)
    # Change CRS to EPSG:4326
    lsoa = lsoa.to_crs(epsg=4326)
    print(lsoa.crs)
    # Calculate centroids and extract latitude and longitude
    lsoa['centroid'] = lsoa['geometry'].centroid
    lsoa['latitude'] = lsoa['centroid'].y
    lsoa['longitude'] = lsoa['centroid'].x
    lsoa.drop(columns=['centroid'], inplace=True)

    # Reverse geocode to get NUTS levels
    with Geocoder() as geocoder:
        lsoa['NUTS1'] = geocoder.reverse_geocode_nuts(lsoa[['latitude', 'longitude']].to_numpy(), year=2021, level=1)
        lsoa['NUTS2'] = geocoder.reverse_geocode_nuts(lsoa[['latitude', 'longitude']].to_numpy(), year=2021, level=2)
        lsoa['NUTS3'] = geocoder.reverse_geocode_nuts(lsoa[['latitude', 'longitude']].to_numpy(), year=2021, level=3)

    # Merge capacity data with LSOA df
    capacity_df = df.merge(lsoa, right_on='LSOA11CD',left_on='llsoa', how='left')
    capacity_df['NUTS0'] = 'UK'

    # Aggregate capacity by NUTS3
    nuts3_cumulative_capacity = aggregate_capacity_by_year(capacity_df, 'NUTS3', 'dc_capacity_mwp', calculate_cumulative= True)

    # Load NUTS boundaries data
    nuts = load_nuts_boundaries("/gb_pv_capacity_model/data/nuts.shp.zip", download=True)
    # Filter rows with LEVL_CODE equal to 3 and CNTR_CODE equal to 'UK'
    nuts3 = nuts[(nuts['LEVL_CODE'] == 3) & (nuts['CNTR_CODE'] == 'UK')]
    nuts3 = nuts3[['NUTS_ID']]
    # Create region-year DataFrame for NUTS2
    nuts3_region_year_df = create_region_year_df('1995', '2023', nuts3)
    nuts3_region_year_df = nuts3_region_year_df.rename(columns={'NUTS_ID':'NUTS3'})
    # Merge and fill capacity data for NUTS3
    nuts3_cumulative_capacity = merge_and_fill_capacity(nuts3_cumulative_capacity, nuts3_region_year_df, 'cumulative', 'NUTS3')
    nuts3_cumulative_capacity = nuts3_cumulative_capacity.rename(columns={'NUTS3': 'nuts_cd'})
    # Set 'capacity_mw' to NaN where 'nuts_cd' starts with 'UKN' which is northern ireland
    nuts3_cumulative_capacity.loc[nuts3_cumulative_capacity['nuts_cd'].str.startswith('UKN'), 'capacity_mwp'] = np.nan

    # Aggregate capacity by NUTS0
    nuts0_cumulative_capacity = aggregate_capacity_by_year(capacity_df, 'NUTS0', 'dc_capacity_mwp', calculate_cumulative= True)
    nuts0_cumulative_capacity = nuts0_cumulative_capacity.rename(columns={'NUTS0': 'nuts_cd'})

    # Concatenate NUTS3 and NUTS0 DataFrames
    uk_pv_capacity = pd.concat([nuts3_cumulative_capacity, nuts0_cumulative_capacity], ignore_index=True)
    uk_pv_capacity.drop(columns=['year'], inplace=True)
    
    return uk_pv_capacity

def calculate_clc_land_area_by_region(regions, land_use, geo_code_col, code_18_col, threshold=0.01):
    """
    Calculate the land area for each land cover classification within each region.

    Parameters:
        regions (geopandas.GeoDataFrame): GeoDataFrame containing regions and their geometries.
        land_use (geopandas.GeoDataFrame): GeoDataFrame containing land cover data and their geometries.
        geo_code_col (str): Column name representing the region identifier in regions GeoDataFrame.
        code_18_col (str): Column name representing the land cover classification identifier in land_use GeoDataFrame.
        threshold (float, optional): Threshold for comparison (default is 0.01).

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame with the land area for each land cover classification within each region.

    Raises:
        ValueError: If inputs are not valid GeoDataFrames or the specified columns do not exist.
        ValueError: If the sum of land area classifications exceeds 'lsoa_area' for some LSOA regions above the specified threshold.

    Example:
        import geopandas as gpd
        regions = gpd.read_file('regions.shp')
        land_use = gpd.read_file('land_use.shp')
        result_df = calculate_land_area_by_geo_code(regions, land_use, 'geo_code', 'Code_18')
    """
    # Input validation
    if not isinstance(regions, gpd.GeoDataFrame) or not isinstance(land_use, gpd.GeoDataFrame):
        raise ValueError("Inputs 'lsoa_co' and 'land_use' must be GeoDataFrames.")
    
    if geo_code_col not in regions.columns or code_18_col not in land_use.columns:
        raise ValueError("Specified columns 'geo_code_col' and 'code_18_col' not found in GeoDataFrames.")
    
    # Reproject if CRS is different
    if regions.crs != land_use.crs:
        regions = regions.to_crs(land_use.crs)
    
    # # Ensure valid geometries
    # if not regions.geometry.is_valid.all() or not land_use.geometry.is_valid.all():
    #     raise ValueError("Invalid geometries found in GeoDataFrames. Please fix geometry issues.")
    
    # Intersection and aggregation
    intersection_gdf = gpd.overlay(regions, land_use, how='intersection')
    dissolved_gdf = intersection_gdf.dissolve(by=[geo_code_col, code_18_col])

    dissolved_gdf['intersection_area'] = dissolved_gdf.geometry.area

    processed_UK_land_cover_per_lsoa = dissolved_gdf.pivot_table(index=geo_code_col,
                                                                 columns=code_18_col,
                                                                 values='intersection_area',
                                                                 aggfunc='sum')

    processed_UK_land_cover_per_lsoa['clc_area'] = processed_UK_land_cover_per_lsoa.sum(axis=1)

    regions['region_area'] = regions.geometry.area

    # Merge and return the result
    merged_table = processed_UK_land_cover_per_lsoa.merge(regions[[geo_code_col, 'region_area']], on=geo_code_col, how='left')
    

    # Check if the sum of land area classifications ('clc_area') exceeds 'lsoa_area' for some LSOA regions above the threshold
    # using a threshold of 0.01 (1%) means that the sum of land area classifications can be up to 1% more 
    # than the actual LSOA area without triggering the error. 
    excess_area_mask = merged_table['clc_area'] > (1 + threshold) * merged_table['region_area']
    if excess_area_mask.any():
        exceeding_regions = merged_table.loc[excess_area_mask, geo_code_col].tolist()
        raise ValueError(f"The sum of land area classifications ('clc_area') exceeds 'lsoa_area' "
                         f"for the following LSOA regions above the threshold ({threshold*100}%): {exceeding_regions}")

    # convert_columns_to_snake_case(merged_table)
    return merged_table

def aggregate_clc_land_area_categories(df):
    """
    Aggregates land area data from multiple columns into new summarized columns based on specific categories.
    
    Args:
        df (DataFrame): Input DataFrame containing clc land area data.
        
    Returns:
        DataFrame: DataFrame with aggregated land area data.
    """
    def safe_sum(columns):
        """Sums only the columns that exist in the DataFrame."""
        available_columns = df.columns.intersection(columns)
        return df[available_columns].sum(axis=1)

    df['11'] = safe_sum(['111', '112'])
    df['12'] = safe_sum(['121', '122', '123', '124'])
    df['13'] = safe_sum(['131', '132', '133'])
    df['14'] = safe_sum(['141', '142'])
    df['21'] = safe_sum(['211', '212', '213'])
    df['22'] = safe_sum(['221', '222', '223'])
    df['23'] = safe_sum(['231'])  
    df['24'] = safe_sum(['241', '242', '243', '244'])
    df['31'] = safe_sum(['311', '312', '313'])
    df['32'] = safe_sum(['321', '322', '323', '324'])
    df['33'] = safe_sum(['331', '332', '333', '334', '335'])
    df['41'] = safe_sum(['411', '412'])
    df['42'] = safe_sum(['421', '422', '423'])
    df['51'] = safe_sum(['511', '512'])
    df['52'] = safe_sum(['521', '522', '523'])
    
    df['1'] = safe_sum(['111', '112', '121', '122', '123', '124', '131', '132', '133', '141', '142'])
    df['2'] = safe_sum(['211', '212', '213', '221', '222', '223', '231', '241', '242', '243', '244'])
    df['3'] = safe_sum(['311', '312', '313', '321', '322', '323', '324', '331', '332', '333', '334', '335'])
    df['4'] = safe_sum(['411', '412', '421', '422', '423'])
    df['5'] = safe_sum(['511', '512', '521', '522', '523'])
    
    return df


def process_climate_data(path, value_name):
    """
    Process climate data from a CSV file.

    Parameters:
    path (str): The file path to the CSV file containing climate data.
    value_name (str): The name to assign to the value column in the processed DataFrame.

    Returns:
    pandas.DataFrame: A DataFrame containing processed climate data with columns 'date',
                      'NUTS_CD', and the specified 'value_name'.
    
    This function reads climate data from a CSV file, skips 52 rows of metadata, and
    then transforms the data into a long format. The resulting DataFrame contains
    columns 'date' (representing the date of the measurement), 'NUTS_CD' (representing
    the NUTS code), and the specified 'value_name' (representing the climate value).
    """
    df = pd.read_csv(path, skiprows=52)
    processed_df = pd.melt(df, id_vars=['Date'], var_name='nuts_cd', value_name=value_name)
    processed_df=processed_df.rename(columns={'Date':'date'})
    # Filter for UK regions (NUTS codes starting with 'UK')
    processed_df = processed_df[processed_df['nuts_cd'].str.startswith('UK')]
    return processed_df

# FIT 
def process_fit_data(excel_file, sheet_start, sheet_end):
    """
    Process FIT (Feed-in Tariff) data from an Excel file and perform data transformations.

    Args:
        excel_file (str): Path to the Excel file containing FIT data.
        sheet_start (int): Index of the first sheet to process.
        sheet_end (int): Index of the last sheet to process.

    Returns:
        pandas.DataFrame: Aggregated DataFrame with transformed FIT data.

    Example:
        aggregated_data = process_FIT_data('FIT_Rates.xlsx', 2, 5)
    """

    # Read all sheets from the Excel file into a dictionary of DataFrames
    dfs = pd.read_excel(excel_file, sheet_name=None)

    # Concatenate all DataFrames between sheet_start and sheet_end
    dfs_to_concatenate = list(dfs.values())[sheet_start:sheet_end]
    concatenated_df = pd.concat(dfs_to_concatenate)

    # Select rows with 'Technology Type' value 'Photovoltaic'
    selected_rows = concatenated_df[concatenated_df['Technology Type'] == 'Photovoltaic']
    selected_rows = selected_rows.drop(columns=['Technology Type'])

    # Replace NaN with 'Not applicable' in the 'Energy Efficiency Requirement rating' column
    selected_rows['Energy Efficiency Requirement rating'] = selected_rows['Energy Efficiency Requirement rating'].fillna('Not applicable')

    # Create a pivot table to reshape the data
    pivot_df = selected_rows.pivot_table(
        values='Tariff',
        index=['Tariff Start Date', 'Tariff End Date'],
        columns=['Solar PV Installation  Type', 'Minimum Capacity (kW)', 'Maximum Capacity (kW)', 'Energy Efficiency Requirement rating'],
        aggfunc='mean'
    ).reset_index()

    # Flatten the MultiIndex column names and add 'FIT' prefix
    pivot_df.columns = ['FIT ' + ' '.join(map(str, col)).strip() for col in pivot_df.columns.values]

    # Extract year from 'Tariff Start Date' and create a 'date' column with the last day of the year
    pivot_df['date'] = pd.to_datetime(pivot_df['FIT Tariff Start Date']).dt.to_period('Y').dt.end_time 
    pivot_df['date'] = pivot_df['date'].dt.strftime('%Y-%m-%d')  # Format date as 'YYYY-MM-DD'

    # Drop unnecessary columns before aggregation
    columns_to_drop = ['FIT Tariff Start Date', 'FIT Tariff End Date']
    pivot_df = pivot_df.drop(columns=columns_to_drop)

    # Group by 'date' and aggregate the mean values
    aggregated_df = pivot_df.groupby('date').agg('mean')

    return aggregated_df

def process_gva_data(excel_file, sheet_start, sheet_end):
    """
    Process FIT (Feed-in Tariff) data from an Excel file and perform data transformations.

    Args:
        excel_file (str): Path to the Excel file containing FIT data.
        sheet_start (int): Index of the first sheet to process.
        sheet_end (int): Index of the last sheet to process.

    Returns:
        pandas.DataFrame: Aggregated DataFrame with transformed FIT data.

    Example:
        aggregated_data = process_FIT_data('FIT_Rates.xlsx', 2, 5)
    """

    # Read all sheets from the Excel file into a dictionary of DataFrames
    dfs = pd.read_excel(excel_file, sheet_name=None)

    # Concatenate all DataFrames between sheet_start and sheet_end
    dfs_to_concatenate = list(dfs.values())[sheet_start:sheet_end]
    concatenated_df = pd.concat(dfs_to_concatenate)

    # Select rows with 'Technology Type' value 'Photovoltaic'
    selected_rows = concatenated_df[concatenated_df['Technology Type'] == 'Photovoltaic']
    selected_rows = selected_rows.drop(columns=['Technology Type'])

    # Replace NaN with 'Not applicable' in the 'Energy Efficiency Requirement rating' column
    selected_rows['Energy Efficiency Requirement rating'] = selected_rows['Energy Efficiency Requirement rating'].fillna('Not applicable')

    # Create a pivot table to reshape the data
    pivot_df = selected_rows.pivot_table(
        values='Tariff',
        index=['Tariff Start Date', 'Tariff End Date'],
        columns=['Solar PV Installation  Type', 'Minimum Capacity (kW)', 'Maximum Capacity (kW)', 'Energy Efficiency Requirement rating'],
        aggfunc='mean'
    ).reset_index()

    # Flatten the MultiIndex column names and add 'FIT' prefix
    pivot_df.columns = ['FIT ' + ' '.join(map(str, col)).strip() for col in pivot_df.columns.values]

    # Extract year from 'Tariff Start Date' and create a 'date' column with the last day of the year
    pivot_df['date'] = pd.to_datetime(pivot_df['FIT Tariff Start Date']).dt.to_period('Y').dt.end_time 
    pivot_df['date'] = pivot_df['date'].dt.strftime('%Y-%m-%d')  # Format date as 'YYYY-MM-DD'

    # Drop unnecessary columns before aggregation
    columns_to_drop = ['FIT Tariff Start Date', 'FIT Tariff End Date']
    pivot_df = pivot_df.drop(columns=columns_to_drop)

    # Group by 'date' and aggregate the mean values
    aggregated_df = pivot_df.groupby('date').agg('mean')

    return aggregated_df



def process_gva_data(file_path, sheet_index, selected_sic07_codes=None, base_path='/gb_pv_capacity_model/data', date=None, data_type='processed'):
    """
    Processes GVA data from an Excel file, pivots the data, and optionally saves selected SIC07 code groups.

    Args:
        file_path (str): The path to the Excel file.
        sheet_index (int): The index of the sheet to read (0-based).
        selected_sic07_codes (list, optional): List of SIC07 codes to filter and save. If None, processes all.
        base_path (str, optional): Base path for saving data. Defaults to '/gb_pv_capacity_model/data'.
        date (str or datetime.date, optional): Date to use for saving files. Defaults to None.
        data_type (str, optional): Data type for saving files ('raw' or 'processed'). Defaults to 'processed'.

    Returns:
        pd.DataFrame: A processed DataFrame with columns 'ITL code', 'SIC07 code',
                      'SIC07 description', 'date', and 'gva_pounds_million'.
    """
    # Read the specified sheet by index, skipping the first row
    df = pd.read_excel(file_path, sheet_name=sheet_index, skiprows=1)

    # Melt the DataFrame to transform year columns into a single 'date' column
    df_melted = df.melt(
        id_vars=['ITL code', 'Region name', 'SIC07 code', 'SIC07 description'],
        var_name='date',
        value_name='gva_pounds_million'
    )

    # Drop 'Region name' column
    df_melted = df_melted.drop(columns='Region name')

    # Ensure the 'date' column contains only year values and remove non-year rows
    df_melted = df_melted[df_melted['date'].astype(str).str.match(r'^\d{4}$')]

    # Convert 'date' to end-of-year datetime (e.g., 2022 becomes 2022-12-31)
    df_melted['date'] = pd.to_datetime(df_melted['date'].astype(str) + '-12-31', format='%Y-%m-%d')

    # Convert from ITL to NUTS
    df_melted = convert_nuts_itl(df_melted, column='ITL code')

    # Helper function to sanitize SIC07 codes
    def sanitize_sic07(sic07):
        # Remove spaces and replace other non-word characters with underscores
        sic07_no_spaces = str(sic07).replace(' ', '')  # Remove spaces
        sanitized = re.sub(r'[^\w]', '_', sic07_no_spaces)  # Replace non-word characters with underscores
        return sanitized.rstrip('_')  # Remove trailing underscores

    # Split DataFrames by 'SIC07 code' with sanitized keys
    dfs_by_sic07 = {sanitize_sic07(sic07): group for sic07, group in df_melted.groupby('SIC07 code')}

    # Rename 'gva_pounds_million' column for each DataFrame
    for sic07, df_group in dfs_by_sic07.items():
        df_group = df_group.rename(columns={'gva_pounds_million': f'gva_pounds_million_{sic07}'})
        df_group = df_group.drop(columns=['SIC07 code', 'SIC07 description'])
        dfs_by_sic07[sic07] = df_group  # Update the dictionary with the renamed DataFrame

    # Save only selected SIC07 codes if provided
    if selected_sic07_codes:
        selected_dfs = {sanitize_sic07(code): dfs_by_sic07[sanitize_sic07(code)] for code in selected_sic07_codes if sanitize_sic07(code) in dfs_by_sic07}
        if selected_dfs:
            save_data(selected_dfs, subdirectory='gva', base_path=base_path, date=date, data_type=data_type)
        else:
            print("No valid SIC07 codes found in the selected list.")

    return df_melted


def convert_nuts_itl(df, column='nuts_cd'):
    """
    Converts NUTS codes to ITL codes and vice versa by replacing 'UK' with 'TL' and 'TL' with 'UK' in the specified column.
    Ensures the returned DataFrame has the column named 'nuts_cd' or 'itl_cd'.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the NUTS or ITL codes.
    column (str): The name of the column containing the NUTS/ITL codes. Default is 'nuts_cd'.

    Returns:
    pd.DataFrame: A DataFrame with the converted codes in 'nuts_cd' or 'itl_cd'.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    def replace_codes(code):
        if isinstance(code, str):
            if code.startswith('UK'):
                return code.replace('UK', 'TL', 1)
            elif code.startswith('TL'):
                return code.replace('TL', 'UK', 1)
        return code

    # Apply the conversion
    df[column] = df[column].apply(replace_codes)

    # Detect whether column should be 'nuts_cd' or 'itl_cd'
    # Check if there are any 'UK' or 'TL' codes in the column
    if df[column].str.startswith('UK').any():
        new_column_name = 'nuts_cd'
    elif df[column].str.startswith('TL').any():
        new_column_name = 'itl_cd'
    else:
        new_column_name = column  # Default to the original column name if neither 'UK' nor 'TL' are found
    
    df = df.rename(columns={column: new_column_name})
    
    return df

def lsoa_to_nuts(df, left_on='LSOA11CD'):
    """
    Maps LSOA codes to NUTS regions by calculating centroids and performing reverse geocoding.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing LSOA codes.
        left_on (str): The name of the column in `df` that contains LSOA codes. Default is 'LSOA11CD'.

    Returns:
        pd.DataFrame: The input DataFrame merged with NUTS-level information (NUTS0, NUTS1, NUTS2, NUTS3).
    """
    # Load LSOA boundaries data
    lsoa = load_lsoa_boundaries(
        "/gb_pv_capacity_model/data/engwales",
        "/gb_pv_capacity_model/data/scotland.zip",
        download=True
    )

    print(f"Original CRS: {lsoa.crs}")
    
    # Change CRS to EPSG:4326 (WGS 84)
    lsoa = lsoa.to_crs(epsg=4326)
    print(f"Converted CRS: {lsoa.crs}")
    
    # Calculate centroids and extract latitude and longitude
    lsoa['centroid'] = lsoa['geometry'].centroid
    lsoa['latitude'] = lsoa['centroid'].y
    lsoa['longitude'] = lsoa['centroid'].x
    lsoa.drop(columns=['centroid'], inplace=True)

    # Reverse geocode to get NUTS levels using a geocoder
    with Geocoder() as geocoder:
        coordinates = lsoa[['latitude', 'longitude']].to_numpy()
        lsoa['NUTS1'] = geocoder.reverse_geocode_nuts(coordinates, year=2021, level=1)
        lsoa['NUTS2'] = geocoder.reverse_geocode_nuts(coordinates, year=2021, level=2)
        lsoa['NUTS3'] = geocoder.reverse_geocode_nuts(coordinates, year=2021, level=3)

    # Merge the input DataFrame with the LSOA DataFrame
    merged_df = df.merge(lsoa, left_on=left_on, right_on='LSOA11CD', how='left')

    # Add NUTS0 as 'UK'
    merged_df['NUTS0'] = 'UK'

    return merged_df

def merge_dataframes_by_region_and_year(dfs, region):
    """
    Merge a list of DataFrames based on 'year' and 'region' columns.

    This function takes a list of pandas DataFrames and a region column name as input. It performs
    merging operations by iteratively left merging the DataFrames based on the 'year' and 'region'
    columns. The 'year' is extracted from the 'date' column if present, and the merging is performed
    using this 'year' along with the 'region' column if available. If the 'date' column is absent,
    only the 'region' column is used for merging.
        
    Args:
        dfs (list of pandas.DataFrame): A list of pandas DataFrames to be merged.
        region (str): The name of the column to be used as the region identifier for merging.
        
    Returns:
        pandas.DataFrame: A merged DataFrame containing the data from all input DataFrames.
               
    """
    merged_df = dfs[0]  # Start with the first DataFrame

    for df in dfs[1:]:
        # Check if 'date' columns are present in both DataFrames
        if 'date' in merged_df.columns and 'date' in df.columns:
            # Extract 'year' from 'date' column
            merged_df['year'] = pd.to_datetime(merged_df['date']).dt.year
            df['year'] = pd.to_datetime(df['date']).dt.year
            # Check if 'region' is present in columns for merging
            if region in merged_df.columns and region in df.columns:
                # Merge based on both 'year' and 'region'
                merge_on = ['year', region]
            else:
                # Merge based on 'year' only if 'region' is not available
                merge_on = ['year']
             
            merged_df = pd.merge(merged_df, df, on=merge_on, suffixes=('', '_' + df.columns[1]), how='left')
        else: 
            # Merge based on 'region' if 'date' is not available
            merged_df = pd.merge(merged_df, df, on=region, how='left')

    # Drop the columns starting with "date_" 
    merged_df = merged_df.drop(columns=merged_df.filter(regex='^date_', axis=1))
    
    return merged_df

def fetch_capacity_by_nuts_level(country_dfs, nuts_level):
    """
    Extracts capacity data based on the specified NUTS (Nomenclature of Territorial Units for Statistics) level.

    Parameters:
    country_dfs (list): A list of pandas DataFrames containing capacity data for different countries.
    nuts_level (int): The NUTS level for which to extract capacity data. It should be 0, 1, 2, or 3.

    Returns:
    pandas DataFrame: A DataFrame containing capacity data for the specified NUTS level.
    
    Raises:
    ValueError: If nuts_level is not 0, 1, 2, or 3.
    """
    capacity = pd.concat(country_dfs, ignore_index=True)

    if nuts_level == 3:
        nuts_capacity = capacity[capacity['nuts_cd'].str.len() == 5]
    elif nuts_level == 2:
        nuts_capacity = capacity[capacity['nuts_cd'].str.len() == 4]
    elif nuts_level == 1:
        nuts_capacity = capacity[capacity['nuts_cd'].str.len() == 3]
    elif nuts_level == 0:
        nuts_capacity = capacity[capacity['nuts_cd'].str.len() == 2]
        # Rename the 'nuts_cd' column to 'country_cd'
        nuts_capacity.rename(columns={'nuts_cd': 'country_cd', 'capacity_mwp':'national_capacity_mwp'}, inplace=True)
    else:
        raise ValueError("Invalid value for 'nuts_level'. It should be 0, 1, 2, or 3.")

    return nuts_capacity


def disaggregate_climate_from_nuts2_to_nuts3(nuts2, GHI):
    """
    Disaggregates the climate data from NUTS2 to NUTS3 based on 'date' and the first 4 characters of 'nuts_cd'.
    
    Args:
        nuts2 (pd.DataFrame): The NUTS2 DataFrame containing 'date' and 'nuts_cd' columns.
        GHI (pd.DataFrame): The GHI DataFrame containing 'date' and 'nuts_cd' columns.
    
    Returns:
        pd.DataFrame: A DataFrame with the climate data disaggregated to NUTS3, with 'nuts_cd' renamed 
                       and unnecessary columns dropped.
    """
    # Convert 'date' columns to datetime
    nuts2['date'] = pd.to_datetime(nuts2['date'])
    GHI['date'] = pd.to_datetime(GHI['date'])

    # Extract the first 4 characters of 'nuts_cd' in both dataframes
    nuts2['nuts_cd_4'] = nuts2['nuts_cd'].str[:4]
    GHI['nuts_cd_4'] = GHI['nuts_cd'].str[:4]

    # Merge based on 'date' and the first 4 characters of 'nuts_cd'
    ghi_nuts3 = nuts2.merge(GHI, how='left', left_on=['date', 'nuts_cd_4'], right_on=['date', 'nuts_cd_4'])

    ghi_nuts3.drop(columns=['nuts_cd_4', 'nuts_cd_y'], inplace=True)

    # Rename 'nuts_cd_x' to 'nuts_cd'
    ghi_nuts3.rename(columns={'nuts_cd_x': 'nuts_cd'}, inplace=True)

    return ghi_nuts3
