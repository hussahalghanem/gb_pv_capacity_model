


def calculate_regional_capacity_percentage(df):
    '''
    Calculate the regional capacity percentage relative to national capacity.

    This function calculates the percentage of regional capacity relative to the national capacity.
    It adds a new column 'regional_capacity_percentage' to the DataFrame, which represents the percentage of regional capacity
    relative to the national capacity for each record.

    Args:
        df (pandas.DataFrame): DataFrame containing 'capacity_mwp' column representing regional capacity and another column 
        'national_capacity_mwp' representing national capacity.

    Returns:
        pandas.DataFrame: DataFrame with an additional column 'regional_capacity_percentage' showing the percentage of regional capacity
        relative to the national capacity for each record.

    Example:
        df:
            region   | capacity_mwp | national_capacity_mwp
            ------------------------------------------------
            Region A | 10            | 100
            Region B | 20            | 150
            Region C | 15            | 100

        After calling calculate_regional_capacity_percentage(df):

            region   | capacity_mwp | national_capacity_mwp | regional_capacity_percentage
            -------------------------------------------------------------------------------
            Region A | 10            | 100                   | 10.0
            Region B | 20            | 150                   | 13.333333
            Region C | 15            | 100                   | 15.0

    Notes:
        - The function calculates the percentage by dividing the regional capacity (capacity_mwp) by the national capacity
          (national_capacity_mwp) and multiplying by 100.
        - The sum of the percentage capacity per year is printed along with a check to see if the sums are close to one,
          accounting for possible floating-point errors.

    '''

    df['regional_capacity_percentage']=df['capacity_mwp']/df['national_capacity_mwp'] * 100

    # Sum the percentage capacity per year
    sum_percentage_capacity_per_year = df.groupby(['year', 'country_cd'])['regional_capacity_percentage'].sum()

    # Check if the sum is close to 1 (accounting for possible floating-point errors)
    is_close_to_one = (sum_percentage_capacity_per_year - 1).abs() < 1e-3

    # Print the results
    print("Sum of percentage capacity per year:")
    print(sum_percentage_capacity_per_year)
    print("Are the sums close to one?")
    print(is_close_to_one)
    
    return df


# land area as a percentage of national 

def calculate_regional_clc_percentage(df):
    '''
    Calculate the percentage of regional CLC area relative to national land area.

    Args:
        df (pandas.DataFrame): DataFrame with CLC category columns and 'regional_area' representing regional land area,
        and 'national_area' representing national land area.

    Returns:
        pandas.DataFrame: DataFrame with new columns for the percentages of regional CLC area relative to national land area.

    Example:
        df:
            111  | 112  | 121  | ... | regional_area | country_area
            -------------------------------------------------------
            100  | 200  | 300  | ... | 500           | 1000

        After calling calculate_regional_clc_percentage(df):

            111_percentage  | 112_percentage  | 121_percentage  | ...
            --------------------------------------------------------
            10.0            | 20.0            | 30.0            | ...
    '''
    
    # List of column names to create as percentages
    percentage_columns = ['111', '112', '121', '122', '123', '124', '131', '132', '133', '141', '142', '211', '212', '213',
                          '221', '222', '223', '231', '241', '242', '243', '244', '311', '312', '313', '321', '322', '323', 
                          '324', '331', '332', '333', '334', '335', '411', '412', '421', '422', '423', '511', '512', '521', 
                          '522', '523', '11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '41', '42', '51', 
                          '52', '1', '2', '3', '4', '5', 'region_area' ]

    # Ensure both 'country_area' and 'region_area' exist
    if 'country_area' in df.columns and 'region_area' in df.columns:
        # Calculate and add the new columns as percentages for the columns that exist
        for column in percentage_columns:
            if column in df.columns:  # Check if the column exists before performing the calculation
                df[f"{column}_percentage"] = (df[column] / df['country_area']) * 100
        
        # Drop the original columns that were used for the calculation
        df.drop(columns=[col for col in percentage_columns if col in df.columns], inplace=True)
        df.drop(columns=['country_area'], inplace=True)
    
    else:
        raise ValueError("Both 'country_area' and 'region_area' columns must be present in the DataFrame.")
    
    return df



# weather data percentage of national
def calculate_weather_percentage(df):
    """
    Calculate the percentage of various weather variables relative to the mean
    of each variable for each country in the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing weather data with columns including 'year',
        'country_cd', and the weather variables to be processed.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with additional columns representing the calculated percentages
        for each weather variable relative to the country-specific means.
    """
    
    variable_column_names = ['ghi', 'total_precipitation', 'mean_sea_level_pressure', 'air_temperature', 'wind_speed']

    # Calculate the mean of the variable for each country
    country_means = df.groupby(['year', 'country_cd'])[variable_column_names].transform('mean')

    # Calculate the percentage for each variable
    for column in variable_column_names:
        df[f"{column}_percentage"] = (df[column] / country_means[column]) * 100
    
    # Drop the original columns
    df.drop(columns=variable_column_names, inplace=True)
    
    return df

# socioeconomic data percentage of national

def calculate_socioeconomic_percentages_sum(df):
    """
    Calculate the percentage of various socioeconomic variables relative to the sum
    of each variable for each country in the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing population and socioeconomic data with columns including 'year',
        'country_cd', and the socioeconomic variables to be processed.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with additional columns representing the calculated percentages
        for each socioeconomic variable relative to the sum of those variables for the country.

    Notes:
    ------
    This function calculates the percentage for each socioeconomic variable by dividing
    the value of each variable for the region by the sum of those variables for the entire country
    and multiplying by 100.
    """
    
    variable_column_names = [
    "gva_pounds_million_Total", "gva_pounds_million_A_E", "gva_pounds_million_AB_1_9",
    "gva_pounds_million_C_10_33", "gva_pounds_million_CA_10_12", "gva_pounds_million_CB_13_15",
    "gva_pounds_million_CC_16_18", "gva_pounds_million_CD_CG_19_23", "gva_pounds_million_CH_24_25",
    "gva_pounds_million_CI_CJ_26_27", "gva_pounds_million_CK_CL_28_30", "gva_pounds_million_CM_31_33",
    "gva_pounds_million_DE_35_39", "gva_pounds_million_F_41_43", "gva_pounds_million_41",
    "gva_pounds_million_42", "gva_pounds_million_43", "gva_pounds_million_G_T",
    "gva_pounds_million_G_45_47", "gva_pounds_million_45", "gva_pounds_million_46",
    "gva_pounds_million_47", "gva_pounds_million_H_49_53", "gva_pounds_million_49_51",
    "gva_pounds_million_52", "gva_pounds_million_53", "gva_pounds_million_I_55_56",
    "gva_pounds_million_55", "gva_pounds_million_56", "gva_pounds_million_J_58_63",
    "gva_pounds_million_58_60", "gva_pounds_million_61_63", "gva_pounds_million_K_64_66",
    "gva_pounds_million_64", "gva_pounds_million_65_66", "gva_pounds_million_L_68",
    "gva_pounds_million_68IMP", "gva_pounds_million_68", "gva_pounds_million_M_69_75",
    "gva_pounds_million_69", "gva_pounds_million_70", "gva_pounds_million_71",
    "gva_pounds_million_72_73", "gva_pounds_million_74", "gva_pounds_million_75",
    "gva_pounds_million_N_77_82", "gva_pounds_million_77", "gva_pounds_million_78_80",
    "gva_pounds_million_81", "gva_pounds_million_82", "gva_pounds_million_O_84",
    "gva_pounds_million_P_85", "gva_pounds_million_Q_86_88", "gva_pounds_million_86",
    "gva_pounds_million_87", "gva_pounds_million_88", "gva_pounds_million_R_90_93",
    "gva_pounds_million_90_91", "gva_pounds_million_92_93", "gva_pounds_million_S_94_96",
    "gva_pounds_million_94", "gva_pounds_million_95", "gva_pounds_million_96",
    "gva_pounds_million_T_97_98"
]

    # Calculate the sum of the variable for each country
    country_means = df.groupby(['year', 'country_cd'])[variable_column_names].transform('sum')

    # Calculate the percentage for each variable
    for column in variable_column_names:
        df[f"{column}_percentage"] = (df[column] / country_means[column]) * 100
    
    # Drop the original columns
    df.drop(columns=variable_column_names, inplace=True)
    
    return df
