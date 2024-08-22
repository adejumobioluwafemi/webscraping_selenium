import os
import re
import time
import io
import requests
import gzip
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

from general import CONFIG


# Define constants
WEATHER_STATION_NUMBER = 34937
DEFAULT_CITY = "Усть-Лабинск"
collection_name = CONFIG['SATELLITE_COLLECTIONS']['DEFAULT']
collection_start_date = CONFIG['SATELLITE_COLLECTIONS'][collection_name]['START_DATE']


def download_file(url):
    """
    Downloads a file from the given URL into memory.

    Parameters:
    -----------
    url : str
        The URL from which to download the file.

    Returns:
    --------
    io.BytesIO
        A BytesIO object containing the downloaded file's content.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
    # Store the response content in a BytesIO buffer
    return io.BytesIO(response.content)


def process_csv_gz(buffer):
    """
    Processes a .csv.gz file buffer, decompresses it, and loads it into a pandas DataFrame.

    Parameters:
    -----------
    buffer : io.BytesIO
        A buffer containing the compressed .csv.gz file.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the data from the decompressed CSV file.

    Notes:
    ------
    - Lines starting with '#' (comments) are skipped during processing.
    - The CSV file is expected to be semicolon-delimited (';').
    """
    # Decompress the .csv.gz file
    with gzip.GzipFile(fileobj=buffer) as gz:
        # Read the decompressed content
        content = gz.read().decode('utf-8')

        # Split the content into lines and skip comments
        lines = content.splitlines()
        data_lines = [line for line in lines if not line.startswith('#')]

        # Convert the list of lines back into a single string
        cleaned_content = '\n'.join(data_lines)

        # Use StringIO to create a file-like object for pandas
        cleaned_buffer = io.StringIO(cleaned_content)

        # Read the CSV data into a DataFrame
        df = pd.read_csv(cleaned_buffer, delimiter=';', encoding='utf-8')

    return df


def get_end_date():
    current_date = datetime.now().date()
    previous_day = current_date - timedelta(days=1)
    return previous_day


def get_weather_df(
        collection_start_date,
        end_date_=get_end_date(),
        headless_opt=True,
        weather_station_num=WEATHER_STATION_NUMBER,
        city=DEFAULT_CITY,
        config=CONFIG,
        save_df=False,
        save_path='rp5_weather.csv'
):
    """
    Scrapes weather data from the rp5.ru website for a specific weather station and date range, 
    processes the data into a DataFrame, and optionally saves the DataFrame to a CSV file.

    Parameters:
    -----------
    collection_start_date : str
        The start date for weather data collection in 'YYYY-MM-DD' format.
    headless_opt : bool, optional, default=True
        If True, the web browser operates in headless mode (no UI).
    weather_station_num : str
        The WMO ID of the weather station to collect data from.
    city : str, optional
        The name of the city to search for on rp5.ru. Defaults to a global variable DEFAULT_CITY.
    config : dict
        Configuration dictionary containing settings such as data folder paths.
    save_df : bool, optional, default=False
        If True, the resulting DataFrame will be saved to the path specified by `save_path`.
    save_path : str, optional, default='rp5_weather.csv'
        The path where the DataFrame will be saved if `save_df` is True.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the weather data from the specified weather station and date range.
    """
    url = 'https://rp5.ru/'

    # Convert end_date_ to datetime if it's a string
    if isinstance(end_date_, str):
        end_date_ = datetime.strptime(end_date_, '%Y-%m-%d').date()

    options = Options()
    options.headless = headless_opt
    # options.add_experimental_option("prefs", {
    #   # "download.default_directory": weather_data_folder_temp,
    #    "download.prompt_for_download": False,
    #    "download.directory_upgrade": True,
    #    "safebrowsing.enabled": True
    # })

    driver = webdriver.Chrome(service=Service(), options=options)

    try:
        driver.get(url)
        WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.ID, 'langButton'))
        ).click()

        russian_option = driver.execute_script("""
            var options = document.querySelectorAll('#langMenu .langMenu-li');
            for (var i = 0; i < options.length; i++) {
                if (options[i].innerText.includes('Русский')) {
                    options[i].click();
                    return options[i];
                }
            }
            return null;
        """)

        if russian_option is None:
            print("Required 'Русский' option not found or not clickable")
            return

        search_location = driver.find_element(By.NAME, 'searchStr')
        search_location.send_keys(city)
        search_location.send_keys(Keys.ENTER)

        search_results_table = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located(
                (By.CSS_SELECTOR, 'table.searchResults'))
        )
        ust_labinsk_link = search_results_table.find_element(
            By.LINK_TEXT, city)
        ust_labinsk_link.click()

        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'wug_link'))
        ).click()

        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'archive_link'))
        ).click()

        weather_station = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.NAME, 'wmo_id'))
        )
        weather_station.clear()
        weather_station.send_keys(weather_station_num)
        time.sleep(5)
        weather_station.send_keys(Keys.ENTER)

        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located(
                (By.XPATH,
                 "//div[@id='tabSynopDLoad' and text()='Скачать архив погоды']")
            )
        ).click()

        # Parse the date string into a datetime object
        date_obj = datetime.strptime(collection_start_date, '%Y-%m-%d')

        # Format the datetime object as day.month.year
        start_date = date_obj.strftime('%d.%m.%Y')
        start_date_list = start_date.split('.')
        # print(start_date_list)

        end_date = end_date_.strftime('%d.%m.%Y')
        end_date_list = end_date.split('.')
        # print(end_date)

        start_date_input = driver.find_element(by=By.NAME, value='ArchDate1')
        start_date_input.click()
        start_year_element = driver.find_element(By.ID, 'dp_year_value')
        start_year_element.click()
        start_year_ = driver.find_element(
            By.ID, 'dp_year' + start_date_list[2])
        start_year_.click()
        start_month_element = driver.find_element(By.ID, 'dp_month_value')
        start_month_element.click()
        start_month_ = driver.find_element(
            By.ID, 'dp_month' + str(int(start_date_list[1]) - 1))
        start_month_.click()
        time.sleep(2)
        start_day_element = driver.find_element(
            By.XPATH, f"//a[text()={str(int(start_date_list[0]))}]")
        start_day_element.click()

        time.sleep(2)
        end_date_input = driver.find_element(by=By.NAME, value='ArchDate2')
        end_date_input.click()
        end_year_element = driver.find_element(By.ID, 'dp_year_value')
        end_year_element.click()
        end_year_ = driver.find_element(By.ID, 'dp_year' + end_date_list[2])
        end_year_.click()
        end_month_element = driver.find_element(By.ID, 'dp_month_value')
        end_month_element.click()
        end_month_ = driver.find_element(
            By.ID, 'dp_month' + str(int(end_date_list[1]) - 1))
        end_month_.click()
        time.sleep(2)
        end_day_element = driver.find_element(
            By.XPATH, f"//a[text()={str(int(end_date_list[0]))}]")
        end_day_element.click()

        wait = WebDriverWait(driver, 10)
        csv_radio_button = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//label[@class='input_radio']//input[@id='format2']")))
        driver.execute_script("arguments[0].click();", csv_radio_button)
        assert csv_radio_button.is_selected(), "The CSV radio button was not selected."

        utf8_radio_button = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//label[@class='input_radio']//input[@id='coding2']")))
        driver.execute_script("arguments[0].click();", utf8_radio_button)
        assert utf8_radio_button.is_selected(), "The UTF-8 radio button was not selected."

        gz_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[contains(text(), 'Выбрать в файл GZ (архив)')]")))
        gz_button.click()

        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "f_result")))
        time.sleep(2)

        download_span = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "f_result"))
        )
        download_link = download_span.find_element(
            By.TAG_NAME, "a").get_attribute("href")

        # Download the file into memory
        file_buffer = download_file(download_link)

        # Process the file buffer
        file_buffer.seek(0)  # Move to the start of the buffer
        # Decompress and read CSV data into DataFrame
        df = process_csv_gz(file_buffer)

    finally:
        driver.quit()

    # Save DataFrame if save_df is True
    if save_df:
        df.to_csv(save_path, index=False)

    return df


def map_columnname_rus_eng(df, column_name_map=None, dataset='rp5', save_df=False, save_path='rp5_weather.csv'):
    """
    Maps Russian column names in the DataFrame to English equivalents and optionally saves the resulting DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The original DataFrame with Russian column names.
    column_name_map : dict, optional
        A dictionary mapping Russian column names to English column names. 
        If None, a default mapping will be used for the 'rp5' dataset.
    dataset : str, optional, default='rp5'
        The dataset type. Currently supports 'rp5' which uses a predefined column mapping.
    save_df : bool, optional, default=False
        If True, the resulting DataFrame will be saved to the path specified by `save_path`.
    save_path : str, optional, default='rp5_weather.csv'
        The path where the DataFrame will be saved if `save_df` is True.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with columns renamed to their English equivalents and selected columns retained.
    """
    df_ = df.copy()

    # Default column name mapping for 'rp5' dataset
    default_column_name_map = {
        'Местное время в Усть-Лабинске': 'datetime',
        'T': 'air_temp',
        'Po': 'atm_pressure',
        'P': 'atm_pressure_norm',
        'Pa': 'pressure_tendency',
        'U': 'humidity_2m_above',
        'DD': 'wind_direction',
        'Ff': 'wind_speed',
        'ff10': 'max_wind_gust_at10m',
        'ff3': 'max_wind_gust',
        'N': 'cloudiness',
        'WW': 'curr_weather',
        'W1': 'past_weather_obs_date1',
        'W2': 'past_weather_obs_date2',
        'Tn': 'minimum_air_temp_last12hrs',
        'Tx': 'maximum_air_temp_last12hrs',
        'Cl': 'nimbus_cloud',
        'Nh': 'num_obs_cloud',
        'H': 'height_base_lowest_cloud(m)',
        'Cm': 'shining_rain_cloud',
        'Ch': 'cirrostratus_cloud',
        'VV': 'vision_km',
        'Td': 'dew_point_temp',
        'RRR': 'prec',
        'tR': 'prec_accum',
        'E': 'soil_surface_condition',
        'Tg': 'minimum_soil_surface_temp',
        "E'": 'ice_cover',
        'sss': 'snow_depth'
    }

    # Use provided column_name_map or fall back to default
    column_name_map = column_name_map or default_column_name_map

    # Rename columns if the dataset is 'rp5'
    if dataset == 'rp5':
        for rus, eng in column_name_map.items():
            if rus in df_.columns:
                df_.rename(columns={rus: eng}, inplace=True)
            else:
                print(f"{rus} does not exist in the default 'column_name_map'")

    # Reset index and reorder columns
    df_.reset_index(inplace=True)
    # print(df_.head())
    df_.columns = ['datetime', 'air_temp', 'atm_pressure', 'atm_pressure_norm',
                   'pressure_tendency', 'humidity_2m_above', 'wind_direction',
                   'wind_speed', 'max_wind_gust_at10m', 'max_wind_gust', 'cloudiness',
                   'curr_weather', 'past_weather_obs_date1', 'past_weather_obs_date2',
                   'minimum_air_temp_last12hrs', 'maximum_air_temp_last12hrs',
                   'nimbus_cloud', 'num_obs_cloud', 'height_base_lowest_cloud(m)',
                   'shining_rain_cloud', 'cirrostratus_cloud', 'vision_km',
                   'dew_point_temp', 'prec', 'prec_accum', 'soil_surface_condition',
                   'minimum_soil_surface_temp', 'ice_cover', 'snow_depth', 'indx']

    # print(df_.head())
    # Select relevant columns
    df_ = df_[['datetime', 'air_temp', 'atm_pressure', 'humidity_2m_above',
               'wind_speed', 'cloudiness', 'prec']]

    # Save DataFrame if save_df is True
    if save_df:
        df_.to_csv(save_path, index=False)

    return df_


def ymd_now(date):
    """
    Convert a datetime object to an integer representation in the format YYYYMMDD.

    Parameters:
    date (pd.Timestamp): A pandas Timestamp object.

    Returns:
    int: The integer representation of the date in the format YYYYMMDD.
    """
    return date.year * 10000 + date.month * 100 + date.day


def create_date_period_ymdp(df_, date_col='datetime', create_ymdp=True, create_period=False, dataset='rp5'):
    """
    Modify a DataFrame to add a new column based on time periods and create a combined date-period identifier.

    The function performs the following steps:
    1. Converts the `datetime` column to a date column.
    2. Adds a `period` column based on specific time intervals.
    3. Creates a `ymdp` column that combines the date and period for unique identification.

    Parameters:
    df_ (pd.DataFrame): The input DataFrame containing weather data.
    date_col (str): The name of the datetime column in the DataFrame. Default is 'datetime'.
    create_ymdp (bool): Whether to create the 'ymdp' column. Default is True.
    create_period (bool): Whether to create the 'period' column. Default is False.
    dataset (str): The dataset type to determine specific operations. Default is 'rp5'.

    Returns:
    pd.DataFrame: The modified DataFrame with the new columns.
    """
    df = df_.copy()

    if dataset == 'rp5':
        # Convert the datetime column to pandas datetime format
        df[date_col] = pd.to_datetime(df[date_col], format='%d.%m.%Y %H:%M')

        # Mapping of time to period codes
        time_mapping = {
            '00:00': 0,
            '03:00': 1,
            '06:00': 2,
            '09:00': 3,
            '12:00': 4,
            '15:00': 5,
            '18:00': 6,
            '21:00': 7
        }

        if create_period or create_ymdp:
            # Create 'period' column based on the time mapping
            df['period'] = df[date_col].dt.strftime('%H:%M').map(time_mapping)

        if create_ymdp:
            # Create 'ymd' column and 'ymdp' column by combining 'ymd' and 'period'
            df['ymd'] = df[date_col].apply(ymd_now)
            df['ymdp'] = df['ymd'] * 10 + df['period']
            ymdp_column = df.pop('ymdp')
            df.insert(0, 'ymdp', ymdp_column)
            # Drop intermediate columns
            df.drop(columns=['ymd', 'period'], inplace=True)

        if create_period:
            # Ensure 'period' column is placed immediately after 'ymdp' if it exists
            period_column = df.pop('period')
            df.insert(1, 'period', period_column)

        # Rename the original datetime column to 'date'
        df.rename(columns={date_col: 'date'}, inplace=True)

    return df


def process_wind_dir(df_, wind_dir_column='wind_direction'):
    """
    Processes the wind direction column in a DataFrame by mapping Russian wind direction descriptions to their English abbreviations.

    Parameters:
    df_ (pd.DataFrame): The input DataFrame containing weather data.
    wind_dir_column (str): The name of the column containing wind direction data. Default is 'wind_direction'.

    Returns:
    pd.DataFrame: The modified DataFrame with wind directions converted to English abbreviations.
    """
    df = df_.copy()
    wind_direction_rus_eng = {
        'Ветер, дующий с запада': 'W',
        'Штиль, безветрие': 'no wind',
        'Ветер, дующий с юго-юго-востока': 'SSE',
        'Ветер, дующий с юго-востока': 'SE',
        'Ветер, дующий с северо-запада': 'NW',
        'Ветер, дующий с севера': 'N',
        'Ветер, дующий с востока': 'E',
        'Ветер, дующий с востоко-северо-востока': 'ENE',
        'Ветер, дующий с востоко-юго-востока': 'ESE',
        'Ветер, дующий с северо-востока': 'NE',
        'Ветер, дующий с юго-запада': 'SW',
        'Ветер, дующий с юго-юго-запада': 'SSW',
        'Ветер, дующий с юга': 'S',
        'Ветер, дующий с северо-северо-запада': 'NNW',
        'Ветер, дующий с западо-юго-запада': 'WSW',
        'Ветер, дующий с западо-северо-запада': 'WNW',
        'Ветер, дующий с северо-северо-востока': 'NNE'
    }

    df[wind_dir_column] = df[wind_dir_column].map(wind_direction_rus_eng)
    return df


def process_prec(df_):
    """
    Processes the precipitation column in a DataFrame by mapping Russian precipitation descriptions to their corresponding numeric values.

    Parameters:
    df_ (pd.DataFrame): The input DataFrame containing weather data.

    Returns:
    pd.DataFrame: The modified DataFrame with a new 'precipitation' column, where precipitation values are converted to float.
    """
    df = df_.copy()
    prec_rus_eng = {
        'Следы осадков': '0.0',
        'Осадков нет': '0.0'
    }

    df['precipitation'] = df['prec'].map(prec_rus_eng).fillna(df['prec'])
    df['precipitation'] = df['precipitation'].astype(float)
    df.drop(columns=['prec'], inplace=True)
    return df

# if df['cloudiness'].unique() gives array(['100%.', '0%.', '70 – 80%.', '40%.', '20–30%.', '60%.', '95%.','5%.', '50%.', nan], dtype=object)
# make the cloudiness column to be value of before % divided 100
# and if the value is something like 70 – 80, add the them together and divide by 2 then divide by 100


def clean_cloudiness(value):
    """
    Cleans and converts cloudiness descriptions into numeric values representing the percentage of cloud coverage.

    Parameters:
    value (str): A string containing the cloudiness description.

    Returns:
    float: The cleaned cloudiness value as a percentage (e.g., 0.5 for 50%), or NaN if the value cannot be processed.
    """
    if isinstance(value, str):
        matches = [int(match) for match in re.findall(r'\d+', value)]
        if len(matches) > 0:
            cleaned_value = np.mean(matches) / 100
        else:
            cleaned_value = np.nan
        return cleaned_value
    else:
        return np.nan


def process_cloudiness(df_):
    """
    Processes the cloudiness column in a DataFrame by mapping Russian cloudiness descriptions to English equivalents and then converting them to numeric values.

    Parameters:
    df_ (pd.DataFrame): The input DataFrame containing weather data.

    Returns:
    pd.DataFrame: The modified DataFrame with cloudiness values converted to numeric percentages.
    """
    df = df_.copy()
    cloudiness_rus_eng_map = {
        'Небо не видно из-за тумана и/или других метеорологических явлений.': 'The sky is not visible due to fog and/or other meteorological phenomena.',
        'Облаков нет.': 'There are no clouds.',
        '90  или более, но не 100%': '90 or more, but not 100%',
        '10%  или менее, но не 0': "10% or less, but not 0",
    }

    cloudiness_map = {
        'The sky is not visible due to fog and/or other meteorological phenomena.': '0%.',
        'There are no clouds.': '0%.',
        '90 or more, but not 100%': '95%.',
        "10% or less, but not 0": '5%.'
    }

    df['cloudiness'] = df['cloudiness'].map(
        cloudiness_rus_eng_map).fillna(df['cloudiness'])
    df['cloudiness'] = df['cloudiness'].map(
        cloudiness_map).fillna(df['cloudiness'])
    df['cloudiness'] = df['cloudiness'].apply(clean_cloudiness)
    return df


def populate_na(df, window_size=30):
    """
    fillna in each column with the mean of the previous last 30 values before each nan in the respective column
    """
    # Identify non-datetime columns
    non_datetime_cols = df.select_dtypes(
        exclude=['datetime', 'datetime64']).columns

    # Apply rolling mean only to non-datetime columns
    rolling_mean = df[non_datetime_cols].rolling(
        window=window_size, min_periods=1).mean()

    # Fill NaNs in non-datetime columns with the rolling mean
    df[non_datetime_cols] = df[non_datetime_cols].fillna(rolling_mean)

    return df


def rp5_data_daily(df_, date_col='date', columns_to_sum=None, columns_to_avg=None, save_path=None):
    """
    Aggregates weather data from the rp5 dataset on a daily basis by summing or averaging specified columns.

    Parameters:
    df_ (pd.DataFrame): The input DataFrame containing weather data.
    date_col (str): The name of the column containing date information. Default is 'date'.
    columns_to_sum (list of str, optional): A list of column names to be summed for each day.
    columns_to_avg (list of str, optional): A list of column names to be averaged for each day.

    Returns:
    pd.DataFrame: A new DataFrame with daily aggregated data, where the specified columns are either summed or averaged.

    Notes:
    - The date column is converted to datetime format if it is not already.
    - The columns specified in `columns_to_sum` and `columns_to_avg` are converted to floats before aggregation.
    - The returned DataFrame will contain the summed columns first (if any), followed by the averaged columns.
    - The `date_col` column in the returned DataFrame will only contain the date part (YYYY-MM-DD).
    """
    df = df_.copy()
    # Convert to datetime if not already
    df[date_col] = pd.to_datetime(df[date_col])
    result_df = pd.DataFrame()

    if columns_to_sum is not None:
        for col in columns_to_sum:
            if col in df.columns:
                # Convert column to float if not already
                df[col] = df[col].astype(float)
        # Extract date part after converting to datetime
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        sum_df = df.groupby(date_col)[columns_to_sum].sum().reset_index()
        result_df = pd.concat([result_df, sum_df], axis=1)

    if columns_to_avg is not None:
        if all(col in df.columns for col in columns_to_avg):
            for col in columns_to_avg:
                # Convert column to float if not already
                df[col] = df[col].astype(float)
            # Extract date part after converting to datetime
            df[date_col] = pd.to_datetime(df[date_col]).dt.date
            avg_df = df.groupby(date_col)[columns_to_avg].mean().reset_index()
            avg_df.drop(columns=date_col, inplace=True)
            result_df = pd.concat([result_df, avg_df], axis=1)

    if save_path:
        df_.to_csv(save_path, index=False)

    return result_df


def get_latest_date_from_csv(file_path):
    """Get the maximum date from the existing weather data CSV file."""
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path, parse_dates=['date'])
        max_date = df_existing['date'].max()  # .date()
        return max_date
    return None


def get_new_data(start_date, end_date):
    """Fetch weather data from the given start_date to end_date."""
    return get_weather_df(
        collection_start_date=start_date,
        end_date_=end_date,
        headless_opt=True,
        weather_station_num=WEATHER_STATION_NUMBER,
        city=DEFAULT_CITY,
        config=CONFIG
    )


def update_weather_data2(weather_data_file_path):
    # Load existing weather data if it exists
    if os.path.exists(weather_data_file_path):
        latest_date = get_latest_date_from_csv(weather_data_file_path)
    else:
        latest_date = None

    # Determine the new data range
    current_date = datetime.now().date()
    if latest_date:
        new_data_start_date = latest_date + timedelta(days=1)
    else:
        new_data_start_date = datetime.strptime(
            collection_start_date, '%Y-%m-%d').date()

    if new_data_start_date <= current_date - timedelta(days=1):
        # One day before the current date
        new_data_end_date = current_date - timedelta(days=1)
        new_data = get_new_data(new_data_start_date.strftime(
            '%Y-%m-%d'), new_data_end_date.strftime('%Y-%m-%d'))

        # Process the new data
        new_data = map_columnname_rus_eng(new_data)
        new_data = create_date_period_ymdp(new_data)
        new_data = process_prec(new_data)
        new_data = process_cloudiness(new_data)
        new_data_filled = populate_na(new_data, window_size=30)

        # Daily aggregation of new data
        columns_to_sum = ['precipitation']
        columns_to_avg = ['air_temp', 'atm_pressure',
                          'humidity_2m_above', 'wind_speed', 'cloudiness']
        df_daily = rp5_data_daily(df_=new_data_filled, date_col='date',
                                  columns_to_sum=columns_to_sum, columns_to_avg=columns_to_avg)

        # Append the new data to the existing CSV
        if os.path.exists(weather_data_file_path):
            df_existing = pd.read_csv(
                weather_data_file_path, parse_dates=['date'])
            # Ensure no duplicate dates and concatenate
            combined_df = pd.concat(
                [df_existing, df_daily], ignore_index=True).drop_duplicates(subset=['date'])
        else:
            combined_df = df_daily

        # Save the updated data
        combined_df.to_csv(weather_data_file_path, index=False)

    return combined_df


def update_weather_rp5(weather_data_file_path):
    # Load existing weather data if it exists
    if os.path.exists(weather_data_file_path):
        latest_date = get_latest_date_from_csv(weather_data_file_path)
    else:
        latest_date = None

    # Determine the new data range
    current_date = datetime.now().date()
    if latest_date:
        if isinstance(latest_date, str):
            latest_date = datetime.strptime(latest_date, '%Y-%m-%d')
        new_data_start_date = (latest_date + timedelta(days=1)).date()
    else:
        new_data_start_date = datetime.strptime(
            collection_start_date, '%Y-%m-%d').date()

    if new_data_start_date > current_date - timedelta(days=1):
        print("rp5: No new date was found.")
        return

    # Fetch and process new data
    # One day before the current date
    new_data_end_date = current_date - timedelta(days=1)
    new_data = get_new_data(new_data_start_date.strftime(
        '%Y-%m-%d'), new_data_end_date.strftime('%Y-%m-%d'))

    # Process the new data
    new_data = map_columnname_rus_eng(new_data)
    new_data = create_date_period_ymdp(new_data)
    new_data = process_prec(new_data)
    new_data = process_cloudiness(new_data)
    new_data_filled = populate_na(new_data, window_size=30)

    # Daily aggregation of new data
    columns_to_sum = ['precipitation']
    columns_to_avg = ['air_temp', 'atm_pressure',
                      'humidity_2m_above', 'wind_speed', 'cloudiness']
    df_daily = rp5_data_daily(df_=new_data_filled, date_col='date',
                              columns_to_sum=columns_to_sum, columns_to_avg=columns_to_avg)

    # Append the new data to the existing CSV
    if os.path.exists(weather_data_file_path):
        df_existing = pd.read_csv(weather_data_file_path, parse_dates=['date'])
        # Ensure no duplicate dates and concatenate
        combined_df = pd.concat(
            [df_existing, df_daily], ignore_index=True).drop_duplicates(subset=['date'])
    else:
        combined_df = df_daily

    # Save the updated data
    combined_df.to_csv(weather_data_file_path, index=False)

    print("rp5: Weather data updated. New data from {} to {} was added.".format(
        new_data_start_date, new_data_end_date))
