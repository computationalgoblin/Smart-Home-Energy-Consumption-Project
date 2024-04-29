import pandas as pd
import numpy as np
import os

def clean_data(df, output_path=None):
    # Drop 'Unnamed: 0' column
    df.drop(columns=['Unnamed: 0'], inplace=True, axis=1)
    
    # Drop last row
    df = df.iloc[:-1]
    
    # Replace invalid values in 'cloudCover' column
    df['cloudCover'].replace('cloudCover', method="bfill", inplace=True)
    
    # Standardize column names and remove "[kW]"
    df.columns = df.columns.str.replace('[kW]', '').str.strip().str.lower()
    
    # Check and drop duplicate columns
    if all(df['use'] == df['house overall']):
        df.drop(columns=['house overall'], inplace=True)
    if all(df['gen'] == df['solar']):
        df.drop(columns=['solar'], inplace=True)
    
    # Join furnace and kitchen values 
    df['t1'] = df.filter(like='furnace').sum(axis=1)
    df['t2'] = df.filter(like='kitchen').sum(axis=1)
    df.drop(columns=df.filter(like='furnace').columns, inplace=True)
    df.drop(columns=df.filter(like='kitchen').columns, inplace=True)
    df.drop(columns=["summary", "icon"],inplace=True, axis=1)
    df.rename(columns={"t1":"furnace", "t2":"kitchen"}, inplace=True)
    
    # Convert time to datetime and correct unit to minutes
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(df),  freq='min'))

    # Setting time as index
    df = df.set_index('time')

    # Group time into different columns
    df = df.assign(
        month=df.index.month,
        day=df.index.day,
        weekday=df.index.day_name(),
        hour=df.index.hour,
        minute=df.index.minute
    )
    # Export cleaned data to CSV if output_path is provided
    if output_path:
        df.to_csv(output_path, index=True)


def split_csv(input_csv, output_path1, output_path2):
    # Load clean data
    df = pd.read_csv(input_csv)
    
    # Divide the dataframe in two
    half_rows = len(df) // 2
    df1 = df.iloc[:half_rows]
    df2 = df.iloc[half_rows:]
    
    # Export both csv files
    df1.to_csv(output_path1, index=True)
    df2.to_csv(output_path2, index=True)
    
    # Delete files other than df1 and df2 from the directory
    folder_path = os.path.dirname(input_csv)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path != output_path1 and file_path != output_path2 and file_path.endswith('.csv'):
            os.remove(file_path)


