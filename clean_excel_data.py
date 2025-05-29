import pandas as pd
import numpy as np
from datetime import datetime

def clean_excel_file(filename):
    print(f'Processing {filename}...')
    
    # Read the Excel file
    df = pd.read_excel(filename)
    
    # Clean column names (remove extra spaces and special characters)
    df.columns = df.columns.str.strip()
    
    # Convert datetime columns with specific format handling
    for col in ['Entry Datetime', 'Meter Datetime']:
        if col in df.columns:
            # Vectorized string replacement for better performance
            df[col] = df[col].astype(str)
            # Replace the pattern to fix microseconds format
            df[col] = df[col].str.replace(r':(\d{6})$', r'.\1', regex=True)
            # Convert to datetime
            df[col] = pd.to_datetime(df[col], format='%b %d, %Y, %H:%M:%S.%f', errors='coerce')
    
    # Remove rows where datetime parsing failed
    before_shape = df.shape[0]
    df = df.dropna(subset=['Entry Datetime', 'Meter Datetime'])
    after_shape = df.shape[0]
    if before_shape != after_shape:
        print(f'  Removed {before_shape - after_shape} rows with invalid datetime values')
    
    # Clean numeric columns
    numeric_cols = ['Import active power (QI+QIV)[W]', 'Export active power (QII+QIII)[W]']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Remove duplicates
    before_shape = df.shape[0]
    df = df.drop_duplicates()
    after_shape = df.shape[0]
    if before_shape != after_shape:
        print(f'  Removed {before_shape - after_shape} duplicate rows')
    
    print(f'  Final shape: {df.shape}')
    print(f'  Date range: {df["Entry Datetime"].min()} to {df["Entry Datetime"].max()}')
    print(f'  Unique meters: {df["HES Meter Id"].nunique()}')
    
    return df

def combine_and_export():
    # Process both files
    excel_files = ['Readings_LoadProfileElectrical_V2 (1)_100.xlsx', 'Readings_LoadProfileElectrical_V2 (2)_100.xlsx']
    cleaned_dfs = []

    for file in excel_files:
        df_clean = clean_excel_file(file)
        cleaned_dfs.append(df_clean)
        print()

    print('Individual file processing completed.')
    
    # Combine the dataframes
    print('Combining datasets...')
    combined_df = pd.concat(cleaned_dfs, ignore_index=True)
    
    # Remove any duplicates that might exist across files
    before_shape = combined_df.shape[0]
    combined_df = combined_df.drop_duplicates()
    after_shape = combined_df.shape[0]
    if before_shape != after_shape:
        print(f'Removed {before_shape - after_shape} duplicate rows across files')
    
    # Sort by Entry Datetime and Meter ID for better organization
    combined_df = combined_df.sort_values(['HES Meter Id', 'Entry Datetime'])
    
    print(f'Combined dataset shape: {combined_df.shape}')
    print(f'Date range: {combined_df["Entry Datetime"].min()} to {combined_df["Entry Datetime"].max()}')
    print(f'Total unique meters: {combined_df["HES Meter Id"].nunique()}')
    
    # Export to CSV
    csv_filename = 'combined_load_profile_electrical.csv'
    combined_df.to_csv(csv_filename, index=False)
    print(f'Data exported to: {csv_filename}')
    
    # Display basic statistics
    print('\nBasic statistics:')
    print(f'Total records: {len(combined_df):,}')
    print(f'Average import power: {combined_df["Import active power (QI+QIV)[W]"].mean():.2f} W')
    print(f'Average export power: {combined_df["Export active power (QII+QIII)[W]"].mean():.2f} W')
    print(f'Max import power: {combined_df["Import active power (QI+QIV)[W]"].max():.2f} W')
    print(f'Max export power: {combined_df["Export active power (QII+QIII)[W]"].max():.2f} W')
    
    return combined_df

if __name__ == "__main__":
    combined_data = combine_and_export()