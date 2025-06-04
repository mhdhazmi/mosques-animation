import pandas as pd
import numpy as np
from mosque_anomaly_detection import MosqueAnomalyDetector, run_mosque_anomaly_detection

def explore_parquet_structure(file_path):
    """First explore the parquet file to understand its structure"""
    print(f"Loading {file_path}...")
    
    try:
        # Load a sample first to check structure
        df_sample = pd.read_parquet(file_path, engine='pyarrow').head(1000)
        
        print(f"\nDataset sample shape: {df_sample.shape}")
        print(f"\nColumn names:")
        for i, col in enumerate(df_sample.columns):
            print(f"  {i}: {col}")
        
        print(f"\nData types:")
        print(df_sample.dtypes)
        
        print(f"\nFirst few rows:")
        print(df_sample.head(3))
        
        # Identify datetime columns
        datetime_cols = []
        for col in df_sample.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_cols.append(col)
                # Try to convert to datetime
                try:
                    df_sample[col] = pd.to_datetime(df_sample[col])
                    print(f"\n{col} successfully converted to datetime")
                except:
                    print(f"\n{col} could not be converted to datetime")
        
        # Identify numeric columns that might be consumption
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
        consumption_candidates = []
        
        for col in numeric_cols:
            # Look for keywords that indicate consumption/power
            keywords = ['consumption', 'power', 'kwh', 'kw', 'energy', 'import', 'export', 'active', 'reactive']
            if any(keyword in col.lower() for keyword in keywords):
                consumption_candidates.append(col)
                print(f"\nPotential consumption column: {col}")
                print(f"  Range: {df_sample[col].min():.2f} - {df_sample[col].max():.2f}")
                print(f"  Mean: {df_sample[col].mean():.2f}")
        
        # Look for meter/device ID columns
        id_candidates = []
        for col in df_sample.columns:
            if 'meter' in col.lower() or 'device' in col.lower() or 'id' in col.lower():
                id_candidates.append(col)
                nunique = df_sample[col].nunique()
                print(f"\nPotential ID column: {col}")
                print(f"  Unique values: {nunique}")
                if nunique < 20:
                    print(f"  Sample values: {df_sample[col].unique()[:5]}")
        
        return df_sample, datetime_cols, consumption_candidates, id_candidates
        
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return None, [], [], []

def run_analysis_with_detected_columns(file_path):
    """Run anomaly detection with automatically detected columns"""
    
    # First explore the file
    df_sample, datetime_cols, consumption_candidates, id_candidates = explore_parquet_structure(file_path)
    
    if df_sample is None:
        print("Failed to load parquet file")
        return
    
    print("\n" + "="*60)
    print("COLUMN DETECTION SUMMARY")
    print("="*60)
    
    # Try to identify the best columns
    datetime_col = None
    consumption_col = None
    meter_col = None
    
    # Select datetime column
    if datetime_cols:
        datetime_col = datetime_cols[0]  # Take the first datetime column
        print(f"Selected datetime column: {datetime_col}")
    else:
        # Look for any column that might contain dates
        for col in df_sample.columns:
            try:
                pd.to_datetime(df_sample[col].head(10))
                datetime_col = col
                print(f"Found datetime column: {datetime_col}")
                break
            except:
                continue
    
    # Select consumption column
    if consumption_candidates:
        # Prefer columns with 'active' and 'power' or 'import'
        for col in consumption_candidates:
            if 'active' in col.lower() and ('power' in col.lower() or 'import' in col.lower()):
                consumption_col = col
                break
        if not consumption_col:
            consumption_col = consumption_candidates[0]
        print(f"Selected consumption column: {consumption_col}")
    else:
        # If no candidates found, look for any numeric column with reasonable values
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_sample[col].max() > 0 and df_sample[col].max() < 1000000:  # Reasonable power range
                consumption_col = col
                print(f"Selected numeric column as consumption: {consumption_col}")
                break
    
    # Select meter ID column (optional)
    if id_candidates:
        for col in id_candidates:
            if df_sample[col].nunique() > 1 and df_sample[col].nunique() < len(df_sample):
                meter_col = col
                print(f"Selected meter ID column: {meter_col}")
                break
    
    if not datetime_col or not consumption_col:
        print("\nERROR: Could not identify required columns")
        print("Please specify the column names manually in the script")
        return
    
    print("\n" + "="*60)
    print("RUNNING ANOMALY DETECTION")
    print("="*60)
    
    try:
        # Run the anomaly detection
        df_results, summary = run_mosque_anomaly_detection(
            file_path, datetime_col, consumption_col, meter_col
        )
        
        print("\nAnomaly detection completed successfully!")
        
        # Additional mosque-specific insights
        print("\n" + "="*60)
        print("MOSQUE-SPECIFIC INSIGHTS")
        print("="*60)
        
        if 'prayer_period' in df_results.columns:
            prayer_consumption = df_results.groupby('prayer_period')[consumption_col].agg(['mean', 'max', 'count'])
            print("\nConsumption by Prayer Period:")
            print(prayer_consumption.sort_values('mean', ascending=False))
            
        if 'is_friday' in df_results.columns:
            friday_comparison = df_results.groupby('is_friday')[consumption_col].mean()
            print(f"\nAverage consumption:")
            print(f"  Regular days: {friday_comparison[0]:.2f}")
            print(f"  Fridays: {friday_comparison[1]:.2f}")
            print(f"  Friday increase: {(friday_comparison[1]/friday_comparison[0] - 1)*100:.1f}%")
        
    except Exception as e:
        print(f"\nError during anomaly detection: {e}")
        print("\nTrying with manual column specification...")
        
        # Common column patterns for smart meter data
        common_patterns = [
            ('Meter Datetime', 'Import active power (QI+QIV)[W]', 'HES Meter Id'),
            ('timestamp', 'consumption', 'meter_id'),
            ('datetime', 'power', 'device_id'),
            ('date_time', 'kWh', 'meter_number')
        ]
        
        for dt_col, cons_col, meter_col in common_patterns:
            if dt_col in df_sample.columns and cons_col in df_sample.columns:
                print(f"\nTrying with: datetime={dt_col}, consumption={cons_col}")
                try:
                    df_results, summary = run_mosque_anomaly_detection(
                        file_path, dt_col, cons_col, 
                        meter_col if meter_col in df_sample.columns else None
                    )
                    print("\nAnomaly detection completed successfully!")
                    break
                except:
                    continue

if __name__ == "__main__":
    # Run the analysis
    file_path = 'center_east_Mousq_2_Filttered_with_Adhan_All_2.parquet'
    run_analysis_with_detected_columns(file_path)