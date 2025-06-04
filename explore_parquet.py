import pandas as pd
import numpy as np

# Load the parquet file
print("Loading parquet file...")
df = pd.read_parquet('center_east_Mousq_2_Filttered_with_Adhan_All_2.parquet')

print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\nData types:")
print(df.dtypes)

print(f"\nFirst few rows:")
print(df.head())

print(f"\nBasic statistics:")
print(df.describe())

# Check for datetime columns
datetime_cols = df.select_dtypes(include=['datetime64']).columns
print(f"\nDatetime columns: {datetime_cols.tolist()}")

# Check for null values
print(f"\nNull values per column:")
print(df.isnull().sum())

# If there's a consumption column, show its distribution
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    print(f"\nNumeric columns: {numeric_cols.tolist()}")
    for col in numeric_cols[:3]:  # Show first 3 numeric columns
        print(f"\n{col} statistics:")
        print(f"  Min: {df[col].min()}")
        print(f"  Max: {df[col].max()}")
        print(f"  Mean: {df[col].mean()}")
        print(f"  Std: {df[col].std()}")

# Check unique values in categorical columns
object_cols = df.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"\nCategorical columns: {object_cols.tolist()}")
    for col in object_cols[:3]:  # Show first 3 categorical columns
        nunique = df[col].nunique()
        print(f"\n{col}: {nunique} unique values")
        if nunique < 20:
            print(f"  Values: {df[col].unique()[:10]}")