import pandas as pd
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the data file
file_path = os.path.join(script_dir, '..', 'data', 'spotify-2023.csv')

# Load the dataset with a different encoding
df = pd.read_csv(file_path, encoding='latin1')

# Explore the structure
print("First 5 rows:")
print(df.head())
print("\nColumns:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe())