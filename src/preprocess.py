import pandas as pd
import numpy as np
import os

# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '..', 'data', 'spotify-2023.csv')
df = pd.read_csv(file_path, encoding='latin1')

# Inspect the data
print("Initial Data Overview:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

# Convert columns that should be numerical but contain commas
df['in_deezer_playlists'] = df['in_deezer_playlists'].str.replace(',', '').astype(float)
df['in_shazam_charts'] = pd.to_numeric(df['in_shazam_charts'].str.replace(',', ''), errors='coerce')

# Function to impute missing values with mean (numerical columns)
def impute_mean(column):
    col = column.dropna()
    mean_val = np.mean(col)
    return column.fillna(mean_val)

# Function to impute missing values with mode (categorical columns)
def impute_mode(column):
    mode_val = column.mode()[0]
    return column.fillna(mode_val)

# Apply imputation
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    df[col] = impute_mean(df[col])

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = impute_mode(df[col])

# Verify no missing values remain
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Function to cap outliers using IQR
def cap_outliers(column):
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.clip(column, lower_bound, upper_bound)

# Apply to numerical columns
for col in numerical_cols:
    df[col] = cap_outliers(df[col])

# Print summary statistics after handling outliers
print("\nSummary Statistics After Handling Outliers:")
print(df[numerical_cols].describe())

# Function for min-max scaling
def min_max_scale(column):
    min_val = np.min(column)
    max_val = np.max(column)
    if max_val != min_val:
        return (column - min_val) / (max_val - min_val)
    else:
        return column

# Apply scaling to numerical columns
for col in numerical_cols:
    df[col] = min_max_scale(df[col])

# Print summary statistics after scaling
print("\nSummary Statistics After Scaling:")
print(df[numerical_cols].describe())

# Add polynomial features (degree 2)
for col in numerical_cols:
    if col != 'streams':
        df[f'{col}_squared'] = df[col] ** 2

# Update numerical_cols to include the new features
numerical_cols = df.select_dtypes(include=[np.number]).columns.difference(['streams'])

# Convert 'streams' to numeric
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
if df['streams'].isnull().sum() > 0:
    df['streams'] = impute_mean(df['streams'])

# Drop irrelevant columns
df = df.drop(columns=['track_name', 'artist(s)_name'], errors='ignore')

# Encode categorical variables
categorical_cols_to_encode = ['key', 'mode']
df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)

# Separate features (X) and target (y)
X = df.drop(columns=['streams'])
y = df['streams']

# Verify data types in X
print("\nData types in X before conversion:")
print(X.dtypes)

# Convert to NumPy arrays
X = X.to_numpy().astype(float)
y = y.to_numpy().astype(float)

# Print shapes
print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)

# Save preprocessed data
np.save('data/X_preprocessed.npy', X)
np.save('data/y_preprocessed.npy', y)
print("\nPreprocessed data saved as 'X_preprocessed.npy' and 'y_preprocessed.npy'.")