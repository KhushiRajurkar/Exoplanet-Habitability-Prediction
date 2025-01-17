import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Read the original dataset
data = pd.read_csv("hwc.csv")
print("Original Data Class Distribution:")
print(data['P_HABITABLE'].value_counts())

# Check if 'P_HABITABLE' exists
if 'P_HABITABLE' not in data.columns:
    print("'P_HABITABLE' column not found. Adding it with default values.")
    data['P_HABITABLE'] = 0  # Default to non-habitable if missing
else:
    print("'P_HABITABLE' column found.")
    # Check the data type of 'P_HABITABLE'
    if data['P_HABITABLE'].dtype == 'object':
        print("'P_HABITABLE' is of type object. Attempting to convert to numeric.")
        # Convert to numeric, coercing errors to NaN
        data['P_HABITABLE'] = pd.to_numeric(data['P_HABITABLE'], errors='coerce')
        # Check for any conversion issues
        if data['P_HABITABLE'].isnull().any():
            print("Some 'P_HABITABLE' entries could not be converted to numeric. Filling with mode.")
            mode_value = data['P_HABITABLE'].mode()[0]
            data['P_HABITABLE'] = data['P_HABITABLE'].fillna(mode_value)
        print("Conversion complete. Class distribution:")
        print(data['P_HABITABLE'].value_counts())
    else:
        print("'P_HABITABLE' is already numeric.")
        print("Class distribution:")
        print(data['P_HABITABLE'].value_counts())

# Select numeric columns, ensuring 'P_HABITABLE' is included
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Ensure 'P_HABITABLE' is included in numeric columns
if 'P_HABITABLE' not in numeric_cols:
    numeric_cols.append('P_HABITABLE')

# Select only numeric columns
data = data[numeric_cols]

# Handle missing 'P_RADIUS' and 'P_MASS' columns as before
if 'P_RADIUS' not in data.columns or 'P_MASS' not in data.columns:
    print("Columns 'P_RADIUS' or 'P_MASS' not found. Skipping calculations related to them.")
else:
    print("Handling missing 'P_RADIUS' and 'P_MASS' values.")
    # Calculate planet radius/mass if missing
    for i, row in data.iterrows():
        if pd.isna(row['P_RADIUS']) and not pd.isna(row['P_MASS']):
            if row['P_MASS'] < 2.04:
                c, s = 0.00346, 0.2790
            elif row['P_MASS'] < 132:
                c, s = -0.0925, 0.589
            elif row['P_MASS'] < 26600:
                c, s = 1.25, -0.044
            else:
                c, s = -2.85, 0.881
            data.loc[i, 'P_RADIUS'] = c + row['P_MASS'] * s
        elif pd.isna(row['P_MASS']) and not pd.isna(row['P_RADIUS']):
            if row['P_RADIUS'] < 1.23:
                c, s = 0.00346, 0.2790
            elif row['P_RADIUS'] < 11.1:
                c, s = -0.0925, 0.589
            else:
                c, s = -2.85, 0.881
            data.loc[i, 'P_MASS'] = (row['P_RADIUS'] - c) / s

# Impute missing values instead of dropping rows
print("\nChecking for missing values before imputation:")
print(data.isnull().sum())

# Define an imputer (you can choose different strategies like 'mean', 'median', 'most_frequent')
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

print("\nData after imputation:")
print(data_imputed.head())

print("\nClass distribution after imputation:")
print(data_imputed['P_HABITABLE'].value_counts())

# Separate features and target column
features = data_imputed.drop(columns=['P_HABITABLE'])
target = data_imputed['P_HABITABLE']

# Ensure 'P_HABITABLE' is not included in features
assert 'P_HABITABLE' not in features.columns, "'P_HABITABLE' is still in features!"

# Normalize only the feature columns using z-score
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)
normalized_df = pd.DataFrame(normalized_features, columns=features.columns)

# Add the target column back without normalization
normalized_df['P_HABITABLE'] = target.values
print("\nNormalized Data Class Distribution:")
print(normalized_df['P_HABITABLE'].value_counts())

# Export to Excel
normalized_df.to_excel("hwc.xlsx", index=False)
print("\nNormalized data saved to 'hwc.xlsx'.")