
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read the dataset
# data = pd.read_excel("hwc.xlsx")
data = pd.read_csv("hwc.csv")
xlsx_file = "hwc.xlsx"
data.to_excel(xlsx_file, index=False)

print("Original Data:")
print(data.head())

# Calculate planet radius/mass if missing
for i, row in data.iterrows():
    if pd.isna(row['pl_rade']) and not pd.isna(row['pl_bmasse']):
        if row['pl_bmasse'] < 2.04:
            c, s = 0.00346, 0.2790
        elif row['pl_bmasse'] < 132:
            c, s = -0.0925, 0.589
        elif row['pl_bmasse'] < 26600:
            c, s = 1.25, -0.044
        else:
            c, s = -2.85, 0.881
        data.loc[i, 'pl_rade'] = c + row['pl_bmasse'] * s
    elif pd.isna(row['pl_bmasse']) and not pd.isna(row['pl_rade']):
        if row['pl_rade'] < 1.23:
            c, s = 0.00346, 0.2790
        elif row['pl_rade'] < 11.1:
            c, s = -0.0925, 0.589
        else:
            c, s = -2.85, 0.881
        data.loc[i, 'pl_bmasse'] = (row['pl_rade'] - c) / s

# Remove rows with missing values
data = data.dropna()
print("Data after Cleaning:")
print(data.head())

# Normalize using z-score
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data.iloc[:, :-1])  # Exclude the habitability column
normalized_df = pd.DataFrame(normalized_data, columns=data.columns[:-1])
normalized_df['Habitable'] = data['Habitable']
print("Normalized Data:")
print(normalized_df.head())

# Perform SVD for dimensionality reduction
pca = PCA(n_components=8)  # Adjust components as needed
principal_components = pca.fit_transform(normalized_df.iloc[:, :-1])
principal_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(8)])
principal_df['Habitable'] = normalized_df['Habitable']

# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_df['PC1'], principal_df['PC2'], principal_df['PC3'],
           c=principal_df['Habitable'], cmap='viridis')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("3D Scatter Plot of Principal Components")
plt.show()

# Export processed data to Excel
principal_df.to_excel("Processed_hwc.xlsx", index=False)
print("Processed data saved to 'Processed_hwc.xlsx'.")
