import h5py
import numpy as np
import pandas as pd
import os
import random

# Define the HDF5 file path
hdf5_file_path = "C:/Gal_Msc/Dataset/14/14_XYZ_CNN_Features_Patches.h5"

# Function to read data and dataset names from HDF5
def read_data_and_names(file):
    def extract_data(file, group_name):
        data = []
        dataset_names = []
        with h5py.File(file, 'r') as f:
            group = f[group_name]
            for key in group.keys():
                dataset = np.array(group[key])
                data.append(dataset)
                dataset_names.append(key)  # Get the dataset name (key) from the HDF5 group
        return np.array(data), dataset_names

    X, dataset_names = extract_data(file, 'Features/Test')
    return X, dataset_names

# Load data and dataset names
X, dataset_names = read_data_and_names(hdf5_file_path)


# Ensure all values are numeric
X = X.astype(np.float32)

# Select 5 random samples
num_samples = 5
selected_indices = random.sample(range(len(dataset_names)), num_samples)

# Prepare data for the Excel file
selected_data = {
    "Sample Name": [f"{dataset_names[i]} patch {i}" for i in selected_indices],
    "Features": [X[i].tolist() for i in selected_indices]  # Convert features to list for Excel storage
}

# Create a DataFrame for the selected samples
selected_samples_df = pd.DataFrame(selected_data)

# Save the selected samples to an Excel file
output_file_path = "C:/Gal_Msc/Dataset/selected_samples_features.xlsx"
selected_samples_df.to_excel(output_file_path, index=False)

print(f"Selected samples saved to {output_file_path}")
