import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib  # For loading the saved model
import os

# Define the new HDF5 file path
dataset_og_name='test1'
dataset_name = "test1_All_Features_Patches"


new_samples_file_path = f"C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/{dataset_og_name}/{dataset_name}.h5"
trained_model_path = "C:/Gal_Msc/Ipublic-repo/inverse-model-frustrated-composites/saved_models/RandomForest/14_All_CNN_Patches_20240724.pkl"

# If there is ground truth information:
is_ground_truth = False
dataset_labels_name = "VAL1_All_CNN_Labels_Patches"
new_samples_labels_file_path = f"C:/Gal_Msc/Dataset/VAL1/{dataset_labels_name}.h5"

# Function to read new HDF5 data
def read_new_hdf5_data_and_name(file):
    data = []
    dataset_name = []
    with h5py.File(file, 'r') as f:
        group = f['Features/Test']
        for key in group.keys():
             dataset = np.array(group[key])
             data.append(dataset)
             dataset_name.append(key)  # Get the dataset name (key) from the HDF5 group
        x_new = np.array(data)

    return x_new, dataset_name
# Function to read ground truth labels if available


def read_labels(file):
    labels = []
    with h5py.File(file, 'r') as f:
        group = f['Labels/Test']
        for key in group.keys():
            label = np.array(group[key])
            labels.append(label)
        y_true = np.array(labels).flatten()
    return y_true


# Load new data and dataset name
X_new, dataset_names = read_new_hdf5_data_and_name(new_samples_file_path)

# Reshape if necessary (flatten each sample)
X_new = X_new.reshape(X_new.shape[0], -1, order='F')

# Ensure all values are numeric
X_new = X_new.astype(np.float32)

# Load the trained model
rf = joblib.load(trained_model_path)

# Predict labels for the new data
y_pred_new = rf.predict(X_new)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({
    "Sample Name": [f"{dataset_names[i]} " for i in range(len(y_pred_new))],
    "Prediction": y_pred_new
})

# If ground truth labels are available, include them in the DataFrame
if is_ground_truth:
    y_true = read_labels(new_samples_labels_file_path)
    predictions_df['Ground Truth'] = y_true


# Save predictions and ground truth to an Excel file
output_file_path = f"C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/{dataset_og_name}/{dataset_name}_predictions.xlsx"
predictions_df.to_excel(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
