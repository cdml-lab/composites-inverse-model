
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import joblib
import datetime
import matplotlib
matplotlib.use('Agg')  # remove if you want to see plots
import seaborn as sns

import matplotlib.pyplot as plt
import os


"""# Define Paths to the HDF5 Files"""

# Step 3: Define paths to the HDF5 files

og_dataset_name = '14_16small'
dataset_name = '14_16small_MaxCV_overlap0'
patches = '_Patches'

train = 'yes'


num_rows, num_cols, num_depth = 5, 5, 3  # Adjusted to match the actual feature structure. The depth isn't used in the code at all so doesn't really need adjusting.


current_date = datetime.datetime.now().strftime("%Y%m%d")
model_name = f"{dataset_name}{patches}_{current_date}.pkl"

# Define the path and name for saving the model
save_model_path = 'C:/Gal_Msc/Ipublic-repo/inverse-model-frustrated-composites/saved_model/RandomForest/' + model_name
load_model_path = 'C:/Gal_Msc/Ipublic-repo/inverse-model-frustrated-composites/saved_model/RandomForest/' + model_name

# Define the path for saving the importances
excel_path = os.path.join(os.path.dirname(save_model_path), f"{dataset_name}_importances_{current_date}.xlsx")


# Set dataset files
features_file = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + og_dataset_name + '/' + dataset_name + '_Features' + patches + '.h5'
labels_file = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + og_dataset_name + '/' + dataset_name + '_Labels' + patches + '.h5'

#CUDA
"""# Read the HDF5 Files"""
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_hdf5_data(features_file, labels_file):
    def extract_data(file, group_name):
        data = []
        with h5py.File(file, 'r') as f:
            group = f[group_name]
            for key in group.keys():
                # Read the dataset within each group
                dataset = np.array(group[key])
                data.append(dataset)
        return np.array(data)

    X_train = extract_data(features_file, 'Features/Train')
    X_test = extract_data(features_file, 'Features/Test')
    y_train = extract_data(labels_file, 'Labels/Train')
    y_test = extract_data(labels_file, 'Labels/Test')

    # Reshape if necessary (flatten each sample)
    X_train = X_train.reshape(X_train.shape[0], -1, order='F')
    X_test = X_test.reshape(X_test.shape[0], -1, order='F')
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = read_hdf5_data(features_file, labels_file)

# Print shapes of the datasets for debugging
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

"""# Filter Out Non-Numeric Data (Titles)"""

def filter_numeric_data(data):
    # Attempt to convert each element to float, filter out non-numeric values
    numeric_data = []
    for element in data:
        try:
            numeric_element = np.array(element, dtype=np.float32)
            numeric_data.append(numeric_element)
        except ValueError:
            # Skip non-numeric elements
            continue
    return np.array(numeric_data)

X_train_filtered = filter_numeric_data(X_train)
X_test_filtered = filter_numeric_data(X_test)
y_train_filtered = filter_numeric_data(y_train)
y_test_filtered = filter_numeric_data(y_test)

# Print shapes of the filtered datasets for debugging
print(f"X_train_filtered shape: {X_train_filtered.shape}")
print(f"X_test_filtered shape: {X_test_filtered.shape}")
print(f"y_train_filtered shape: {y_train_filtered.shape}")
print(f"y_test_filtered shape: {y_test_filtered.shape}")


"""# Ensure All Values are Numeric"""

# Ensure all values are numeric
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train_filtered = y_train_filtered.astype(np.int32)
y_test_filtered = y_test_filtered.astype(np.int32)
print("final feature shape", X_train.shape)
print("final label shape", y_train_filtered.shape)

"""# Train Initial Model and Compute Feature Importance"""

rf = RandomForestClassifier(n_estimators=500, random_state=42)

if train == 'yes':
    # Save the trained model
    joblib.dump(rf, save_model_path)
    print("Model saved to..." + save_model_path)
    rf.fit(X_train, y_train_filtered)
else:
    # Load the pre-trained model
    rf = joblib.load(load_model_path)
    print(f"Model loaded from {load_model_path}")

# Compute feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature importances
print("Feature importances:")
for f in range(X_train.shape[1]):
    print(f"Feature {indices[f]}: {importances[indices[f]]}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict and evaluate the model
y_pred = rf.predict(X_test_filtered)
accuracy = accuracy_score(y_test_filtered, y_pred) # Use y_test_filtered here
print(f"Accuracy with all features: {accuracy}")

# Compute the confusion matrix
cm = confusion_matrix(y_test_filtered, y_pred) # And here

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

"""# Visualize Feature Importances"""

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# Assuming X_train is shaped (num_samples, num_rows * num_cols)
num_samples, num_features = X_train.shape

# Map feature importances to (row, column, depth) pairs
feature_importances = rf.feature_importances_.reshape(num_rows, num_cols, num_depth)

# Sum importances for each row, column, and depth channel
row_importances = feature_importances.sum(axis=(1, 2))
column_importances = feature_importances.sum(axis=(0, 2))
depth_importances = feature_importances.sum(axis=(0, 1))

# Visualize the importances
plt.figure(figsize=(18, 6))

# Row importances
plt.subplot(1, 3, 1)
sns.heatmap(row_importances.reshape(-1, 1), annot=True, cmap='viridis', cbar=False)
plt.title('Row Importances')
plt.xlabel('Importance')
plt.ylabel('Row Index')

# Column importances
plt.subplot(1, 3, 2)
sns.heatmap(column_importances.reshape(1, -1), annot=True, cmap='viridis', cbar=False)
plt.title('Column Importances')
plt.ylabel('Importance')
plt.xlabel('Column Index')

# Depth importances
plt.subplot(1, 3, 3)
sns.heatmap(depth_importances.reshape(1, -1), annot=True, cmap='viridis', cbar=False)
plt.title('Depth Importances')
plt.ylabel('Importance')
plt.xlabel('Depth Channel')

plt.tight_layout()
plt.show()


# Number of samples to display
num_samples = 20

# Ensure there are enough samples to select from
if num_samples > len(y_test):
    num_samples = len(y_test)

# Randomly select indices
random_indices = np.random.choice(len(y_test), num_samples, replace=False)

# Get the ground truth and predicted labels for the selected samples
selected_y_test = y_test[random_indices]
selected_y_pred = y_pred[random_indices]

# Plot the ground truth and predicted labels
fig, ax = plt.subplots(figsize=(10, 6))

# Create a table to display the ground truth and predicted labels
cell_text = []
for i in range(num_samples):
    cell_text.append([random_indices[i], selected_y_test[i], selected_y_pred[i]])

# Set up the table
table = ax.table(cellText=cell_text, colLabels=['Index', 'Ground Truth', 'Predicted'], cellLoc='center', loc='center')

# Remove the x and y axis
ax.axis('off')

# Adjust the table to fit the figure
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

plt.title('Random 20 Samples: Ground Truth vs Predicted Labels')
plt.show()
# Compute feature importances
importances = rf.feature_importances_

# Ensure that importances match the expected shape
expected_shape = num_rows * num_cols * num_depth
if importances.shape[0] != expected_shape:
    raise ValueError(f"Unexpected number of importances: expected {expected_shape}, got {importances.shape[0]}")

# Reshape feature importances to (row, col, depth) structure
feature_importances = importances.reshape(num_rows, num_cols, num_depth)

# Sum importances for each row, column, and depth
row_importances = feature_importances.sum(axis=(1, 2))
column_importances = feature_importances.sum(axis=(0, 2))
depth_importances = feature_importances.sum(axis=(0, 1))

# Visualize the importances
plt.figure(figsize=(18, 6))

# Row importances
plt.subplot(1, 3, 1)
sns.heatmap(row_importances.reshape(-1, 1), annot=True, cmap='viridis', cbar=False)
plt.title('Row Importances')
plt.xlabel('Importance')
plt.ylabel('Row Index')

# Column importances
plt.subplot(1, 3, 2)
sns.heatmap(column_importances.reshape(1, -1), annot=True, cmap='viridis', cbar=False)
plt.title('Column Importances')
plt.ylabel('Importance')
plt.xlabel('Column Index')

# Depth importances
plt.subplot(1, 3, 3)
sns.heatmap(depth_importances.reshape(1, -1), annot=True, cmap='viridis', cbar=False)
plt.title('Depth Importances')
plt.ylabel('Importance')
plt.xlabel('Depth Channel')

plt.tight_layout()
plt.show()

# Prepare data for saving
column_importance_data = {
    'Column': [f'Col {i}' for i in range(num_cols)],
    'Importance': column_importances
}

row_importance_data = {
    'Row': [f'Row {i}' for i in range(num_rows)],
    'Importance': row_importances
}

depth_importance_data = {
    'Depth Channel': [f'Depth {i}' for i in range(num_depth)],
    'Importance': depth_importances
}

# Convert to DataFrame
column_importance_df = pd.DataFrame(column_importance_data)
row_importance_df = pd.DataFrame(row_importance_data)
depth_importance_df = pd.DataFrame(depth_importance_data)

# Prepare the matrix for row x column importances
row_col_importance_matrix = pd.DataFrame(feature_importances.sum(axis=2),
                                         index=[f'Row {i}' for i in range(num_rows)],
                                         columns=[f'Col {i}' for i in range(num_cols)])

# Prepare the matrix for depth channel importances by x/y location
depth_xy_importance_matrix = {}
for depth in range(num_depth):
    depth_xy_importance_matrix[f'Depth {depth}'] = pd.DataFrame(feature_importances[:, :, depth],
                                                                index=[f'Row {i}' for i in range(num_rows)],
                                                                columns=[f'Col {i}' for i in range(num_cols)])

# Save to Excel
with pd.ExcelWriter(excel_path) as writer:
    column_importance_df.to_excel(writer, sheet_name='Column Importances', index=False)
    row_importance_df.to_excel(writer, sheet_name='Row Importances', index=False)
    depth_importance_df.to_excel(writer, sheet_name='Depth Importances', index=False)
    row_col_importance_matrix.to_excel(writer, sheet_name='Row-Column Matrix')

    for depth_name, matrix in depth_xy_importance_matrix.items():
        matrix.to_excel(writer, sheet_name=f'{depth_name} Matrix')

print(f"Column, row, depth importances, and matrices saved to {excel_path}")