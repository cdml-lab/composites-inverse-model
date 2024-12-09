# Imports
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

# Input Files

model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\30-35_MaxMinCurvature_20241112.pkl"

new_samples_file_path_features = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\100\100_MaxCV_Features_Reshaped.h5"
new_samples_file_path_labels = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\100\100_MaxCV_Labels_Reshaped.h5"
excel_file_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\rhino_to_model_inverse.xlsx"


features_channels = 8
labels_channels = 1

channels_to_use = [0,1,2,3,4,5,6,7]

features_main_group = 'Features'
labels_main_group = 'Labels'
category = 'Train'
compute_certainty = True

x=1 # Random sample selection

# Normalization Aspect
global_labels_max = 180.0
global_labels_min = 0.0


#
global_features_max = [10.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
global_features_min = [-10.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5]
# global_features_max = 10.0
# global_features_min = -10.0



# Model Architecture
class OurModel(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(OurModel, self).__init__()

        self.conv_1 = torch.nn.Conv2d(in_channels=features_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_8 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_9 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_10 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_11 = torch.nn.Conv2d(in_channels=512, out_channels=labels_channels, kernel_size=3, padding=1)

        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=32)
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=64)
        self.batch_norm_3 = torch.nn.BatchNorm2d(num_features=64)
        self.batch_norm_4 = torch.nn.BatchNorm2d(num_features=128)
        self.batch_norm_5 = torch.nn.BatchNorm2d(num_features=128)
        self.batch_norm_6 = torch.nn.BatchNorm2d(num_features=256)
        self.batch_norm_7 = torch.nn.BatchNorm2d(num_features=256)
        self.batch_norm_8 = torch.nn.BatchNorm2d(num_features=512)
        self.batch_norm_9 = torch.nn.BatchNorm2d(num_features=512)
        self.batch_norm_10 = torch.nn.BatchNorm2d(num_features=512)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)

        x = self.dropout(x)  # Dropout after every 3 layers

        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.relu(x)

        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.relu(x)

        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.relu(x)

        x = self.dropout(x)  # Dropout after every 3 layers

        x = self.conv_7(x)
        x = self.batch_norm_7(x)
        x = self.relu(x)

        x = self.conv_8(x)
        x = self.batch_norm_8(x)
        x = self.relu(x)

        x = self.conv_9(x)
        x = self.batch_norm_9(x)
        x = self.relu(x)

        x = self.conv_10(x)
        x = self.batch_norm_10(x)
        x = self.relu(x)

        x = self.conv_11(x)
        # Don't apply ReLU if this is a regression problem, so no activation on the final layer
        return x

#Functions
def export_each_channel_to_excel(prediction_np, base_save_path="predictions_channel"):

    df = pd.DataFrame(prediction_np)

    # Define a unique filename for each channel
    save_path = f"{base_save_path}.xlsx"
    df.to_excel(save_path, index=False, sheet_name=f"Channel_1", header=False)

    print(f"predictions exported to {save_path}")

def load_features_h5_data(features_file, features_main_group, category, global_features_min, global_features_max):
    """
    Load data from an HDF5 file for the specified main group and category, with normalization.

    Args:
        features_file (str): Path to the features HDF5 file.
        feature_main_group (str): Main group within the features HDF5 file ('Features').
        category (str): Subgroup within the main group ('Train' or 'Test').
        global_feature_min (float): Global minimum value for feature normalization.
        global_feature_max (float): Global maximum value for feature normalization.

    Returns:
        torch.Tensor: The normalized feature tensor.
    """
    data = []

    with h5py.File(features_file, 'r') as f:
        group = f[features_main_group][category]
        for dataset_name in group.keys():
            dataset = np.array(group[dataset_name])
            if dataset.size == 0:
                continue  # Skip empty datasets
            data.append(dataset)

    # Convert to a single NumPy array
    data = np.array(data).squeeze()

    # print(f"Features Before Normalization: {data}")
    print(f"Original data shape {data.shape}")
    # print(data)

    # Normalize the features using the global min and max
    normalized_data = (data - global_features_min) / (global_features_max - global_features_min)

    # Convert to PyTorch tensor and add a batch dimension
    feature_tensor = torch.tensor(normalized_data, dtype=torch.float32)

    return feature_tensor

def load_labels_h5_data(labels_file, labels_main_group, category):

    data = []

    with h5py.File(labels_file, 'r') as f:
        group = f[labels_main_group][category]
        for dataset_name in group.keys():
            dataset = np.array(group[dataset_name])
            if dataset.size == 0:
                continue  # Skip empty datasets
            data.append(dataset)

    # Convert to a single NumPy array
    data = np.array(data)


    # Convert to PyTorch tensor and add a batch dimension
    feature_tensor = torch.tensor(data, dtype=torch.float32)

    return data

def excel_to_np_array(file_path, sheet_name='Sheet1', global_features_max=10, global_features_min=-10):
    f"""
    Reads an Excel file with X amount of columns and 300 rows and converts it into a NumPy array
    of shape (20, 15, features channels), with each column representing a channel and reorganizing the
    rows using Fortran order.

    Parameters:
    - file_path (str): Path to the Excel file.
    - sheet_name (str): Name of the sheet in the Excel file. Default is 'Sheet1'.

    Returns:
    - np.ndarray: A NumPy array of shape (20, 15, features_channels) with the data organized by (height, width, channels).
    """
    # Read the xlsx file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Drop the header and convert to NumPy array
    data = df.to_numpy()

    # Check if the data has the correct shape (300, 4)
    if data.shape != (300, features_channels):
        raise ValueError(f"Unexpected data shape {data.shape}, expected (300, {features_channels})")

    # Reshape each channel (column) from 300 to (20, 15) using Fortran order (column-major)
    reshaped_data = [np.reshape(data[:, i], (20, 15), order='F') for i in range(features_channels)]

    # Stack the reshaped arrays along the third axis (channels)
    final_array = np.stack(reshaped_data, axis=-1)


    print(f"from excel{final_array.shape}")


    for c in range(features_channels):
        img = final_array[:,:,c]
        df_img = pd.DataFrame(img)
        df_img.to_excel(f'reshape_debug_channel_{c}.xlsx', index=False)
        # plt.imshow(img, cmap='gray')
        # plt.show()

    # Convert the NumPy array to a PyTorch tensor
    print(data)
    data = torch.from_numpy(final_array).unsqueeze(0).float()


    print(f"after converting to tensor {data.size()}")
    data = torch.permute(data, dims=(0,3,1,2))

    # Convert global_features_min and global_features_max to PyTorch tensors
    global_features_min = torch.tensor(global_features_min, dtype=torch.float32).view(1, features_channels, 1, 1)
    global_features_max = torch.tensor(global_features_max, dtype=torch.float32).view(1, features_channels, 1, 1)

    # Normalize the features
    normalized_data = (data - global_features_min) / (global_features_max - global_features_min)

    return normalized_data




# Main

# Test From Excel
input_from_excel = excel_to_np_array(file_path=excel_file_path, sheet_name='Sheet1', global_features_max=global_features_max, global_features_min=global_features_min)


input_curvature = input_from_excel


# Make prediction using model
model = OurModel()
model.load_state_dict(torch.load(model_path))


if compute_certainty:
    # Number of stochastic passes
    N = 10

    model.train()  # Enable dropout for MC Dropout

    # Storage for multiple predictions
    mc_predictions = []

    # Perform N forward passes
    with torch.no_grad():
        for _ in range(N):
            prediction = model(input_curvature)
            mc_predictions.append(prediction)

    # Stack predictions into a single tensor
    mc_predictions = torch.stack(mc_predictions, dim=0)  # Shape: [N, batch_size, channels, height, width]

    # Compute mean prediction
    mean_prediction = mc_predictions.mean(dim=0)  # Shape: [batch_size, channels, height, width]

    # Compute uncertainty (variance or standard deviation)
    variance = mc_predictions.var(dim=0)  # Shape: [batch_size, channels, height, width]
    std_dev = variance.sqrt()  # Standard deviation (optional)

    # Print information
    print(f"Mean Prediction Size: {mean_prediction.size()}")
    print(f"Variance Size: {variance.size()}")

    # Confidence (optional): Convert variance to confidence
    confidence = 1 / (1 + variance)  # Confidence is inversely related to uncertainty
    print(f"Confidence: {confidence.mean()}")


# Make prediction
model.eval()
with torch.no_grad():
    predicted_fiber_orientations = model(input_curvature)
    print(f"Predicted Fiber Orientations datatype: {predicted_fiber_orientations.dtype} Size: {predicted_fiber_orientations.size()}")




# Output the prediction to excel file
predicted_fiber_orientations_denorm = predicted_fiber_orientations.clone()  # Clone to avoid modifying the original tensor
for c in range(labels_channels):
    predicted_fiber_orientations_denorm[:, c, :, :] = predicted_fiber_orientations_denorm[:, c, :, :] * (global_labels_max - global_labels_min) + global_labels_min

predicted_fiber_orientations_denorm_np = predicted_fiber_orientations_denorm.squeeze().numpy()  # Convert to NumPy for plotting

print(f"after numpy{np.shape(predicted_fiber_orientations_denorm_np)}")

export_each_channel_to_excel(prediction_np=predicted_fiber_orientations_denorm_np, base_save_path="predicted_fiber_orientation_inverse")

# Load corresponding labels

gt_fiber_orientation = load_labels_h5_data(new_samples_file_path_labels, labels_main_group, category)
print(f"gt fiber orientation: {gt_fiber_orientation.shape}")
gt_fiber_orientation = gt_fiber_orientation[x:x+1,:, :, :].squeeze()
print(f"After selecting 1 sample {gt_fiber_orientation.shape}")

export_each_channel_to_excel(prediction_np=gt_fiber_orientation, base_save_path="gt_fiber_orientation_inverse")







