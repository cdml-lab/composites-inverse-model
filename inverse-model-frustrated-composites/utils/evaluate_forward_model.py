# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Imports                                         │
# └───────────────────────────────────────────────────────────────────────────┘

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from numpy.ma.extras import average
import matplotlib.pyplot as plt
import math
import nlopt
import os

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Definitions                                     │
# └───────────────────────────────────────────────────────────────────────────┘

file_path_features = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\30-35\30-35_MaxMinCurvature_Labels_Reshaped.h5"
file_path_labels = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\30-35\30-35_MaxMinCurvature_Features_Reshaped.h5"

model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\true_plant_337.pkl"

save_folder = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\plots\Evaluating_Forward_Channels_By_Range"

features_channels = 1
labels_channels = 8

height = 20
width = 15


# Curvature Max and Min
global_labels_max = [10.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
global_labels_min = [-10.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5]

# Curvature Max and Min New
# global_labels_max = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# global_labels_min = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

# All Usefull
# global_labels_max = [9.7, 1.8, 1.0, 1.0, 0.6,
#                     1.0, 1.0, 0.5, 0.5, 0.5,
#                     1.0, 1.0, 0.7, 0.5, 0.5, 1.0, 0.5]
# global_labels_min = [-5.9, -1.2, -1.0, -1.0, -0.6,
#                     -1.0, -1.0, -0.5, -0.5, -0.5,
#                     0.8, 0.7, -0.2, -0.5, -0.6, 0.9, -0.5]

# Normal
# global_labels_max = [0.5, 0.5, 1.0]
# global_labels_min = [-0.5, -0.5, 0.85]


global_features_min = 0
global_features_max = 180

features_main_group = 'Labels'
labels_main_group = 'Features'
category = 'Test'
name='curvautre_true_plant'

show = False

save = True

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                  Model                                    │
# └───────────────────────────────────────────────────────────────────────────┘

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
        self.sigmoid = torch.nn.Sigmoid()

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

        x = self.dropout(x)  # Dropout

        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.relu(x)

        x = self.conv_7(x)
        x = self.batch_norm_7(x)
        x = self.relu(x)

        x = self.conv_8(x)
        x = self.batch_norm_8(x)
        x = self.relu(x)

        x = self.dropout(x)  # Dropout

        x = self.conv_9(x)
        x = self.batch_norm_9(x)
        x = self.relu(x)

        x = self.conv_10(x)

        x = self.batch_norm_10(x)
        x = self.relu(x)

        x = self.conv_11(x)

        # Constrain output values to the label range (0, 1)
        x = torch.sigmoid(x)

        return x

class OurVgg16tn(torch.nn.Module):
    """
    same as vgg16t(up then down) but with no fc layers and flattening, just conv layers
    """
    def __init__(self, dropout=0.3, height = height, width = width):
        super(OurVgg16tn, self).__init__()

        self.conv_1 = torch.nn.Conv2d(in_channels=features_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_6 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_7 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_8 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_9 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_10 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv_11 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_12 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_13 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_14 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_15 = torch.nn.Conv2d(64, labels_channels, kernel_size=3, padding=1)


        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=64)
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=128)
        self.batch_norm_3 = torch.nn.BatchNorm2d(num_features=128)
        self.batch_norm_4 = torch.nn.BatchNorm2d(num_features=256)
        self.batch_norm_5 = torch.nn.BatchNorm2d(num_features=256)
        self.batch_norm_6 = torch.nn.BatchNorm2d(num_features=512)
        self.batch_norm_7 = torch.nn.BatchNorm2d(num_features=512)
        self.batch_norm_8 = torch.nn.BatchNorm2d(num_features=512)
        self.batch_norm_9 = torch.nn.BatchNorm2d(num_features=512)
        self.batch_norm_10 = torch.nn.BatchNorm2d(num_features=256)
        self.batch_norm_11 = torch.nn.BatchNorm2d(num_features=256)
        self.batch_norm_12 = torch.nn.BatchNorm2d(num_features=128)
        self.batch_norm_13 = torch.nn.BatchNorm2d(num_features=128)
        self.batch_norm_14 = torch.nn.BatchNorm2d(num_features=64)
        self.batch_norm_15 = torch.nn.BatchNorm2d(num_features=64)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.fc1 = nn.Linear(512 * height * width, 512)  # Output size adjusted for 1 channel with resolution 20x15
        # self.fc2 = nn.Linear(512, labels_channels * height * width)  # Output size adjusted for 1 channel with resolution 20x15
        self.fc3 = nn.Linear(64 * height * width, labels_channels * height * width)
        self.upsample = torch.nn.Upsample(size=(height, width), mode='nearest')
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.relu(x)
        x = self.dropout(x)


        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_7(x)
        x = self.batch_norm_7(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv_8(x)
        x = self.batch_norm_8(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_9(x)
        x = self.batch_norm_9(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv_10(x)
        x = self.batch_norm_10(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_11(x)
        x = self.batch_norm_11(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv_12(x)
        x = self.batch_norm_12(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_13(x)
        x = self.batch_norm_13(x)
        x = self.relu(x)
        x = self.dropout(x)
        # print(f"after conv13 {x.shape}")

        x = self.conv_14(x)
        x = self.batch_norm_14(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # print(f"after conv14 {x.shape}")


        x = self.conv_15(x)

        x = self.sigmoid(x)



        return x


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Functions                                       │
# └───────────────────────────────────────────────────────────────────────────┘

def load_h5_data(features_file, features_main_group, category, global_features_min, global_features_max):
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

    print(f"Before Normalization: {data.shape}")

    # Convert global min and max to NumPy arrays if they are lists
    global_features_min = np.array(global_features_min)
    global_features_max = np.array(global_features_max)

    # Normalize the features using the global min and max
    normalized_data = (data - global_features_min) / (global_features_max - global_features_min)

    # Convert to PyTorch tensor and add a batch dimension
    feature_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0)

    return feature_tensor

def plot_scatter(predictions, labels_data):

    # Reshape the labels and predictions to 1D arrays for scatter plot
    labels_data = labels_data.reshape(-1).cpu().numpy()  # Ensure labels_data is detached
    predictions = predictions.reshape(-1).detach().cpu().numpy()  # Detach predictions

    # Set the figure size
    plt.figure(figsize=(20, 20))  # Adjust width and height as desired

    # Plot the scatter plot
    plt.scatter(
        labels_data,
        predictions,
        s=1,  # Adjust the size of the dots (e.g., 10 for small dots)
        c='teal',  # Set a uniform color (e.g., 'blue') or pass an array for varying colors
        alpha=0.1  # Adjust transparency (e.g., 0.7 for semi-transparent dots)
    )
    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot: True Labels vs Predictions')
    plt.show()


def plot_filtered_scatter_per_channel(true_labels, predictions, value_range, channel_number,save_folder):
    """
    Plots a scatter plot of predictions against true labels for a specific channel, filtering by a value range.

    Args:
        true_labels (np.ndarray or torch.Tensor): The true labels.
        predictions (np.ndarray or torch.Tensor): The model predictions.
        value_range (tuple): A tuple (min_value, max_value) to filter true labels.
        channel_number (int): The channel number to select from the 3D tensors (e.g., if data is [batch, channel, height, width]).
    """
    # Ensure the data is in NumPy format (detach if necessary)
    true_labels = true_labels.detach().cpu().numpy() if isinstance(true_labels, torch.Tensor) else true_labels
    predictions = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions

    # Select the specific channel for both true labels and predictions
    true_labels_channel = true_labels[:, channel_number, :, :].reshape(-1)  # Flatten the channel to 1D
    predictions_channel = predictions[:, channel_number, :, :].reshape(-1)  # Flatten the channel to 1D

    # Filter true labels based on the specified range
    min_value, max_value = value_range[0], value_range[1]
    mask = (true_labels_channel >= min_value) & (true_labels_channel <= max_value)

    # Apply the mask to get filtered true labels and corresponding predictions
    filtered_labels = true_labels_channel[mask]
    filtered_predictions = predictions_channel[mask]

    # Set the figure size
    plt.figure(figsize=(30, 30))  # Adjust width and height as desired

    # Plot the filtered scatter plot for the specified channel
    plt.clf()
    plt.scatter(
        filtered_labels,
        filtered_predictions,
        s=1,  # Adjust the size of the dots (e.g., 10 for small dots)
        c='blue',  # Set a uniform color (e.g., 'blue') or pass an array for varying colors
        alpha=0.5  # Adjust transparency (e.g., 0.7 for semi-transparent dots)
    )
    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    plt.title(f'Scatter Plot (True Labels vs Predictions) for Channel {channel_number} and Range {value_range}')

    # Set the axis limits to [0, 1] to ensure consistency
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if save:
        # Define the file path and save the plot
        save_file_path = os.path.join(save_folder,
                                      f'{name}_channel_{channel_number}_range_{value_range[0]}_{value_range[1]}.png')

        plt.savefig(save_file_path)
        print(f"Saved to {save_file_path}")

    if show:
        plt.show()


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                 Main                                      │
# └───────────────────────────────────────────────────────────────────────────┘

# Load Features
features_data = load_h5_data(
    features_file=file_path_features,
    features_main_group=features_main_group,
    global_features_min=global_features_min,
    global_features_max=global_features_max,
    category=category
)


labels_data = load_h5_data(
    features_file=file_path_labels,
    features_main_group=labels_main_group,
    global_features_min=global_labels_min,
    global_features_max=global_labels_max,
    category=category
)

# Organize shape
labels_data = labels_data.squeeze().permute(0,3,1,2)
features_data = features_data.permute(1,0,2,3)

print(f"Features shape: {features_data.shape}")
print(f"Labels shape: {labels_data.shape}")

# Load and evaluate the model
# model = OurModel()
model = OurVgg16tn()
model.load_state_dict(torch.load(model_path))
model.eval()


# Predicting using the model
predictions = model(features_data)

# plot scatter
plot_scatter(predictions, labels_data)

value_range = [0.0, 1.000]  # Example range, you can adjust this as needed

# Define the ranges and channels
channels = range(labels_channels)  # Assuming 8 channels (0 to 7)
range_start = 0.0  # Starting value for the range
range_end = 1.0    # End value for the range
range_step = 0.1   # Step size for the range


# # Loop through all channels and ranges
# for channel_number in channels:
#     for start in np.arange(range_start, range_end, range_step):
#         value_range = [start, start + range_step]
#         plot_filtered_scatter_per_channel(labels_data, predictions, value_range, channel_number, save_folder)

# Loop through all channels and ranges
for channel_number in channels:
    plot_filtered_scatter_per_channel(labels_data, predictions, value_range, channel_number, save_folder)