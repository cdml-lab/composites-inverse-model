# imports
import matplotlib

matplotlib.use('Agg')

import datetime
import h5py
import os
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import pandas as pd
import wandb


seed = 42  # Set the seed for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

# Set variables

## Set dataset name
og_dataset_name = "17-24"
dataset_name = "17-24_All"

features_channels = 1
labels_channels = 3

x_size=15
y_size=20

# PAY ATTENTION: since this is a forward models the files are flipped and the labels file will be the original features
# file! and the same foe feature will be the original labels file, meant for in inverse model.
features_file = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + og_dataset_name + '/' + dataset_name + '_Labels_Reshaped.h5'
labels_file = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + og_dataset_name + '/' + dataset_name + '_Features_Reshaped.h5'

# OPTIONAL!!! For testing variations of labels(original features
# Make sure to adapt labels channels
labels_file = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\17-24\Combined_Features\Location_Features_Reshaped.h5"
labels_channels = 3
dataset_name = "17-24_Location_Features"


# Define the path and name for saving the model
current_date = datetime.datetime.now().strftime("%Y%m%d")
model_name = f"{dataset_name}_{current_date}.pkl"

save_model_path = 'C:/Gal_Msc/Ipublic-repo/inverse-model-frustrated-composites/saved_model/Forward/' + model_name
load_model_path = 'C:/Gal_Msc/Ipublic-repo/inverse-model-frustrated-composites/saved_model/Forward/' + model_name

train = 'yes'  #If you want to load previously trained model for evaluation - set to 'load' and correct the load_model_path
train_arch = 'no'
model_type = 'ourmodel'  # 'arch' for testing architectures
is_random = 'no'
separate_labels = 'yes'
add_curvature = False
add_curvature_channels = 0




import yaml




# Function to read HDF5 data (maybe this is not needed)
def read_hdf5_data(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as h5file:
        data = {}
        for category in h5file.keys():
            main_group = h5file[category]
            data[category] = {}
            for folder_suffix in main_group:
                sub_group = main_group[folder_suffix]
                data[category][folder_suffix] = {}
                for dataset_name in sub_group:
                    dataset = sub_group[dataset_name]
                    if dataset.size == 0:
                        print(f"Skipping empty dataset {dataset_name} in {folder_suffix}")
                        continue
                    data[category][folder_suffix][dataset_name] = dataset[:]
    return data

# Function to print if file exists
def file_exists(file):
    if os.path.isfile(file):
        print(f"file{file} exists")
    else:
        print(f"file{file} does not exist!")

def calculate_global_min_max(features_file, labels_file, feature_main_group, label_main_group):
    with h5py.File(labels_file, 'r') as f:
        label_data = f[label_main_group]['Train']
        global_label_min = np.inf * np.ones(label_data[list(label_data.keys())[0]].shape[2])
        global_label_max = -np.inf * np.ones(label_data[list(label_data.keys())[0]].shape[2])

        for key in label_data.keys():
            data = label_data[key][:]
            for i in range(data.shape[2]):
                global_label_min[i] = min(global_label_min[i], data[:, :, i].min())
                global_label_max[i] = max(global_label_max[i], data[:, :, i].max())

    with h5py.File(features_file, 'r') as f:
        feature_data = f[feature_main_group]['Train']
        global_feature_min = np.inf
        global_feature_max = -np.inf

        for key in feature_data.keys():
            data = feature_data[key][:]
            global_feature_min = min(global_feature_min, data.min())
            global_feature_max = max(global_feature_max, data.max())

    # Print global min and max values for debugging
    print(f"Global Feature Min: {global_feature_min}")
    print(f"Global Feature Max: {global_feature_max}")
    for i in range(len(global_label_min)):
        print(f"Global Label Min for channel {i}: {global_label_min[i]}")
        print(f"Global Label Max for channel {i}: {global_label_max[i]}")

    return global_feature_min, global_feature_max, global_label_min, global_label_max
def calculate_reference_curvature_vector_3d(epsilon, t, top_angle_degrees, bottom_angle_degrees):
    """
    Calculate the reference curvature vector for a given point in a sheet where
    the top and bottom layers are oriented at different angles in 3D space.

    Parameters:
    -----------
    epsilon : float
        The inelastic strain of the material.
    t : float
        The thickness of the sheet.
    top_angle_degrees : float
        The orientation angle (in degrees) of the fibers in the top layer.
    bottom_angle_degrees : float
        The orientation angle (in degrees) of the fibers in the bottom layer.

    Returns:
    --------
    tuple of floats
        The reference curvature vector components (k_x, k_y, k_z) in the x-, y-, and z-directions.
    """
    # Convert angles to radians for trigonometric functions
    top_angle_radians = np.radians(top_angle_degrees)
    bottom_angle_radians = np.radians(bottom_angle_degrees)

    # Calculate the principal curvature magnitudes
    # Here we assume a basic relation: k = epsilon / (t * sqrt(2)) adjusted for the angles
    # Magnitudes are modified based on the difference in the angle between top and bottom layers
    k_magnitude_top = epsilon / (t * np.sqrt(2))
    k_magnitude_bottom = epsilon / (t * np.sqrt(2))

    # Compute the curvature components in the x-, y-, and z-directions
    k_x = k_magnitude_top * np.cos(top_angle_radians) + k_magnitude_bottom * np.cos(bottom_angle_radians)
    k_y = k_magnitude_top * np.sin(top_angle_radians) + k_magnitude_bottom * np.sin(bottom_angle_radians)

    # The z-component (out-of-plane curvature)
    # Assuming a simple out-of-plane curvature proportional to the sine of the angular difference
    angular_difference = bottom_angle_degrees - top_angle_degrees
    k_z = k_magnitude_top * np.sin(np.radians(angular_difference))  # A proxy for out-of-plane deformation

    return k_x, k_y, k_z

# Custom Class of Data
class FolderHDF5Data(Dataset):
    def __init__(self, features_file, labels_file, feature_main_group, label_main_group, category, global_feature_min,
                 global_feature_max, global_label_min, global_label_max):
        """
        Initialize the dataset with the paths to the features and labels HDF5 files,
        the main groups ('Features' and 'Labels'), and the category ('Train' or 'Test').

        Args:
            features_file (str): Path to the features HDF5 file.
            labels_file (str): Path to the labels HDF5 file.
            feature_main_group (str): Main group within the features HDF5 file ('Features').
            label_main_group (str): Main group within the labels HDF5 file ('Labels').
            category (str): Subgroup within the main group ('Train' or 'Test').
            global_feature_min (float): Global minimum value for feature normalization.
            global_feature_max (float): Global maximum value for feature normalization.
            global_label_min (float): Global minimum value for label normalization.
            global_label_max (float): Global maximum value for label normalization.
        """
        self.features_file = features_file
        self.labels_file = labels_file
        self.feature_main_group = feature_main_group
        self.label_main_group = label_main_group
        self.category = category
        self.global_feature_min = global_feature_min
        self.global_feature_max = global_feature_max
        self.global_label_min = global_label_min
        self.global_label_max = global_label_max
        self.filenames = self._get_filenames()

    def _get_filenames(self):
        """
        Retrieve the dataset names (keys) within the specified main group and category.

        Returns:
            list: List of dataset names.
        """
        with h5py.File(self.features_file, 'r') as f:
            return list(f[self.feature_main_group][self.category].keys())

    def __len__(self):
        """
        Return the number of datasets within the specified main group and category.

        Returns:
            int: Number of datasets.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        with h5py.File(self.features_file, 'r') as f_features, h5py.File(self.labels_file, 'r') as f_labels:
            dataset_name = self.filenames[idx]
            feature = f_features[self.feature_main_group][self.category][dataset_name][()]
            label = f_labels[self.label_main_group][self.category][dataset_name][()]

            if feature.size == 0 or label.size == 0:
                return None

            if add_curvature == True:
                # Extract necessary parameters for curvature calculation from the features (fiber angles)
                top_angle_degrees = feature[:, :, 0]  # Assuming top angles are stored in feature channel 0
                bottom_angle_degrees = (top_angle_degrees + 90) % 180  # Ensuring the result is within 0 to 180 degrees

                epsilon = 0.07  # Example value; change as needed
                t = 0.2  # Example value for thickness; change as needed

                # Calculate the curvature vector for each point
                curvature_features = np.zeros((feature.shape[0], feature.shape[1], 3))  # For k_x, k_y, k_z
                for i in range(feature.shape[0]):
                    for j in range(feature.shape[1]):
                        curvature_features[i, j, :] = calculate_reference_curvature_vector_3d(
                            epsilon, t, top_angle_degrees[i, j], bottom_angle_degrees[i, j]
                        )

                # Concatenate the curvature features with the existing features
                feature = np.concatenate((feature, curvature_features), axis=2)

            # Transform the feature and label tensors
            feature_tensor, label_tensor = data_transform(feature, label, self.global_feature_min,
                                                          self.global_feature_max, self.global_label_min,
                                                          self.global_label_max)

            return feature_tensor, label_tensor


# Transform. doesn't currently resize.
import torch

def data_transform(feature, label, global_feature_min, global_feature_max, global_label_min, global_label_max):
    """
    Transform the feature and label data into the required format for the model.

    Args:
        feature (np.ndarray): The feature data array with shape (height, width, channels).
        label (np.ndarray): The label data array with shape (height, width, channels).
        global_feature_min (float or list of floats): Global minimum value(s) for feature normalization.
        global_feature_max (float or list of floats): Global maximum value(s) for feature normalization.
        global_label_min (float or list of floats): Global minimum value(s) for label normalization.
        global_label_max (float or list of floats): Global maximum value(s) for label normalization.

    Returns:
        tuple: Transformed feature and label tensors with shape:
               - feature_tensor: (channels, height, width)
               - label_tensor: (channels, height, width)
    """

    # Convert feature data to tensor
    feature_tensor = torch.tensor(feature, dtype=torch.float32)

    # Normalize features
    if isinstance(global_feature_min, (float, int)):
        feature_tensor = (feature_tensor - global_feature_min) / (global_feature_max - global_feature_min)
    else:
        global_feature_min_tensor = torch.tensor(global_feature_min, dtype=torch.float32)
        global_feature_max_tensor = torch.tensor(global_feature_max, dtype=torch.float32)
        for c in range(feature_tensor.shape[2]):
            feature_tensor[:, :, c] = (feature_tensor[:, :, c] - global_feature_min_tensor[c]) / (
                    global_feature_max_tensor[c] - global_feature_min_tensor[c])

    # Reorder dimensions: from (height, width, channels) to (channels, height, width)
    feature_tensor = feature_tensor.permute(2, 0, 1).float()

    # Convert label data to tensor
    label_tensor = torch.tensor(label, dtype=torch.float32)

    # Normalize labels
    if isinstance(global_label_min, (float, int)):
        label_tensor = (label_tensor - global_label_min) / (global_label_max - global_label_min)
    else:
        global_label_min_tensor = torch.tensor(global_label_min, dtype=torch.float32)
        global_label_max_tensor = torch.tensor(global_label_max, dtype=torch.float32)

        # Iterate over channels, but make sure to stay within the range of the global_label_min/max tensor size
        num_channels = min(label_tensor.shape[0], global_label_min_tensor.size(0))

        for c in range(num_channels):  # Safeguard against out-of-bound indexing
            label_tensor[c, :, :] = (label_tensor[c, :, :] - global_label_min_tensor[c]) / (
                    global_label_max_tensor[c] - global_label_min_tensor[c])


    # Reorder dimensions: from (height, width, channels) to (channels, height, width)
    label_tensor = label_tensor.permute(2, 0, 1).float()

    return feature_tensor, label_tensor


import torch


class OurModel(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(OurModel, self).__init__()

        self.conv_1 = torch.nn.Conv2d(in_channels=features_channels + add_curvature_channels, out_channels=32, kernel_size=3, padding=1)
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
        # Don't apply ReLU if this is a regression problem, so no activation on the final layer

        # Constrain output values to the label range (0, 1)
        x = torch.sigmoid(x)
        return x


# Train function. set the epochs and patience here.
import torch.optim as optim

import torch
import time


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200, patience=12,
                clip_value=1.0):
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    training_log = []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Gradient Monitoring
            # total_norm = 0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # print(f"Batch Gradient L2 Norm: {total_norm:.4f}")

            # Update Weights
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)  # Step the scheduler

        end_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs} | Time: {end_time - start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        training_log.append((epoch + 1, train_loss, val_loss))

        # Early stopping mechanism
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'forward_best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {patience} epochs of no improvement.')
            early_stop = True
            break

    if early_stop:
        print("Loading best model from checkpoint...")
        model.load_state_dict(torch.load('forward_best_model.pth'))

    return model, training_log


def evaluate_model(model, val_loader, criterion, plot_dir):
    print("evaluating model...")
    model.eval()
    model.to(device)  # Ensure model is on GPU
    val_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Convert tensors to CPU before converting to numpy
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_loss /= len(val_loader)

    # Use NumPy to concatenate arrays
    errors = np.concatenate(all_predictions, axis=0).flatten() - np.concatenate(all_labels, axis=0).flatten()

    plot_error_histogram(errors, plot_dir=plot_dir)

    print(f'Validation Loss: {val_loss:.4f}')

    # Use NumPy to concatenate arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Debug: Check the shape and range before denormalization
    print(f"Shape of Predictions before denormalization: {all_predictions.shape}")
    print(f"Shape of Labels before denormalization: {all_labels.shape}")
    print(f"Predictions (min, max) before denormalization: {all_predictions.min()}, {all_predictions.max()}")
    print(f"Labels (min, max) before denormalization: {all_labels.min()}, {all_labels.max()}")

    # === Denormalize the predictions and labels based on the normalization method ===
    if wandb.config.normalization == "global":
        # Global denormalization
        all_predictions = all_predictions * (
                    global_labels_max_all_channels - global_labels_min_all_channels) + global_labels_min_all_channels
        all_labels = all_labels * (
                    global_labels_max_all_channels - global_labels_min_all_channels) + global_labels_min_all_channels
    else:
        # Per-channel denormalization, considering that predictions are clamped
        for c in range(labels_channels):
            # Denormalize the predictions only if they were not clamped
            if all_predictions[:, :, :, c].min() >= 0 and all_predictions[:, :, :, c].max() <= 1:
                all_predictions[:, :, :, c] = all_predictions[:, :, :, c] * (
                            global_label_max[c] - global_label_min[c]) + global_label_min[c]
            all_labels[:, :, :, c] = all_labels[:, :, :, c] * (global_label_max[c] - global_label_min[c]) + \
                                     global_label_min[c]

    # Debug: Check the min and max values after denormalization
    print(f"Predictions (min, max) after denormalization: {all_predictions.min()}, {all_predictions.max()}")
    print(f"Labels (min, max) after denormalization: {all_labels.min()}, {all_labels.max()}")

    # Now flatten the arrays for scatter plot
    all_predictions_flat = all_predictions.flatten()
    all_labels_flat = all_labels.flatten()

    # Now plot the scatter plot with denormalized values
    plot_scatter_plot(all_labels_flat, all_predictions_flat, save_path=os.path.join(plot_dir, 'scatter_plot.png'))

    return val_loss, all_labels_flat, all_predictions_flat


# Testing different loss functions
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Flatten the tensors to (batch_size, -1)
        y_pred = y_pred.view(y_pred.size(0), -1)
        y_true = y_true.view(y_true.size(0), -1)

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(y_pred, y_true, dim=1)

        # Compute the loss as 1 - cosine similarity
        loss = 1 - cos_sim.mean()

        return loss

class MeanErrorLoss(nn.Module):
    def __init__(self):
        super(MeanErrorLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(y_pred - y_true)

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, input, target):
        abs_diff = torch.abs(input - target)
        loss = torch.where(abs_diff < self.delta,
                           0.5 * abs_diff ** 2,
                           self.delta * (abs_diff - 0.5 * self.delta))
        return loss.mean()

class CauchyLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, input, target):
        x = torch.abs(input - target) / self.delta
        loss = self.delta * torch.log(1 + x ** 2)
        return loss.mean()

class TukeyBiweightLoss(nn.Module):
    def __init__(self, c=4.685):
        super().__init__()
        self.c = c

    def forward(self, input, target):
        x = torch.abs(input - target) / self.c
        x = torch.clamp(x, min=0, max=1)
        loss = self.c ** 2 * (1 - (1 - x ** 2) ** 3) / 6
        return loss.mean()

# Visualization Functions
def plot_quiver(actual, predicted, sample_index=1, plot_dir="plots"):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    actual_magnitude = actual[0, :, :]
    actual_x = actual[1, :, :]
    actual_y = actual[2, :, :]
    actual_x *= actual_magnitude
    actual_y *= actual_magnitude

    predicted_magnitude = predicted[0, :, :]
    predicted_x = predicted[1, :, :]
    predicted_y = predicted[2, :, :]
    predicted_x *= predicted_magnitude
    predicted_y *= predicted_magnitude

    y, x = np.mgrid[0:actual_x.shape[0], 0:actual_x.shape[1]]

    axs[0].quiver(x, y, actual_x, actual_y)
    axs[0].set_title('Actual Direction Vectors')
    axs[0].invert_yaxis()

    axs[1].quiver(x, y, predicted_x, predicted_y)
    axs[1].set_title('Predicted Direction Vectors')
    axs[1].invert_yaxis()

    plt.tight_layout()
    img_path = os.path.join(plot_dir, f"quiver_plot_{sample_index}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Saved quiver plot for sample {sample_index} to {img_path}")

import matplotlib.pyplot as plt
import torch
import random

def show_random_samples_curvature(model, dataset, num_samples=6, is_random='yes', save_path="random_samples.png"):
    model.eval()

    for i in range(num_samples):
        if is_random == 'yes':
            idx = random.randint(0, len(dataset) - 1)
        else:
            idx = i
        feature_tensor, label_tensor = dataset[idx]

        # Generate prediction
        with torch.no_grad():
            prediction_tensor = model(feature_tensor.unsqueeze(0).to(device)).squeeze(0)

        # Determine the number of subplots: one for each feature, one for each label, and one for each prediction channel
        num_features = feature_tensor.shape[0]
        num_labels = label_tensor.shape[0]
        num_predictions = prediction_tensor.shape[0]
        total_subplots = num_features + num_labels + num_predictions

        # Create a figure with subplots
        fig, axs = plt.subplots(total_subplots, 1, figsize=(10, 5 * total_subplots))
        fig.suptitle(f'Sample {i + 1}', fontsize=20)

        # Display each label channel
        for l in range(num_labels):
            label_img = label_tensor[l, :, :].cpu().numpy()
            label_img = (label_img - label_img.min()) / (label_img.max() - label_img.min())
            axs[l].imshow(label_img, cmap='plasma')
            axs[l].axis('off')
            axs[l].set_title(f'Ground Truth Label Channel {l + 1}')

        # Plot each feature channel separately
        for c in range(num_features):
            feature_img = feature_tensor[c, :, :].cpu().numpy()
            feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
            axs[num_labels + c].imshow(feature_img, cmap='viridis')
            axs[num_labels + c].axis('off')
            axs[num_labels + c].set_title(f'Feature Channel {c + 1}')

        # Display each prediction channel
        for p in range(num_predictions):
            prediction_img = prediction_tensor[p, :, :].cpu().numpy()
            prediction_img = (prediction_img - prediction_img.min()) / (prediction_img.max() - prediction_img.min())
            axs[num_labels + num_features + p].imshow(prediction_img, cmap='plasma')
            axs[num_labels + num_features + p].axis('off')
            axs[num_labels + num_features + p].set_title(f'Prediction Channel {p + 1}')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.3)  # Add space between rows

        # Save the figure as an image file
        sample_save_path = save_path.replace(".png", f"_{i + 1}.png")
        plt.savefig(sample_save_path)
        plt.close()
        print(f"Sample {i + 1} saved to {sample_save_path}")
        wandb.log({f"random_samples{i+1}": wandb.Image(sample_save_path)})

def show_random_samples(model, dataset, num_samples=6, is_random='yes', save_path="random_samples.png"):
    model.eval()

    for i in range(num_samples):
        if is_random == 'yes':
            idx = random.randint(0, len(dataset) - 1)
        else:
            idx = i
        feature_tensor, label_tensor = dataset[idx]

        # Generate prediction
        with torch.no_grad():
            prediction_tensor = model(feature_tensor.unsqueeze(0).to(device)).squeeze(0)

        # Determine the number of subplots: one for each feature, one for each label, and one for each prediction channel
        num_features = feature_tensor.shape[0]
        num_labels = label_tensor.shape[0]
        num_predictions = prediction_tensor.shape[0]
        curvature_start_idx = num_features - 3  # Assuming the last 3 channels are curvature components
        original_num_features = num_features - 3  # Original feature channels count
        total_subplots = num_features + num_labels + num_predictions

        # Create a figure with subplots
        fig, axs = plt.subplots(total_subplots, 1, figsize=(10, 5 * total_subplots))
        fig.suptitle(f'Sample {i + 1}', fontsize=20)

        # Display each label channel
        for l in range(num_labels):
            label_img = label_tensor[l, :, :].cpu().numpy()
            label_img = (label_img - label_img.min()) / (label_img.max() - label_img.min())
            axs[l].imshow(label_img, cmap='plasma')
            axs[l].axis('off')
            axs[l].set_title(f'Ground Truth Label Channel {l + 1}')

        # Plot each feature channel separately
        for c in range(original_num_features):
            feature_img = feature_tensor[c, :, :].cpu().numpy()
            feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
            axs[num_labels + c].imshow(feature_img, cmap='viridis')
            axs[num_labels + c].axis('off')
            axs[num_labels + c].set_title(f'Feature Channel {c + 1}')

        # Plot curvature features (k_x, k_y, k_z)
        curvature_titles = ['Curvature k_x', 'Curvature k_y', 'Curvature k_z']
        for j, title in enumerate(curvature_titles):
            curvature_img = feature_tensor[curvature_start_idx + j, :, :].cpu().numpy()
            curvature_img = (curvature_img - curvature_img.min()) / (curvature_img.max() - curvature_img.min())  # Normalize
            axs[num_labels + original_num_features + j].imshow(curvature_img, cmap='plasma')
            axs[num_labels + original_num_features + j].axis('off')
            axs[num_labels + original_num_features + j].set_title(title)

        # Display each prediction channel
        for p in range(num_predictions):
            prediction_img = prediction_tensor[p, :, :].cpu().numpy()
            prediction_img = (prediction_img - prediction_img.min()) / (prediction_img.max() - prediction_img.min())
            axs[num_labels + num_features + p].imshow(prediction_img, cmap='plasma')
            axs[num_labels + num_features + p].axis('off')
            axs[num_labels + num_features + p].set_title(f'Prediction Channel {p + 1}')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.3)  # Add space between rows

        # Save the figure as an image file
        sample_save_path = save_path.replace(".png", f"_{i + 1}.png")
        plt.savefig(sample_save_path)
        plt.close()
        print(f"Sample {i + 1} saved to {sample_save_path}")
        wandb.log({f"random_samples{i+1}": wandb.Image(sample_save_path)})

def plot_samples_with_annotations(loader_type, data_loader, num_samples=6, plot_dir="plots"):
    """
    Iterate through the data_loader and plot samples with RGB values annotated for every 5x5 pixel block.

    Args:
        data_loader (DataLoader): DataLoader object for either train or test data.
        num_samples (int): Number of random samples to display.
        plot_dir (str): Directory to save the plot images.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for i, (features, labels) in enumerate(data_loader):
        if i >= num_samples:
            break

        # Get the first sample from the batch
        feature = features[0]
        label = labels[0]

        # Convert tensors to numpy arrays
        feature_img = feature.permute(1, 2, 0).cpu().numpy()

        # Normalize for visualization
        # feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())

        fig, axs = plt.subplots(1, labels_channels + 1, figsize=(30, 7))  # Create enough subplots

        fig.suptitle(f'Sample {i + 1} - Features and Labels with Annotations', fontsize=16)

        axs[0].imshow(feature_img)
        axs[0].axis('off')
        axs[0].set_title('Features')

        for c in range(labels_channels):
            label_img = label[c, :, :].cpu().numpy()
            # label_img = (label_img - label_img.min()) / (label_img.max() - label_img.min())
            axs[c + 1].imshow(label_img)
            axs[c + 1].axis('off')
            axs[c + 1].set_title(f'Label Channel {c + 1}')

            # Annotate each 5x5 pixel block for labels
            for y in range(0, label_img.shape[0], 5):
                for x in range(0, label_img.shape[1], 5):
                    label_text = f"{label_img[y, x]:.2f}"
                    axs[c + 1].text(x, y, label_text, fontsize=8, color='white',
                                    bbox=dict(facecolor='black', alpha=0.5))

        img_path = os.path.join(plot_dir, f"debug_sample_forward_{loader_type}_{i + 1}.png")
        plt.savefig(img_path)
        plt.close()

        print(f"Saved debug plot for sample {i + 1} to {img_path}")

def plot_error_histogram(errors, plot_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='b')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')
    img_path = os.path.join(plot_dir, "error_histogram.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Saved error histogram to {img_path}")

def plot_heatmaps(actual, predicted, sample_index=1, plot_dir="plots"):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    num_channels, height, width = actual.shape

    fig, axs = plt.subplots(2, num_channels, figsize=(15, 10))

    for channel in range(num_channels):
        axs[0, channel].imshow(actual[channel, :, :], cmap='viridis')
        axs[0, channel].set_title(f'Actual - Channel {channel}')
        axs[1, channel].imshow(predicted[channel, :, :], cmap='viridis')
        axs[1, channel].set_title(f'Predicted - Channel {channel}')

    plt.tight_layout()
    img_path = os.path.join(plot_dir, f"heatmap_sample_{sample_index}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Saved heatmap plot for sample {sample_index} to {img_path}")

def plot_scatter_plot(labels, predictions, save_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(labels, predictions, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_residuals(predictions, labels, save_path):
    """
    Plots the residuals against the predicted values.

    Args:
        predictions: A numpy array or list of predicted values.
        labels: A numpy array or list of true labels.
        save_path: The path to save the plot.
    """
    # Convert predictions and labels to tensors and flatten
    all_predictions_flat = torch.tensor(predictions).view(-1).numpy()  # Convert to tensor, flatten, and back to numpy
    all_labels_flat = torch.tensor(labels).view(-1).numpy()  # Convert to tensor, flatten, and back to numpy

    residuals = all_predictions_flat - all_labels_flat

    plt.figure(figsize=(8, 6))
    plt.scatter(all_predictions_flat, residuals, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved residuals plot to {save_path}")

def plot_training_log(training_log, plot_path):
    """
    Plot the training and validation loss over epochs.

    Args:
        training_log (list of tuples): Each tuple contains (epoch, train_loss, val_loss).
        plot_dir (str): Directory to save the plot image.
    """

    epochs, train_losses, val_losses = zip(*training_log)

    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss over Epochs\nFinal Loss - Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f}')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Training log plot saved to {plot_path}")


import numpy as np




# Test Architectures:
def create_model(architecture, label_channels, dropout_rate=0.2):
    layers = []
    in_channels = features_channels  # Assuming input has 1 channel
    for i, out_channels in enumerate(architecture):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        if i % 3 == 0:  # Apply dropout every 3 layers
            layers.append(nn.Dropout(p=dropout_rate))
        in_channels = out_channels
    layers.append(nn.Conv2d(in_channels, label_channels, kernel_size=3, padding=1))  # Final layer to match output channels
    return nn.Sequential(*layers)

######### Main Code

#CUDA

if __name__ == "__main__":


    # CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU")

    print(torch.__version__)
    print(torch.version.cuda)

    patience = 8

    #Use the full configuration when not sweeping

    # Initialize WandB project
    wandb.init(project="forward_model", config={
        "dataset": "Location_Features",
        "learning_rate": 0.002,
        "epochs": 400,
        "batch_size": 64,
        "architecture": "OurModel",
        "optimizer": "Adam",
        "loss_function": "L1",
        "normalization": "per-channel",
        "dataset_name": dataset_name,
        "features_channels": features_channels,
        "labels_channels": labels_channels,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.1,
        "patience": 12,
        "dropout": 0.3
    })

    wandb.init(project="forward_model")


    # Calculate global min and max values for normalization
    global_feature_min, global_feature_max, global_label_min, global_label_max = calculate_global_min_max(features_file,
                                                                                                          labels_file,
                                                                                                          'Labels',

                                                                                                          'Features')
    # Get global values for all labels together
    global_labels_min_all_channels = min(global_label_min)
    global_labels_max_all_channels = max(global_label_max)

    # Initialize dataset and data loaders
    # PAY ATTENTION: the labels and feature files are flipped on purpose! because this is a forward model and the files are bult for inverse

    # This is also where you define global-global OR per-channel-global normalization
    # for per-channel globalisation choose "global_labels_min" (and similar) variables and for completely global choose "global_labels_min_all_channels" (and similar)

    # since features are only 1 channel it doesn't matter.


    if wandb.config.normalization == "global":
        train_dataset = FolderHDF5Data(features_file, labels_file, 'Labels', 'Features', 'Train',
                                       global_feature_min, global_feature_max, global_labels_min_all_channels,
                                       global_labels_max_all_channels)
        print("Normalization: Global")
        val_dataset = FolderHDF5Data(features_file, labels_file, 'Labels', 'Features', 'Test',
                                     global_feature_min, global_feature_max, global_labels_min_all_channels,
                                     global_labels_max_all_channels)
    else:
        train_dataset = FolderHDF5Data(features_file, labels_file, 'Labels', 'Features', 'Train',
                                       global_feature_min, global_feature_max, global_label_min,
                                       global_label_max)
        val_dataset = FolderHDF5Data(features_file, labels_file, 'Labels', 'Features', 'Test',
                                     global_feature_min, global_feature_max, global_label_min,
                                     global_label_max)
        print("Normalization: Per-Channel")

    # Initialize dataset and data loaders
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)



    # See samples(for debugging)
    plot_samples_with_annotations('train', train_loader, num_samples=4, plot_dir="plots")


    ### Test Architectures
    architectures = [
        [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 512]
    ]

    results = []


    if wandb.config.loss_function == 'CosineSimilarity':
        criterion = CosineSimilarityLoss()
    elif wandb.config.loss_function == 'HuberLoss':
        criterion = HuberLoss()
    elif wandb.config.loss_function == 'TukeyBiweightLoss':
        criterion = TukeyBiweightLoss()
    elif wandb.config.loss_function == 'CauchyLoss':
        criterion = CauchyLoss()
    elif wandb.config.loss_function == 'L2':
        criterion = nn.MSELoss()
    elif wandb.config.loss_function == 'L1':
        criterion = nn.L1Loss()
    else:
        print(f"Unknown loss function: {wandb.config.loss_function}, using cosine similarity")
        criterion = CosineSimilarityLoss()

    # Initialize model
    if model_type == 'ourmodel':
        print(f"model selected {model_type}")
        # model = OurModel().to(device)
        model = OurModel(dropout= wandb.config.dropout).to(device)
    elif model_type == 'arch':
        for idx, architecture in enumerate(architectures):
            print(f"Training architecture {idx + 1}: {architecture}")
            model = create_model(architecture, labels_channels).to(device)
            optimizer = optim.Ranger(model.parameters(), lr=0.003, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)
            criterion = criterion
            model_save_path = f"saved_model_{idx + 1}.pth"

            if train_arch == 'yes':
                # Train the model
                trained_model, training_log = train_model(model, train_loader, val_loader, criterion, optimizer,
                                                          scheduler)

                # Save the model
                torch.save(trained_model.state_dict(), model_save_path)
            else:
                print(f"loading model {model_save_path}")
                model.load_state_dict(torch.load(model_save_path))
                model.eval()  # Set the model to evaluation mode
                trained_model = model
                trained_model.to(device)

            # Evaluate the model
            val_loss, all_labels_flat, all_predictions_flat = evaluate_model(trained_model, val_loader, criterion,
                                                                             plot_dir="plots")
            # Save scatter plot and random samples
            scatter_plot_path = f"forward_scatter_plot_{idx + 1}.png"
            random_samples_path = f"random_samples_{idx + 1}.png"
            residuals_path = f"residuals_{idx + 1}.png"
            training_log_path=f'training_log_{idx + 1}.png'

            try:
                plot_training_log(training_log, training_log_path)
            except:
                print("could not print training log")
            try:
                plot_scatter_plot(all_labels_flat, all_predictions_flat, save_path=scatter_plot_path)
            except:
                print("could not plot scatter plot")
            try:
                show_random_samples(trained_model, val_dataset, num_samples=6, save_path=random_samples_path)
            except:
                print("could not plot random samples")
            try:
                plot_residuals(all_predictions_flat, all_labels_flat, save_path=residuals_path)
            except:
                print("could not print residuals")

            # Record results
            results.append({
                "architecture": architecture,
                "val_loss": val_loss,
                "model_save_path": model_save_path,
                "scatter_plot_path": scatter_plot_path,
                "random_samples_path": random_samples_path,
                "training_log_path": training_log_path
            })

            results_df = pd.DataFrame(results)
            results_df.to_excel("model_evaluation_results.xlsx", index=False)
            print(results_df)

    # Set optimizer
    # optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=wandb.config.scheduler_factor, patience=patience)

    # Run the training
    if train == 'yes':
        print("Training Model")


        config = wandb.config

        trained_model, training_log = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=wandb.config.epochs, patience=wandb.config.patience)


        # Save trained model
        torch.save(trained_model.state_dict(), save_model_path)
        print("Model saved to..." + save_model_path)
        wandb.save(save_model_path)

        # Log training progress
        for epoch, train_loss, val_loss in training_log:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            })

    elif train == 'load':
        print("Loading Pre-trained Model... " + load_model_path)
        model.load_state_dict(torch.load(load_model_path))
        model.eval()  # Set the model to evaluation mode
        trained_model = model
    else:
        print('not loading or training')


    # Evaluate the model
    val_loss, all_labels_flat, all_predictions_flat = evaluate_model(trained_model, val_loader, criterion,
                                                                     plot_dir="plots")

    # Save scatter plot and random samples
    scatter_plot_path = f"forward_scatter_plot_{model_name}.png"
    random_samples_path = f"random_samples_{model_name}.png"
    residuals_path = f"residuals_{model_name}.png"
    training_log_path = f"training_log_{model_name}.png"

    # Log evaluation metrics and images
    try:
        plot_scatter_plot(all_labels_flat, all_predictions_flat, save_path=scatter_plot_path)
        wandb.log({"scatter_plot": wandb.Image(scatter_plot_path)})
    except:
        print("could not plot scatter plot")

    try:
        show_random_samples(trained_model, val_dataset, num_samples=5, save_path=random_samples_path)
        wandb.log({"random_samples": wandb.Image(random_samples_path)})
    except:
        print("could not plot random samples")

    try:
        plot_residuals(all_predictions_flat, all_labels_flat, save_path=residuals_path)
        wandb.log({"residuals_plot": wandb.Image(residuals_path)})
    except:
        print("could not plot residuals")

    try:
        plot_training_log(training_log, training_log_path)
        wandb.log({"training_log": wandb.Image(training_log_path)})
    except:
        print("could not plot training log")



    # Finish the WandB run
    wandb.finish()