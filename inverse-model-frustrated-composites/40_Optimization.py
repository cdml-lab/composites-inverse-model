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
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

matplotlib.use('Agg')

import math
import nlopt
import wandb

# ANSI escape codes for colors
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
PURPLE = '\033[35m'
CYAN = '\033[36m'
RESET = '\033[0m'  # Reset to default color

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Definitions                                     │
# └───────────────────────────────────────────────────────────────────────────┘


# Input Files
model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\drawn-energy-281.pkl"
excel_file_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\rhino_to_model_inverse.xlsx"

inverse_model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\30-35_Curvature_Inverse_20241112.pth"


features_channels = 1
labels_channels = 8

num_of_cols = 3
num_of_rows = 4


# Normalization Aspect
global_labels_min = 0.0
global_labels_max = 180.0

# Define the number of initializations and noise strength
num_initializations = 5
noise_strength = 0.2  # Adjust the strength of the noise

# If using orientation loss the vector elements should be normalized in the same way, length can be seperate

# Curvature Min and Max
global_features_max = [10.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
global_features_min = [-10.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5]

# All Useful
# global_features_max = [9.7, 1.8, 1.0, 1.0, 0.6,
#                     1.0, 1.0, 0.5, 0.5, 0.5,
#                     1.0, 1.0, 0.7, 0.5, 0.5, 1.0, 0.5]
# global_features_min = [-5.9, -1.2, -1.0, -1.0, -0.6,
#                     -1.0, -1.0, -0.5, -0.5, -0.5,
#                     0.8, 0.7, -0.2, -0.5, -0.6, 0.9, -0.5]

# global_features_max = [0.5, 0.5, 1.0]
# global_features_min = [ -0.5, -0.5, 0.85]



# Optimization loop
max_iterations =2
desired_threshold = 0.001
visualize = True
is_show = False
print_steps = 10000 # Once in how many steps to print the prediction
learning_rate = 0.005
patience = 1000

# Variables to change
optimizer_type = 'basic' # basic, nl-opt
gradient_selection = 'average' # average, middle, median, weighted_average
# channels_to_keep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
channels_to_keep = [0,1,2,3,4,5,6,7]
# channels_to_keep = [0,1,2]

feature_titles = {
    1: "Max Curvature Length",
    2: "Min Curvature Length",
    3: "MaCD-X",
    4: "MaCD-Y",
    5: "MaCD-Z",
    6: "MiCD-X",
    7: "MiCD-Y",
    8: "MiCD-Z",
    9: "No-X",
    10: "No-Y",
    11: "No-Z",
    12: "U-X",
    13: "U-Y",
    14: "U-Z",
    15: "V-X",
    16: "V-Y",
    17: "V-Z",
    25: "Angle"
}


start_point = 'ByCurvature' # Should be 'ByCurvature' or a float (0.5 / 1.0 / 0.0  etc). some algorithms igone this
is_weighted_loss = False
log_to_wandb = True # Don't set to false, it breaks some things


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Model Architecture                              │
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

        # Don't apply ReLU if this is a regression problem, so no activation on the final layer
        # Constrain output values to the label range (0, 1)
        x = torch.sigmoid(x)
        return x


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Functions                                       │
# └───────────────────────────────────────────────────────────────────────────┘


def export_each_channel_to_excel(prediction_np, base_save_path="predictions_channel"):
    df = pd.DataFrame(prediction_np)

    # Define a unique filename for each channel
    save_path = f"{base_save_path}.xlsx"
    df.to_excel(save_path, index=False, sheet_name=f"Channel_1", header=False)

    print(f"predictions exported to {save_path}")


import torch
import numpy as np


def vector_to_spherical(x, y, z, length):
    """
    Converts a 3D vector (x, y, z) to spherical coordinates (theta, phi).
    """

    # Calculate theta (azimuthal angle)
    theta = torch.atan2(y, x)  # in radians

    # Calculate phi (polar angle)
    phi = torch.acos(z / length)  # in radians

    return theta, phi


class OrientationLoss(nn.Module):
    def __init__(self, w_theta=1.0,w_phi=1.0, w_length=1.0):
        super(OrientationLoss, self).__init__()
        self.w_length = w_length
        self.w_theta = w_theta
        self.w_phi = w_phi

    def forward(self, predictions, labels):
        # Extract lengths and vector components
        pred_max_length, pred_min_length = predictions[:, 0], predictions[:, 1]
        pred_max_vec = predictions[:, 2:5]
        pred_min_vec = predictions[:, 5:8]
        # print(f"preds: {pred_max_length.flatten()[:2]},{pred_min_length.flatten()[:2]},{pred_max_vec.flatten()[:2]},{pred_min_vec.flatten()[:2]}")

        true_max_length, true_min_length = labels[:, 0], labels[:, 1]
        true_max_vec = labels[:, 2:5]
        true_min_vec = labels[:, 5:8]

        # This is needed because normalization changes the vectors so they are not unit vectors anymore
        eps = 1e-8  # A small constant to avoid division by zero
        pred_max_vec_normalized = pred_max_vec / (pred_max_length.unsqueeze(1).clamp(min=eps))
        pred_min_vec_normalized = pred_min_vec / (pred_min_length.unsqueeze(1).clamp(min=eps))

        true_max_vec_normalized = true_max_vec / (true_max_length.unsqueeze(1).clamp(min=eps))
        true_min_vec_normalized = true_min_vec / (true_min_length.unsqueeze(1).clamp(min=eps))

        # Debugging prints to inspect a few values
        # print(f"Predictions shape {predictions.shape} and sample values: {predictions.flatten()[:4]}")
        # print(f"Pred Max Length: {pred_max_length.flatten()[:2]}")
        # print(f"Pred Min Length: {pred_min_length.flatten()[:2]}")
        # print(f"Pred Max X: {pred_max_vec.flatten()[:2]}")
        # print(f"Pred Min X: {pred_min_vec.flatten()[:2]}")
        #
        # print(f"True Max Length: {true_max_length.flatten()[:2]}")
        # print(f"True Min Length: {true_min_length.flatten()[:2]}")
        # print(f"True Max X: {true_max_vec.flatten()[:2]}")
        # print(f"True Min X: {true_min_vec.flatten()[:2]}")


        # Convert vectors to spherical coordinates
        # print(f"before normalizaion: {pred_max_vec.flatten()[:2]} //// after normalization pred max: {pred_max_vec_normalized.flatten()[:2]}")
        pred_max_theta, pred_max_phi = compute_spherical_coordinates(pred_max_vec_normalized, pred_max_length)
        pred_min_theta, pred_min_phi = compute_spherical_coordinates(pred_min_vec_normalized, pred_min_length)
        true_max_theta, true_max_phi = compute_spherical_coordinates(true_max_vec_normalized, true_max_length)
        true_min_theta, true_min_phi = compute_spherical_coordinates(true_min_vec_normalized, true_min_length)

        # Debug prints for checking intermediate values

        # Flatten the tensor and print only the first 3 elements for clarity
        # print(f"Pred Max Theta (first 3 values): {pred_max_theta.flatten()[:3]}")
        # print(f"Pred Max Phi (first 3 values): {pred_max_phi.flatten()[:3]}")
        # print(f"Pred Min Theta (first 3 values): {pred_min_theta.flatten()[:3]}")
        # print(f"Pred Min Phi (first 3 values): {pred_min_phi.flatten()[:3]}")
        #
        # print(f"True Max Theta (first 3 values): {true_max_theta.flatten()[:3]}")
        # print(f"True Max Phi (first 3 values): {true_max_phi.flatten()[:3]}")
        # print(f"True Min Theta (first 3 values): {true_min_theta.flatten()[:3]}")
        # print(f"True Min Phi (first 3 values): {true_min_phi.flatten()[:3]}")

        # Compute angular losses
        angle_loss_max_theta = angular_difference(pred_max_theta, true_max_theta)
        angle_loss_max_phi = angular_difference(pred_max_phi, true_max_phi)
        angle_loss_min_theta = angular_difference(pred_min_theta, true_min_theta)
        angle_loss_min_phi = angular_difference(pred_min_phi, true_min_phi)

        # print(f"angle losses max theta: {angle_loss_max_theta.flatten()[:2]} phi: {angle_loss_max_phi.flatten()[:2]}"
        #       f" // min theta: {angle_loss_min_theta.flatten()[:2]} phi: {angle_loss_min_phi.flatten()[:2]}")

        angle_loss_max = angle_loss_max_theta + angle_loss_max_phi
        angle_loss_min = angle_loss_min_theta + angle_loss_min_phi

        # Compute length losses
        length_loss_max = torch.abs(pred_max_length - true_max_length).mean()
        length_loss_min = torch.abs(pred_min_length - true_min_length).mean()


        # Total loss
        total_loss = (self.w_theta * (angle_loss_max_theta + angle_loss_min_theta) +
                      self.w_phi * (angle_loss_max_phi + angle_loss_min_phi) +
                      self.w_length * (length_loss_max + length_loss_min))
        # print(f"Total loss: {total_loss}, angle loss max: {angle_loss_max}, angle loss min: {angle_loss_min}, length loss max: {length_loss_max}, length loss min: {length_loss_min}")
        return total_loss


def compute_spherical_coordinates(vec, length):
    # Avoid NaN due to invalid input
    norm_vec = vec / (length.unsqueeze(1).clamp(min=1e-8))

    # Compute spherical coordinates
    x, y, z = norm_vec[:, 0], norm_vec[:, 1], norm_vec[:, 2]
    theta = torch.atan2(torch.sqrt(x ** 2 + y ** 2), z)
    phi = torch.atan2(y, x)

    # Clamp theta and phi to avoid invalid values
    theta = theta.clamp(min=0, max=3.1416)  # Ensure within valid range
    phi = phi.clamp(min=-3.1416, max=3.1416)
    return theta, phi

class VectorDotProductLoss(nn.Module):
    def __init__(self):
        super(VectorDotProductLoss, self).__init__()

    def forward(self, pred_vec, input_vec):
        # Ensure the vectors are normalized
        input_vec = input_vec / input_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pred_vec = pred_vec / pred_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Calculate dot product
        dot_product = torch.sum(pred_vec * input_vec, dim=-1)

        # Loss: (1 - |dot_product|) if both aligned and opposite are good
        loss = 1 - torch.abs(dot_product)

        # Loss: (|dot_product - 1|) if only aligned is good
        # loss =  torch.abs(dot_product - 1)

        #L2
        # loss = loss * loss

        # Enlarge because values are very small
        loss = loss * 1000

        # Mean loss across the batch
        return loss.mean()


class DotProductL1(nn.Module):
    """
    This is a custom loss function that only works if the data is organized like this:
    (maximum curvature length, minimum curvature length, maximum curvature x, maximum curvature y, maximum
    curvature z, minimum curvature x, minimum curvature y, minimum curvature z)
    """
    def __init__(self, w_lengths=0.5, w_max_dot=1.0, w_min_dot=1.0):
        super(DotProductL1, self).__init__()
        self.w_lengths = w_lengths
        self.w_max_dot = w_max_dot
        self.w_min_dot = w_min_dot

    def forward(self, predictions, targets):
        # Slicing for channel-first tensors
        max_length_pred, min_length_pred = predictions[:, 0, :, :], predictions[:, 1, :, :]
        max_vector_pred = predictions[:, 2:5, :, :]  # x, y, z for max curvature
        min_vector_pred = predictions[:, 5:8, :, :]  # x, y, z for min curvature

        max_length_target, min_length_target = targets[:, 0, :, :], targets[:, 1, :, :]
        max_vector_target = targets[:, 2:5, :, :]
        min_vector_target = targets[:, 5:8, :, :]

        # Normalize vectors to obtain unit vectors
        max_unit_pred = max_vector_pred / (torch.norm(max_vector_pred, dim=1, keepdim=True) + 1e-8)
        min_unit_pred = min_vector_pred / (torch.norm(min_vector_pred, dim=1, keepdim=True) + 1e-8)
        max_unit_target = max_vector_target / (torch.norm(max_vector_target, dim=1, keepdim=True) + 1e-8)
        min_unit_target = min_vector_target / (torch.norm(min_vector_target, dim=1, keepdim=True) + 1e-8)

        # Max Dot Loss Options
        # Option 1: Opposite is fine, only perpendicular is bad
        # Treat aligned and opposite directions as equally good (loss = 0), perpendicular is worst (loss = 1)
        # max_dot_loss = 1 - torch.abs(torch.sum(max_unit_pred * max_unit_target, dim=1))
        # min_dot_loss = 1 - torch.abs(torch.sum(min_unit_pred * min_unit_target, dim=1))

        # Option 2: Opposite is bad
        # Penalize opposite directions (loss = 2), aligned is best (loss = 0), perpendicular is moderate (loss = 1)
        # Use this when aligned directions are preferred, and opposite directions are strongly penalized

        max_dot_loss = 1 - torch.sum(max_unit_pred * max_unit_target, dim=1)
        min_dot_loss = 1 - torch.sum(min_unit_pred * min_unit_target, dim=1)

        # Option 3: Intermediate misalignment is worst
        # Aligned and opposite are best (loss = 0), intermediate angles like 45° or 135° are worst (loss = 1)
        # Use this to penalize vectors that deviate moderately rather than completely

        # dot_product_max = torch.sum(max_unit_pred * max_unit_target, dim=1)
        # max_dot_loss = 1 - dot_product_max ** 2
        # dot_product_min = torch.sum(min_unit_pred * min_unit_target, dim=1)
        # min_dot_loss = 1 - dot_product_min ** 2


        # Compute L1 loss for lengths
        length_loss = torch.abs(max_length_pred - max_length_target) + torch.abs(
            min_length_pred - min_length_target)  # Shape [1, 20, 15]

        # print(f"w length: {self.w_lengths}, w max dot: {self.w_max_dot}, w min dot: {self.w_min_dot}")
        # print(f"length loss: {length_loss.mean()} {length_loss.shape}, max dot loss: {max_dot_loss.mean()} {max_dot_loss.shape}"
        #       f", min dot loss: {min_dot_loss.mean()} {min_dot_loss.shape}")
        #
        # print(
        #     f"length_loss: {length_loss.shape}, max_dot_loss: {max_dot_loss.shape}, min_dot_loss: {min_dot_loss.shape}")

        # Combine losses with weights
        total_loss = (
                self.w_lengths * length_loss +
                self.w_max_dot * max_dot_loss +
                self.w_min_dot * min_dot_loss
        )

        # Return mean loss
        return total_loss.mean()


def angular_difference(theta1, theta2):
    diff = torch.abs(theta1 - theta2)
    diff = torch.minimum(diff, 2 * torch.pi - diff).mean()  # Wrap angles to [0, π]
    # print("Angular difference input angle 1", theta1.flatten()[:2],"angle 2: ", theta2.flatten()[:2])
    # print("Computed angular difference:", diff.flatten()[:2])
    return diff

def average_patches(df, patch_size, export_path_original, export_path_average):
    """
    Divide the input 3D DataFrame/array into patches and calculate the average value in each patch.
    Exports the original and averaged data to Excel for verification.

    Parameters:
        df (pd.DataFrame or np.ndarray): The input data of shape (height, width, channels).
        patch_size (tuple): The size of each patch as (patch_height, patch_width).
        export_path (str): The path to save the Excel file.

    Returns:
        np.ndarray: The averaged patches as an array with shape determined by the input and patch size.
    """
    # Ensure the input is a NumPy array
    if isinstance(df, pd.DataFrame):
        data = df.to_numpy()
    else:
        data = df

    h, w, c = data.shape  # Original dimensions
    ph, pw = patch_size  # Patch dimensions

    # Ensure input dimensions are divisible by patch size
    assert h % ph == 0, "Height is not divisible by patch height."
    assert w % pw == 0, "Width is not divisible by patch width."

    # Reshape to split into patches and calculate the mean
    reshaped = data.reshape(h // ph, ph, w // pw, pw, c)
    averaged_patches = reshaped.mean(axis=(1, 3))  # Average over the patch height and width dimensions

    # Export original and averaged data to Excel
    if export_path_original is not None:
        with pd.ExcelWriter(export_path_original) as writer:
            for channel in range(c):
                # Save original data for each channel
                original_channel_data = data[:, :, channel]
                pd.DataFrame(original_channel_data).to_excel(writer, sheet_name=f'Original Channel {channel + 1}',
                                                             index=False)
        with pd.ExcelWriter(export_path_average) as writer:
            for channel in range(c):
                # Save original data for each channel
                average_channel_data = averaged_patches[:, :, channel]
                pd.DataFrame(average_channel_data).to_excel(writer, sheet_name=f'Averaged Patches {channel + 1}',
                                                             index=False)

    return averaged_patches

def average_patches_gradients(data, patch_size):
    """
    Divide the input 2D tensor into patches and calculate the average value in each patch.

    Parameters:
        data (torch.Tensor): The input 2D tensor of shape (height, width).
        the data is the GRADIENTS!
        patch_size (tuple): The size of each patch as (patch_height, patch_width).

    Returns:
        torch.Tensor: The averaged patches as a 2D tensor with shape determined by the input and patch size.
    """

    h, w = data.shape[2],data.shape[3]  # Original dimensions
    # print(f"dimensions data: {h}, {w}")
    ph, pw = patch_size  # Patch dimensions

    # Ensure input dimensions are divisible by patch size
    assert h % ph == 0, "Height is not divisible by patch height."
    assert w % pw == 0, "Width is not divisible by patch width."

    # Reshape to split into patches and calculate the mean
    data = data.reshape(h // ph, ph, w // pw, pw)

    data = data.mean(axis=(1, 3))  # Average over the patch height and width dimensions

    # print("data shape:", data.shape)

    return data

def median_patches_gradients(data, patch_size):
    """
    Divide the input 2D tensor into patches and calculate the average value in each patch.

    Parameters:
        data (torch.Tensor): The input 2D tensor of shape (height, width).
        the data is the GRADIENTS!
        patch_size (tuple): The size of each patch as (patch_height, patch_width).

    Returns:
        torch.Tensor: The averaged patches as a 2D tensor with shape determined by the input and patch size.
    """

    h, w = data.shape[2],data.shape[3]  # Original dimensions
    # print(f"dimensions data: {h}, {w}")
    ph, pw = patch_size  # Patch dimensions

    # Ensure input dimensions are divisible by patch size
    assert h % ph == 0, "Height is not divisible by patch height."
    assert w % pw == 0, "Width is not divisible by patch width."

    # Reshape to split into patches and calculate the mean
    data = data.reshape(h // ph, ph, w // pw, pw)

    data = data.median(axis=(1, 3))  # Average over the patch height and width dimensions

    # print("data shape:", data.shape)

    return data

def weighted_average_patches_gradients(data, patch_size, weight_factor):
    """
    Divide the input 2D tensor into patches and calculate a weighted average in each patch,
    where the middle pixels have more influence than the outermost pixels.

    Parameters:
        data (torch.Tensor): The input 2D tensor of shape (height, width). The data is the GRADIENTS!
        patch_size (tuple): The size of each patch as (patch_height, patch_width).
        weight_factor (float): The factor by which the middle pixels are weighted more than the outermost pixels.

    Returns:
        torch.Tensor: The weighted averaged patches as a 2D tensor with shape determined by the input and patch size.
    """
    h, w = data.shape[2], data.shape[3]  # Original dimensions
    ph, pw = patch_size  # Patch dimensions

    # Ensure input dimensions are divisible by patch size
    assert h % ph == 0, "Height is not divisible by patch height."
    assert w % pw == 0, "Width is not divisible by patch width."

    # Reshape to split into patches
    reshaped_data = data.reshape(h // ph, ph, w // pw, pw)  # Shape: (num_patches_h, ph, num_patches_w, pw)

    # Create weights for the patch
    patch_h_indices = torch.arange(ph, device=data.device)  # Ensure weights are on the same device as data
    patch_w_indices = torch.arange(pw, device=data.device)
    patch_weights = torch.outer(
        1 + weight_factor * (1 - torch.abs(2 * patch_h_indices / (ph - 1) - 1)),
        1 + weight_factor * (1 - torch.abs(2 * patch_w_indices / (pw - 1) - 1))
    )
    patch_weights = patch_weights / patch_weights.sum()  # Normalize weights

    # Apply weights and compute weighted average for each patch
    weighted_patches = reshaped_data * patch_weights.unsqueeze(0).unsqueeze(2)  # Broadcast weights
    weighted_avg = weighted_patches.sum(dim=(1, 3))  # Weighted sum over patch height and width

    return weighted_avg

def middle_pixel_of_patches(data, patch_size):
    """
    Divide the input 2D tensor into patches and return the middle pixel of each patch.

    Parameters:
        data (torch.Tensor): The input 2D tensor of shape (batch_size, channels, height, width).
        patch_size (tuple): The size of each patch as (patch_height, patch_width).

    Returns:
        torch.Tensor: A tensor with the middle pixel of each patch. Shape will be (batch_size, channels, num_patches_vertical, num_patches_horizontal).
    """

    # Get the original dimensions
    batch_size, channels, h, w = data.shape
    ph, pw = patch_size  # Patch dimensions
    # print('------middle pixel debug----------')
    # print(f'batch size: {batch_size}, channels: {channels}, h:{h}, w:{w}, pw:{pw}, ph:{ph}')

    # Ensure input dimensions are divisible by patch size
    assert h % ph == 0, "Height is not divisible by patch height."
    assert w % pw == 0, "Width is not divisible by patch width."

    # Reshape data into patches (batch_size, channels, num_patches_vertical, patch_height, num_patches_horizontal, patch_width)
    data = data.unfold(2, ph, ph).unfold(3, pw, pw)
    # print(f"new data: {data.shape}")

    # Get the center pixel of each patch
    # For each patch, the middle index is the center of the patch
    middle_h = ph // 2
    middle_w = pw // 2
    # print(f"middle pixel h:{middle_h}, w:{middle_w}")

    # Extract the middle pixel (batch_size, channels, num_patches_vertical, num_patches_horizontal)
    middle_pixels = data[:, :, :, :, middle_h, middle_w]  # Selects the middle pixel from each patch
    middle_pixels.squeeze()
    # print(f"middle pixels shape: {middle_pixels.shape}")

    return middle_pixels

def excel_to_np_array(file_path, sheet_name='Sheet1', global_features_max=10.0, global_features_min=-10.0):
    """
    Reads an Excel file with 4 columns and 300 rows and converts it into a NumPy array
    of shape (20, 15, 4), with each column representing a channel and reorganizing the
    rows using Fortran order.

    Parameters:
    - file_path (str): Path to the Excel file.
    - sheet_name (str): Name of the sheet in the Excel file. Default is 'Sheet1'.

    Returns:
    - np.ndarray: A NumPy array of shape (20, 15, 4) with the data organized by (height, width, channels).
    """
    # Read the xlsx file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Drop the header and convert to NumPy array
    data = df.to_numpy()
    # print(f"numpy shape:", data.shape)


    # Check if the data has the correct shape (300, 4)
    if data.shape != (300, labels_channels):
        raise ValueError(f"Unexpected data shape {data.shape}, expected (300, {labels_channels})")

    # Reshape to 20x15x4 with Fortran-style order
    final_array = data.reshape((20, 15, labels_channels), order='F')

    print(f"from excel{final_array.shape}")


    for c in range(labels_channels):
        img = final_array[:, :, c]
        df_img = pd.DataFrame(img)
        df_img.to_excel(f'reshape_debug_channel_{c}.xlsx', index=False)
        # plt.imshow(img, cmap='gray')
        # plt.show()

    # Convert the NumPy array to a PyTorch tensor
    # print(data)
    data = torch.from_numpy(final_array).unsqueeze(0).float()

    print(f"after converting to tensor {data.size()}")
    data = torch.permute(data, dims=(0, 3, 1, 2))

    # Convert to tensors before normalization
    global_features_min = torch.tensor(global_features_min, dtype=torch.float32).view(1, labels_channels, 1, 1)
    global_features_max = torch.tensor(global_features_max, dtype=torch.float32).view(1, labels_channels, 1, 1)

    # Normalize the features using the global min and max
    normalized_data = (data - global_features_min) / (global_features_max - global_features_min)



    return normalized_data, final_array

def print_tensor_stats(tensor):
    max_val = tensor.max().item()
    min_val = tensor.min().item()
    mean_val = tensor.mean().item()
    print(f"Tensor Max: {max_val}, Min: {min_val}, Mean: {mean_val}")
    print(f"Tensor Shape: {tensor.size()}")

def sine_cosine_embedding_l2_loss(x, y):
    # Embed x and y on the unit circle
    x_embed = torch.stack((torch.cos(2 * torch.pi * x), torch.sin(2 * torch.pi * x)), dim=-1)
    y_embed = torch.stack((torch.cos(2 * torch.pi * y), torch.sin(2 * torch.pi * y)), dim=-1)

    # Calculate L2 distance in 2D space
    l2_distance = torch.norm(x_embed - y_embed, dim=-1)
    return l2_distance.mean()

def sine_cosine_embedding_l1_loss(x, y):
    # Embed x and y on the unit circle
    x_embed = torch.stack((torch.cos(2 * torch.pi * x), torch.sin(2 * torch.pi * x)), dim=-1)
    y_embed = torch.stack((torch.cos(2 * torch.pi * y), torch.sin(2 * torch.pi * y)), dim=-1)

    # Calculate L1 distance in 2D space
    l1_distance = torch.sum(torch.abs(x_embed - y_embed), dim=-1)
    return l1_distance.mean()

def visualize_curvature_tensor(tensor, labels_channels, iteration):
    # Remove the batch dimension
    tensor = tensor.squeeze(0)

    # Check if the tensor has the correct number of channels
    if tensor.shape[0] != labels_channels:
        raise ValueError(f"Expected tensor with shape [1, {labels_channels}, 20, 15], but got {tensor.shape}")

    # Calculate the grid size needed for visualization
    grid_size = math.ceil(math.sqrt(labels_channels))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 8))
    fig.suptitle(f"Tensor Visualization Iteration {iteration} ({labels_channels} Channels)")

    # Flatten the axes for easier indexing
    axes = axes.flatten()

    for i in range(labels_channels):
        # Get the channel and display it in the respective subplot
        channel = tensor[i].cpu().detach().numpy()  # Move to CPU and convert to NumPy if needed
        ax = axes[i]
        ax.imshow(channel, cmap="viridis", aspect=0.85)
        ax.set_title(f"Channel {i + 1}")
        ax.axis("off")

    # Hide any unused subplots
    for j in range(labels_channels, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top to fit the suptitle

    if is_show:
        plt.show()
    wandb.log({f"plot iteration {iteration}": wandb.Image(plt)})

def create_random_sample():
    # Initialize the fiber orientation with 16 distinct orientations for the 4x4 patches
    random_orientations = torch.randint(0, 181, (4, 4), dtype=torch.float32)
    # Create the (20, 15) grid by repeating the 16 values across the patches
    initial_fiber_orientation = torch.zeros((1, 1, 20, 15), dtype=torch.float32)
    # Loop over the 4x4 patches and fill the (20x15) grid
    for i in range(4):
        for j in range(4):
            # Calculate the ending indices while making sure they don't exceed the grid dimensions
            row_end = min((i + 1) * 5, 20)
            col_end = min((j + 1) * 5, 15)
            initial_fiber_orientation[:, 0, i * 5:row_end, j * 5:col_end] = random_orientations[i, j]
    return initial_fiber_orientation

def duplicate_pixel_data(initial_fiber_orientation):
    # Gets 12 numbers and duplicates them to match the expected grid of the model
    # Sample data should look like this:
    # random_numbers = [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]

    # Ensure the input is a PyTorch tensor
    if isinstance(initial_fiber_orientation, np.ndarray):
        initial_fiber_orientation = torch.tensor(initial_fiber_orientation, dtype=torch.float32)

    final_fiber_orientation = torch.zeros((1, 1, 20, 15))
    # Loop over the 4x4 patches and fill the (20x15) grid
    for i in range(4):
        for j in range(3):
            # Calculate the ending indices while making sure they don't exceed the grid dimensions
            row_end = min((i + 1) * 5, 20)
            col_end = min((j + 1) * 5, 15)
            final_fiber_orientation[:, 0, i * 5:row_end, j * 5:col_end] = initial_fiber_orientation[i, j]
    return final_fiber_orientation

def angle_with_x_axis(x, y, z):
    # Calculate the magnitude of the vector
    magnitude = math.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Avoid division by zero if the vector is zero
    if magnitude == 0:
        return 0.0

    # Calculate the cosine of the angle
    cos_theta = x / magnitude

    # Calculate the angle in radians and then convert to degrees
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)

    # If the vector points in the negative X direction, adjust the angle
    # if x < 0:
    #     angle_degrees = 180 - angle_degrees

    return angle_degrees

def calculate_angles(vectors):
    """
    Calculates the angles of vectors in a (20, 15, 8) array using the 2nd, 3rd, and 4th channels as x, y, and z.

    Parameters:
    - vectors (np.ndarray): Input array of shape (20, 15, 8).

    Returns:
    - np.ndarray: An array of shape (20, 15, 1) with the calculated angles.
    """
    # Initialize an empty array to store angles
    angle_array = np.zeros((vectors.shape[0], vectors.shape[1], 1))

    if labels_channels > 3:
        print("Calculating starting angle based on the 2,3,4 channels")
        # Select the 2nd, 3rd, and 4th channels for x, y, and z
        x = vectors[:, :, 2]
        y = vectors[:, :, 3]
        z = vectors[:, :, 4]

    # Iterate over each vector in the (20, 15) grid
    for i in range(vectors.shape[0]):  # Iterate over rows (20)
        for j in range(vectors.shape[1]):  # Iterate over columns (15)
            angle_array[i, j, 0] = angle_with_x_axis(x[i, j], y[i, j], z[i, j])

    return angle_array

def average_angles(angles):
    # Convert angles to radians if they are in degrees
    angles = np.deg2rad(angles)

    # Compute the unit circle coordinates
    x = np.cos(angles)
    y = np.sin(angles)

    # Average the coordinates
    avg_x = np.mean(x)
    avg_y = np.mean(y)

    # Calculate the average angle
    avg_angle = np.arctan2(avg_y, avg_x)

    # Convert back to degrees if needed
    avg_angle = np.rad2deg(avg_angle)

    # Ensure angle is between 0 and 360 degrees
    avg_angle = avg_angle % 360

    return avg_angle

def plot_optimization_log(values):
    # Plot the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(values, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss During NLopt Optimization")
    plt.legend()
    plt.grid(True)
    plt.show()

def fiber_orientation_to_excel(initial_fiber_orientation, global_labels_max, filename='optimized_fiber_orientation.xlsx'):
    """
    Converts the optimized fiber orientation tensor to a 2D DataFrame and saves it to Excel.

    Parameters:
        initial_fiber_orientation (torch.Tensor): The tensor with fiber orientations.
        global_labels_max (float): The maximum label value for scaling.
        filename (str): The filename for the output Excel file. Defaults to 'optimized_fiber_orientation.xlsx'.
    """
    # Duplicate the pixel data
    duplicate_for_export = duplicate_pixel_data(initial_fiber_orientation)

    # Convert the duplicated tensor to a DataFrame after scaling
    optimized_fiber_orientation_df = pd.DataFrame(
        np.squeeze((duplicate_for_export * global_labels_max))
    )

    # Save to Excel
    optimized_fiber_orientation_df.to_excel(filename, index=False, header=False)

    print(f"Data saved to {filename}")


def new_objective_function(x, grad=None):
    global call_count, best_loss, best_x, loss_values, max_gradient_value,mean_gradient_value, best_prediction
    call_count += 1
    x = np.array(x)


    # Convert x (numpy) to tensor (if it isn't already a tensor)
    if isinstance(x, np.ndarray):
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
    else:
        x_tensor = x.to(device)

    # print(f"Iteration: {call_count}")


    # Check if size matches the desired shape
    expected_size = (num_of_rows, num_of_cols, 1)
    if x_tensor.numel() != (num_of_rows * num_of_cols * 1):
        print(f"x size: {x_tensor.size()}, num of rows: {num_of_rows}, num of cols: {num_of_cols}")
        raise ValueError(f"Cannot reshape array of size {x_tensor.size()} into shape {expected_size}")

    # Reshape input
    x_tensor = x_tensor.reshape(num_of_rows, num_of_cols, 1)

    if call_count == 1:
        wandb.log({"initial_fiber_inside_objective_function": x})
        print(f"x inside the objective function (type: {type(x)}, shape: {len(x)}): {x}")

    # Duplicate data for prediction
    x_tensor = duplicate_pixel_data(x_tensor).to(device)
    x_tensor = x_tensor.clone().detach().requires_grad_(True).to(device)

    # Forward pass
    predicted = model(x_tensor)

    # Keep only channels to optimize on
    predicted = cull_channels(predicted, channels_to_keep)

    # Print every print_steps
    if call_count % print_steps == 0 or call_count == 1:
        visualize_curvature_tensor(predicted, len(channels_to_keep), call_count)

    loss = loss_fn(predicted, input_tensor)

    # Backward pass
    loss.backward()

    # Manually adjust gradients

    # Scale gradients to change learning rate
    gradients_temp = x_tensor.grad
    gradients_temp = gradients_temp * wandb.config.gradient_scaling_factor

    # Log gradient statistics
    print(f"Gradients Debug (Iter {call_count}): max={gradients_temp.max().item()}, "
          f"min={gradients_temp.min().item()}, mean={gradients_temp.mean().item()}, "
          f"shape={gradients_temp.shape}")

    if not torch.isfinite(gradients_temp).all():
        print(f"Warning: Non-finite gradients detected at Iter {call_count}")
        raise ValueError("Gradient contains NaN or Inf values.")

    # Options for how to convert 300 gradients to 12 gradients:
    if wandb.config.gradient_selection =='average':
        # print("Using average gradient selection")
        # Average grad of patch:
        x_tensor = average_patches_gradients(x_tensor, (5, 5))
        x_tensor.grad = average_patches_gradients(gradients_temp, (5, 5))

    elif wandb.config.gradient_selection =='middle':
        # print("Using middle gradient selection")
        # Middle pixel grad of each patch:
        x_tensor = middle_pixel_of_patches(x_tensor, (5,5))
        x_tensor.grad = middle_pixel_of_patches(gradients_temp, (5, 5))

    elif wandb.config.gradient_selection == 'median':
        x_tensor = median_patches_gradients(x_tensor, (5,5))
        x_tensor.grad = median_patches_gradients(gradients_temp, (5, 5))

    elif wandb.config.gradient_selection == 'weighted_average':
        x_tensor = weighted_average_patches_gradients(x_tensor, (5,5), wandb.config.weight_factor_of_pixels)
        x_tensor.grad = weighted_average_patches_gradients(gradients_temp, (5, 5), wandb.config.weight_factor_of_pixels)


    # Store for plotting
    loss_values.append(loss.item())  # Store the current loss to plot
    max_gradient_value.append(gradients_temp.max().item())
    mean_gradient_value.append(gradients_temp.mean().item())

    wandb.log({
        "loss": loss.item(),
        "gradient_max": gradients_temp.max().item(),
        "gradient mean": gradients_temp.mean().item(),
        "iteration": call_count
    })

    print(f"Iteration: {call_count} | Current Weighted Loss: {loss.item()} and grad mean: {x_tensor.grad.mean()}")

    # Track the best loss and corresponding x
    if loss.item() < best_loss:
        print(f"{YELLOW}Best loss updated{RESET}")
        print(f"{YELLOW}Old best loss: {best_loss} | New loss: {loss.item()}{RESET}")

        best_loss = loss.item()
        best_x = x.copy()
        best_prediction = predicted.detach().cpu()  # Store the best prediction tensor

    if not np.all(np.isfinite(x)):
        raise ValueError(f"Non-finite parameters detected: {x}")
    if not np.all(np.isfinite(grad)):
        raise ValueError(f"Non-finite gradients detected: {grad}")
    if not np.isfinite(loss.item()):
        raise ValueError(f"Non-finite loss detected: {loss.item()}")

    return loss.item()

def calculate_patch_weights(image_size, patch_size, grid_shape=(4, 3)):
    height, width = image_size
    patch_h, patch_w = patch_size
    rows, cols = grid_shape

    # Create an empty tensor for the final weights
    weights = torch.zeros((height, width), dtype=torch.float32)

    # Calculate patch centers for each patch in the grid
    patch_centers_y = torch.linspace(patch_h / 2, height - patch_h / 2, rows)
    patch_centers_x = torch.linspace(patch_w / 2, width - patch_w / 2, cols)

    # Loop through each patch's center and compute the weight for each pixel in the patch's region
    for center_y in patch_centers_y:
        for center_x in patch_centers_x:
            # Calculate exact patch boundaries
            patch_top_left_y = max(int(center_y - patch_h / 2), 0)
            patch_top_left_x = max(int(center_x - patch_w / 2), 0)
            patch_bottom_right_y = min(int(center_y + patch_h / 2), height)
            patch_bottom_right_x = min(int(center_x + patch_w / 2), width)

            # Create a grid of distances for the patch region
            y, x = torch.meshgrid(
                torch.arange(patch_top_left_y, patch_bottom_right_y, dtype=torch.float32),
                torch.arange(patch_top_left_x, patch_bottom_right_x, dtype=torch.float32),
                indexing='ij'
            )

            # Calculate the distance from each pixel in the patch to the patch center
            distance_to_patch_center = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

            # Normalize distances and convert to weights (closer = higher weight)
            max_distance = (patch_h / 2) ** 2 + (patch_w / 2) ** 2
            normalized_distance = 1 - (distance_to_patch_center / torch.sqrt(torch.tensor(max_distance)))

            # Clip weights to ensure they don't wrap around patch boundaries
            weights[
                patch_top_left_y:patch_bottom_right_y,
                patch_top_left_x:patch_bottom_right_x
            ] = torch.maximum(
                weights[
                    patch_top_left_y:patch_bottom_right_y,
                    patch_top_left_x:patch_bottom_right_x
                ],
                normalized_distance
            )
    print("Patch Weights:")
    print(weights)
    return weights

def calculate_confidence(predicted):
    # Calculate confidence scores without requiring gradients
    # Use properties of `predicted` (e.g., magnitude or variation) instead of `.grad`
    confidence_scores = torch.abs(predicted)  # Replace with your logic
    return confidence_scores

def cull_channels(tensor, channels_to_keep):
    """
    Cull specified channels from a tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch, channels, x, y).
        channels_to_keep (list[int]): List of channel indices to retain.

    Returns:
        torch.Tensor: Tensor with only the specified channels retained.
    """

    # print(f"Input tensor requires_grad: {tensor.requires_grad}")


    # Validate input dimensions
    if len(tensor.shape) != 4:
        raise ValueError("Input tensor must have shape (batch, channels, x, y).")

    # Validate channel indices
    num_channels = tensor.shape[1]
    if any(ch < 0 or ch >= num_channels for ch in channels_to_keep):
        raise ValueError("Channels to keep must be within the range of available channels.")

    # Select the specified channels
    culled_tensor = tensor.index_select(1, torch.tensor(channels_to_keep, device=tensor.device))


    # print(f"Output tensor requires_grad: {culled_tensor.requires_grad}")

    return culled_tensor


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Main Code                                       │
# └───────────────────────────────────────────────────────────────────────────┘


print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize WandB project
if log_to_wandb:

    wandb.init(project="optimization", config={
        "channels_to_keep": channels_to_keep,
        "optimizer_type": optimizer_type,
        "start_point": start_point,
        "gradient_selection": gradient_selection,
        "is_weighted_loss": is_weighted_loss,
        "model_path": model_path,
        "max_iterations": max_iterations,
        "weight_factor_of_pixels": 3,
        "gradient_scaling_factor": 1.0,
        "learning_rate": learning_rate

    })
    config = wandb.config
    channels_to_keep = wandb.config.channels_to_keep
else:
    wandb.init(mode="disabled")  # Disables logging


# Define global variables to track the best loss and corresponding x
best_loss = float('inf')
best_x = None
loss_values = []  # Global list to store loss values
max_gradient_value = []
mean_gradient_value = []

# Import the surface to optimize from Excel
input_tensor, vector_df = excel_to_np_array(file_path=excel_file_path, sheet_name='Sheet1',
                                 global_features_max=global_features_max, global_features_min=global_features_min)

input_tensor = input_tensor.to(device)
print_tensor_stats(input_tensor)
if visualize:
    visualize_curvature_tensor(input_tensor, labels_channels,"wanted all channels")

# Keep only certain channels to optimize to
input_tensor = cull_channels(input_tensor,channels_to_keep)

if visualize:
    visualize_curvature_tensor(input_tensor, len(channels_to_keep),"wanted after culling channels")

# Define the Model
model = OurModel()
model.load_state_dict(torch.load(model_path))
model.to(device)

# ┌───────────────────────────────────────────────────────────────────────────┐
# │       Optimization Process - Matching Predicted Output to Excel Data      │
# └───────────────────────────────────────────────────────────────────────────┘


if start_point == 'ByCurvature':
    # print(f"vector array  shape: {vector_df.shape}")
    angles = calculate_angles(vector_df)
    # print(f"calculates angles array shap {angles.shape}")

    average_patches_np = average_patches(angles, (5, 5),
                                         r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\Optimization debug_Original.xlsx",
                                         r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\Optimization debug_Average.xlsx")

    # print("averages array shape: ", average_patches_np.shape)
    num_of_patches = average_patches_np.shape[0] * average_patches_np.shape[1]
    # print( f"number of patches: {num_of_patches}")

    # Convert the NumPy array to a PyTorch tensor
    average_patches_tensor = torch.tensor(average_patches_np, dtype=torch.float32)

    # Normalize the tensor by dividing by 180
    normalized_tensor = average_patches_tensor / 180.0
    # print(f"shape of average degrees array {normalized_tensor.shape}")

    # Export to excel for debugging purposes
    fiber_orientation_to_excel(normalized_tensor, global_labels_max, "initial_fiber_orientations.xlsx")

    initial_fiber_orientation = normalized_tensor.clone().detach().requires_grad_(True)
else:
    initial_fiber_orientation = torch.full((4, 3, 1), start_point).requires_grad_(True)


# Define the loss
loss_fn = nn.L1Loss()
# loss_fn = DotProductL1()
# loss_fn = OrientationLoss(w_theta=1.0,w_phi=2.5, w_length=20.0)
# loss_fn = VectorDotProductLoss()
# loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = nlopt.LD_SLSQP  # Replace with desired optimizer type, e.g., nlopt.LD_MMA
# GN_DIRECT
# LD_LBFGS
# NLOPT_LD_SLSQP

# Log choices to wandb
wandb.config.update({"optimizer": "Adam", # REMEBER TO UPDATE!!!!!
           "loss_fn": loss_fn,
           "initial_fiber_orientation": initial_fiber_orientation,
           "patience_lr" : patience
           })



# Generate the initial fiber orientation tensors with normalized noise
initializations = []
for i in range(num_initializations):
    if i == 0:
        # First initialization with no noise
        noisy_tensor = initial_fiber_orientation.clone().detach().requires_grad_(True)
    else:
        # Create random noise tensor
        noise = torch.randn((4, 3, 1)) * noise_strength  # Gaussian noise scaled by noise_strength

        # Add noise and normalize to [0, 1]
        noisy_tensor = initial_fiber_orientation + noise
        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)  # Ensure values remain in [0, 1]

        # Detach from computation graph and enable gradient computation
        noisy_tensor = noisy_tensor.detach().requires_grad_(True)

    # Append to initializations
    initializations.append(noisy_tensor)

print(f"Generated {len(initializations)} normalized initializations.")
print("initials:", initializations)



if optimizer_type == 'basic':
    print(f"Using basic optimizer")
    # Define the optimizer

    # Best tracking
    global_best_loss = float('inf')  # Track the best loss across all initializations
    global_best_result = None  # Track the best result across all initializations
    losses = []  # Store the best loss for each initialization
    results = []  # Store the best result for each initialization

    # Optimization loop over initializations
    for init_index, fiber_orientation in enumerate(initializations):
        print(f"Starting optimization with initialization {init_index + 1}")

        # Define optimizer and scheduler
        optimizer = optim.Adam(params=[fiber_orientation.requires_grad_(True)], lr=wandb.config.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=wandb.config.patience_lr, factor=0.1,
                                      verbose=True)

        # Local best tracking for this initialization
        local_best_loss = float('inf')
        local_best_result = None

        for step in range(max_iterations):
            optimizer.zero_grad()

            # Duplicate data for prediction
            duplicate_fiber_orientation = duplicate_pixel_data(fiber_orientation).to(device)
            print("duplicate fiber orientation")
            print(duplicate_fiber_orientation)
            # Forward pass
            predicted = model(duplicate_fiber_orientation)
            print("predicted")
            print(predicted)
            predicted = cull_channels(predicted, channels_to_keep)

            # Compute loss
            # print(f"predicted shape: {predicted.shape}")
            # print(f"input shape: {input_tensor.shape}")
            loss = loss_fn(predicted, input_tensor)

            # Track local best for this initialization
            if loss < local_best_loss:
                local_best_loss = loss
                local_best_result = fiber_orientation.clone().detach()
                print(f"{PURPLE}New best loss for initialization {init_index + 1}: {local_best_loss}{RESET}")

            # Print every x steps
            if step % print_steps == 0:
                print("Fiber Orientation Tensor:")
                print_tensor_stats(duplicate_fiber_orientation)

                print("Predicted Tensor:")
                print_tensor_stats(predicted)
                visualize_curvature_tensor(predicted, len(channels_to_keep), step)

            # Compute gradients and update
            loss.backward()
            optimizer.step()

            # Step the scheduler with the latest loss
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(loss.item())
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr < prev_lr:
                print(f"{YELLOW}Learning rate decreased: {prev_lr} -> {new_lr}{RESET}")

            # Print the loss for every 100 steps
            if step % 200 == 0:
                print(f'Step {step + 1}, Loss: {loss.item()}')

                # # Print gradient statistics
                # grad_size = fiber_orientation.grad
                # if grad_size is not None:
                #     max_grad = round(grad_size.max().item(), 4)
                #     min_grad = round(grad_size.min().item(), 4)
                #     mean_grad = round(grad_size.mean().item(), 4)
                #     print(f"Gradient stats - Max: {max_grad}, Min: {min_grad}, Mean: {mean_grad}")

            # Early stopping
            if loss.item() < desired_threshold:
                print(f"Desired threshold reached for initialization {init_index + 1}.")
                break

        # Store the local best result and loss
        losses.append(local_best_loss)
        results.append(local_best_result)

        # Update the global best result if this initialization is better
        if local_best_loss < global_best_loss:
            global_best_loss = local_best_loss
            global_best_result = local_best_result

    # Print all results
    print("\nAll Losses and Results:")
    for i, (loss, result) in enumerate(zip(losses, results)):
        print(f"Initialization {i + 1}: Loss = {loss}")

    # Print the final global best result
    print(f"\n{GREEN}Final Global Best Loss: {global_best_loss}{RESET}")
    print(f"Best Fiber Orientation (Global): {global_best_result}")

    # Final result (detach to prevent further computation)
    final_fiber_orientation = global_best_result.detach()




elif optimizer_type == 'nl-opt':
    print("Using nl-opt optimization")

    # Initialize NLopt optimizer
    # try to 0 with direct
    opt = nlopt.opt(optimizer, initial_fiber_orientation.numel())

    # Initialize the counter
    call_count = 0

    # Set bounds and stopping criteria
    opt.set_lower_bounds(0.0)
    opt.set_upper_bounds(1.0)
    opt.set_stopval(-1e100)  # Prevent stopping for a specific target value
    opt.set_ftol_rel(1e-2)  # Looser relative function tolerance
    opt.set_xtol_rel(1e-2)  # Looser relative parameter tolerance
    opt.set_ftol_abs(1e-2)  # Looser absolute function tolerance
    opt.set_xtol_abs(1e-2)  # Looser absolute parameter tolerance

    opt.set_maxeval(wandb.config.max_iterations)  # Allow up to 1000 iterations
    opt.set_initial_step(0.01)  # Adjust step size based on problem scale

    # Set the objective function
    opt.set_min_objective(new_objective_function)


    # Flatten initial fiber orientation for NLopt
    x0 = initial_fiber_orientation.cpu().detach().numpy().flatten()
    # print(f"x0 outside the objective function (type: {type(x0)}, shape: {len(x0)}): {x0}")

    # Run the optimizer
    optimized_x = opt.optimize(x0)

    # Log the termination reason
    result = opt.last_optimize_result()
    termination_reasons = {
        1: "Converged to target value",
        2: "Stopped due to ftol_rel",
        3: "Stopped due to xtol_rel",
        4: "Stopped due to ftol_abs",
        5: "Stopped due to xtol_abs",
        6: "Maximum evaluations reached",
        -1: "Generic failure",
        -2: "Invalid arguments",
        -3: "Out of memory",
        -4: "Roundoff errors",
        -5: "User-forced termination",
    }
    result = opt.last_optimize_result()
    print(f"Termination reason: {termination_reasons.get(result, 'Unknown')}")

    # Plot optimization log
    if is_show:
        plot_optimization_log(loss_values)
        plot_optimization_log(max_gradient_value)

    print("Optimization completed.")
    print(f"Final loss: {best_loss}, x shape: {best_x.shape}")
    wandb.log({"best loss": best_loss})

    if visualize:
        visualize_curvature_tensor(best_prediction,len(channels_to_keep), "final")


    final_fiber_orientation = optimized_x.reshape(num_of_rows, num_of_cols)


fiber_orientation_to_excel(final_fiber_orientation, global_labels_max)

print("Optimization complete. Result saved to Excel.")
# Finish the WandB run
wandb.finish()

