import h5py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb
import numpy as np
import matplotlib.pyplot as plt
import wandb


# Define paths
# excel_file_path = r"C:\Users\User\OneDrive - Technion\Documents\GitHub\public-repo\inverse-model-frustrated-composites\saved_models_for_checks\test\test1_reshaped.h5"

# new_samples_file_path_features = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\Test2\Test2_All_Features_Reshaped.h5"
# new_samples_file_path_labels = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\Test2\Test2_All_Labels_Reshaped.h5"
# new_samples_file_path_features = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\30\30_Location_Labels_Reshaped.h5"
# new_samples_file_path_labels = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\30\30_Location_Features_Reshaped.h5"


new_samples_file_path_features = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\30-35\30-35_MaxMinCurvature_Labels_Reshaped.h5.h5"
new_samples_file_path_labels = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\30-35\30-35_MaxMinCurvature_Features_Reshaped.h5"

model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\forward_best_model.pth"

x=11 # Random sample selection

import random
random.seed(1)

#In this file:
# Features are angles
# Labels are geometry(xyz)


# Define parameters
features_channels = 1
labels_channels = 8

features_main_group = 'Labels'
labels_main_group = 'Features'
category = 'Test'
feature_data_exists = False


# Curvature Max and Min
global_labels_max = [10.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
global_labels_min = [-10.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5]

global_features_min = 0
global_features_max = 180

# Location
# global_label_max = [10.0,10.0,3.0]
# global_label_min = [-10.0,-10.0,-3.0]

# Inverse
# Global Feature Min: -1.043418
# Global Feature Max: 1.949431
# Global Label Min for channel 0: 0.0
# Global Label Max for channel 0: 179.0



# Initialize wandb
wandb.init(project="test_xyz_prediction", mode="disabled")

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

# Load and preprocess Excel data
# Visualization function
def visualize_xyz(points_xyz, step, plot_name="XYZ Visualization"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_xyz[:, :, 0], points_xyz[:, :, 1], points_xyz[:, :, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(plot_name)

    # Log the figure to WandB
    wandb.log({plot_name: wandb.Image(fig)})

    plt.show()
    plt.close()

def visualize_xyz_channels_2d(points_xyz, step, plot_name="Prediction XYZ Channels 2D"):
    """
    Visualizes the X, Y, and Z predictions as separate 2D images with color gradients.

    :param points_xyz: A tensor or array of shape (height, width, 3) containing XYZ values.
    :param step: Current step in the process (for logging purposes).
    :param plot_name: Base name for logging the plot in WandB.
    """
    channels = ['X', 'Y', 'Z']
    normalized_channels = [(points_xyz[..., i] - points_xyz[..., i].min()) /
                           (points_xyz[..., i].max() - points_xyz[..., i].min())
                           for i in range(3)]

    # Create a figure with three subplots for X, Y, and Z channels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, channel_data in enumerate(normalized_channels):
        # Display each channel as a 2D image with color gradient
        im = axes[i].imshow(channel_data, cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(f'{channels[i]} Channel')
        axes[i].axis('off')

        # Add a color bar to the side of each subplot
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(f'XYZ Channels as 2D Images at Step {step}')

    # Log the figure to WandB
    wandb.log({plot_name: wandb.Image(fig)})

    plt.show()
    plt.close()

def export_each_channel_to_excel(prediction_np, base_save_path="predictions_channel"):

    for i in range(prediction_np.shape[2]):
        channel_data = prediction_np[:, :, i]  # Shape (20, 15, x) for each channel
        df = pd.DataFrame(channel_data)

        # Define a unique filename for each channel
        save_path = f"{base_save_path}_channel_{i + 1}.xlsx"
        df.to_excel(save_path, index=False, sheet_name=f"Channel_{i + 1}", header=False)

        print(f"Channel {i + 1} exported to {save_path}")

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

    print(f"Features Before Normalization: {data}")

    # Normalize the features using the global min and max
    normalized_data = (data - global_features_min) / (global_features_max - global_features_min)

    # Convert to PyTorch tensor and add a batch dimension
    feature_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0)

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

    return feature_tensor

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

def show_samples():

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
        wandb.log({f"random_samples{i + 1}": wandb.Image(sample_save_path)})

import matplotlib.pyplot as plt

def visualize_filtered_scatter(true_values, predicted_values, value_range=(0, 0.1), channel=0):
    """
    Plots a scatter plot of filtered true values against corresponding predicted values
    for a specified range and channel.

    Args:
        true_values (numpy.ndarray): Ground truth values of shape [channels, height, width].
        predicted_values (numpy.ndarray): Predicted values of shape [channels, height, width].
        value_range (tuple): Range of true values to filter for plotting (min, max).
        channel (int): Channel index to visualize (0-indexed).
    """
    # Extract the channel data
    true_channel = true_values[channel, :, :].flatten()
    predicted_channel = predicted_values[channel, :, :].flatten()

    # Filter based on the specified range
    filter_mask = (true_channel >= value_range[0]) & (true_channel <= value_range[1])
    filtered_true = true_channel[filter_mask]
    filtered_predicted = predicted_channel[filter_mask]

    # Scatter plot of filtered values
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_true, filtered_predicted, alpha=0.6, label=f"Channel {channel + 1}")
    plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Ideal Prediction")
    plt.title(f"Scatter Plot for Filtered Values (Range {value_range}) - Channel {channel + 1}")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_curvature_tensor(tensor):
    """
    Visualizes each channel of the tensor in a grid layout.

    Args:
        tensor (torch.Tensor): Input tensor with shape [1, labels_channels, height, width].
    """
    # Remove the batch dimension
    tensor = tensor.squeeze(0)

    # Check if the tensor has the expected number of channels
    if tensor.shape[0] != labels_channels:
        raise ValueError(f"Expected tensor with shape [1, {labels_channels}, height, width], but got {tensor.shape}")

    # Determine the grid size dynamically
    rows = 2
    cols = int(np.ceil(labels_channels / rows))

    # Set up the grid for visualization
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()  # Flatten axes to handle cases with fewer channels than grid slots

    for i in range(labels_channels):
        # Get the channel and display it in the respective subplot
        channel = tensor[i].cpu().detach().numpy()  # Move to CPU and convert to NumPy if needed
        ax = axes[i]
        im = ax.imshow(channel, cmap="viridis", aspect='auto')
        ax.set_title(f"Channel {i + 1}")
        ax.axis("off")
        fig.colorbar(im, ax=ax)

    # Turn off any unused axes
    for ax in axes[labels_channels:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Main script for evaluating the forward model

# Check if feature data already exists
if feature_data_exists:
    # Load feature data (fiber orientation angles) from an HDF5 file
    features_data = load_features_h5_data(
        features_file=new_samples_file_path_features,
        features_main_group=features_main_group,
        global_features_min=global_features_min,
        global_features_max=global_features_max,
        category=category
    )

    # Reduce the number of channels to 1 (assumes data shape: [batch_size, num_samples, height, width])
    features_data = features_data[:, x:x + 1, :, :]

else:
    print("Creating new random fiber orientations")
    # Generate random fiber orientations if data doesn't exist
    features_data = create_random_sample()
    features_data = features_data / global_features_max

# Prepare orientation data for exporting
orientation_array = features_data.numpy()
orientation_array = orientation_array.transpose(2, 3, 1, 0)  # Reorganize dimensions for Excel export
orientation_array = orientation_array[..., 0]  # Keep only relevant channel
orientation_array *= global_features_max  # Denormalize
print(orientation_array.shape)

# Export orientation data to Excel for visualization
export_each_channel_to_excel(orientation_array, base_save_path="fiber_orientation")

# Load the pre-trained model
model = OurModel()
model.load_state_dict(torch.load(model_path))  # Load saved model parameters
model.eval()  # Set the model to evaluation mode

# Generate predictions using the model
with torch.no_grad():
    predicted_xyz = model(features_data)  # Forward pass
    visualize_curvature_tensor(predicted_xyz)  # Visualize curvature tensor
    print(f"Predicted XYZ {predicted_xyz.dtype} Size: {predicted_xyz.size()}")

# Load ground truth labels
labels_data = load_labels_h5_data(
    labels_file=new_samples_file_path_labels,
    labels_main_group=labels_main_group,
    category=category
)
print(f"Ground Truth not normalized shape: {labels_data.size()}")

# Select the relevant data slice
labels_data = labels_data[x:x + 1, :, :, :].squeeze()
print(f"Ground Truth not normalized shape: {labels_data.size()}")

value_range = (0, 0.1)  # Define the range of true values to filter
for channel in range(labels_channels):
    visualize_filtered_scatter(labels_data, predicted_xyz, value_range=value_range, channel=channel)

# Denormalize predictions
predicted_xyz_denorm = predicted_xyz.clone()
for c in range(labels_channels):
    predicted_xyz_denorm[:, c, :, :] = predicted_xyz_denorm[:, c, :, :] * \
                                       (global_labels_max[c] - global_labels_min[c]) + global_labels_min[c]

# Rearrange dimensions for visualization/export
predicted_xyz_denorm = torch.permute(predicted_xyz_denorm, (2, 3, 1, 0))
print(f"after permute {predicted_xyz_denorm.size}")

# Convert predictions to NumPy format
predicted_xyz_np = predicted_xyz_denorm.squeeze().numpy()
print(f"after numpy {np.shape(predicted_xyz_np)}")

# Visualize predicted XYZ data
visualize_xyz(predicted_xyz_np, step=0, plot_name="Predicted XYZ Visualization")
export_each_channel_to_excel(predicted_xyz_np, base_save_path="predicted_labels")

# Convert ground truth labels to NumPy format
labels_data_np = labels_data.numpy()
print(f"GT numpy shape: {labels_data_np.shape}")

# Visualize ground truth XYZ data
visualize_xyz(labels_data_np, step=0, plot_name="GT XYZ Visualization")
export_each_channel_to_excel(labels_data_np, base_save_path="gt_labels")

# Plot true and predicted values for evaluation
import matplotlib.pyplot as plt

# Plot true and predicted values for each channel

# Generate scatter plots for filtered ranges

# End WandB session
wandb.finish()

