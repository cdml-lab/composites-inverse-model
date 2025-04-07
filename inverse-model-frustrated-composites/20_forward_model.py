

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                 Imports                                   |
# └───────────────────────────────────────────────────────────────────────────┘

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
from pathlib import Path
import subprocess

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                 Definitions                               |
# └───────────────────────────────────────────────────────────────────────────┘


seed = 42  # Set the seed for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

# Set variables

# Set dataset name
dataset_name="60-701-82-83-additions_xyz"

features_channels = 1
labels_channels = 3


# PAY ATTENTION: since this is a forward models the files are flipped and the labels file will be the original features
# file! and the same foe feature will be the original labels file, meant for in inverse model.

# Get the script's directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Dataset directory (datasets are parallel to the code folder)
dataset_dir = project_root / "frustrated-composites-dataset"

# Defines the training files
labels_file = f"{dataset_dir}/{dataset_name}/{dataset_name}_Merged_Features.h5"
features_file = f"{dataset_dir}/{dataset_name}/{dataset_name}_Merged_Labels.h5"

# Define the path and name for saving the model
current_date = datetime.datetime.now().strftime("%Y%m%d")
model_name = f"forward_{dataset_name}_{current_date}.pkl"

save_model_path = f"{script_dir}/saved_models/{model_name}"
load_model_path = save_model_path



train = 'yes'  #If you want to load previously trained model for evaluation - set to 'load' and correct the load_model_path
is_random = 'yes' #random samples to visualize

# Set normalization bounds manually!
# If using orientation loss the vector elements should be normalized in the same way, length can be seperate

# Curvature 3 channels
global_feature_max = [180.0]
global_feature_min = [0.0]



# Curvature Max and Min New
# global_label_max = [10.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# global_label_min = [-10.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5]

# global_label_max = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
# global_label_min = [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3]

#XYZ

global_label_max = [21.0, 21.0, 21.0]
global_label_min = [-21.0, -21.0, -1.0]


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           General Functions                               |
# └───────────────────────────────────────────────────────────────────────────┘

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
        """
        Retrieve the feature and label data for the specified index.

        Args:
            idx (int): Index of the dataset to retrieve.

        Returns:
            tuple: Transformed feature and label tensors.
        """
        with h5py.File(self.features_file, 'r') as f_features, h5py.File(self.labels_file, 'r') as f_labels:
            dataset_name = self.filenames[idx]
            feature = f_features[self.feature_main_group][self.category][dataset_name][()]
            label = f_labels[self.label_main_group][self.category][dataset_name][()]

            if feature.size == 0 or label.size == 0:
                return None

            # Transform the feature and the label
            feature_tensor, label_tensor = data_transform(feature, label, label_min=self.global_label_min, label_max=self.global_label_max)

            return feature_tensor, label_tensor

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

def data_transform(feature, label, feature_max=180, label_min=None, label_max=None):
    """
    Normalize feature data and either normalize or standardize label data based on channel-specific ranges.

    Args:
        feature (np.ndarray): Feature data with shape (height, width, channels) and values in [0, 255].
        label (np.ndarray): Label data with shape (height, width, channels) with varying ranges.
        feature_max (float): Maximum feature value for normalization (default: 180).
        label_min (list of floats): Minimum values per label channel, required for normalization.
        label_max (list of floats): Maximum values per label channel, required for normalization.

    Returns:
        tuple: Normalized feature tensor and either normalized or standardized label tensor.
    """
    # Convert and normalize features
    feature_tensor = (torch.tensor(feature, dtype=torch.float32)) # Convert to Tensor

    feature_tensor = feature_tensor / feature_max # Normalize

    feature_tensor = feature_tensor.permute(2, 0, 1)

    # Convert and handle labels based on ranges
    label_tensor = torch.tensor(label, dtype=torch.float32)

    # Normalize each label channel to [0, 1] based on specified min and max
    for c in range(label_tensor.shape[2]):
        label_tensor[:, :, c] = (label_tensor[:, :, c] - label_min[c]) / (label_max[c] - label_min[c] + 1e-8)

    label_tensor = label_tensor.permute(2, 0, 1)

    return feature_tensor, label_tensor

# Train modwl without importances
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200, patience=15):
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    training_log = []

    input_gradients = None  # To store accumulated input gradients
    num_samples = 0  # To keep track of the number of samples

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Enable gradient computation for input
            inputs.requires_grad_(True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Accumulate gradients of the input
            if inputs.grad is not None:
                gradients = inputs.grad.detach().abs()  # Detach to prevent further tracking
                batch_size = inputs.size(0)
                num_samples += batch_size

                # Average gradients over the batch
                batch_gradients = gradients.mean(dim=0)  # Shape: (channels, height, width)

                # Accumulate the gradients
                if input_gradients is None:
                    input_gradients = batch_gradients
                else:
                    input_gradients += batch_gradients

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        end_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs} | Time: {end_time - start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        training_log.append((epoch + 1, train_loss, val_loss))

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

        # Check for NaNs
        if(val_loss != val_loss):
            early_stop = True
            break

    if early_stop:
        print("Loading best model from checkpoint...")
        model.load_state_dict(torch.load('forward_best_model.pth'))

    # # Compute the average gradient map after training
    # this can't be computed for diverse sizes
    # if input_gradients is not None:
    #     # Compute the average gradient map
    #     avg_gradient_map = input_gradients / num_samples
    #
    #     # Take the absolute value (since gradients can be negative)
    #     avg_gradient_map = avg_gradient_map.abs()
    #
    #     # Convert to NumPy and log in wandb
    #     avg_gradient_map_np = avg_gradient_map.cpu().numpy()
    #     log_global_normalized_heatmaps(avg_gradient_map_np)

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
    # errors = np.concatenate(all_predictions, axis=0).flatten() - np.concatenate(all_labels, axis=0).flatten()

    # plot_error_histogram(errors, plot_dir=plot_dir)

    print(f'Validation Loss: {val_loss:.4f}')

    # Use NumPy to concatenate arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Debug: Check the shape and range before denormalization
    print(f"Shape of Predictions before denormalization: {all_predictions.shape}")
    print(f"Shape of Labels before denormalization: {all_labels.shape}")
    print(f"Predictions (min, max) before denormalization: {all_predictions.min()}, {all_predictions.max()}")
    print(f"Labels (min, max) before denormalization: {all_labels.min()}, {all_labels.max()}")

    # # === Denormalize the predictions and labels based on the normalization method ===
    # if wandb.config.normalization == "global":
    #     # Global denormalization
    #     all_predictions = all_predictions * (
    #             global_labels_max_all_channels - global_labels_min_all_channels) + global_labels_min_all_channels
    #     all_labels = all_labels * (
    #             global_labels_max_all_channels - global_labels_min_all_channels) + global_labels_min_all_channels
    # else:
    #     # Per-channel denormalization, considering that predictions are clamped
    #     for c in range(labels_channels):
    #         # Denormalize the predictions only if they were not clamped
    #         if all_predictions[:, :, :, c].min() >= 0 and all_predictions[:, :, :, c].max() <= 1:
    #             all_predictions[:, :, :, c] = all_predictions[:, :, :, c] * (
    #                     global_label_max[c] - global_label_min[c]) + global_label_min[c]
    #         all_labels[:, :, :, c] = all_labels[:, :, :, c] * (global_label_max[c] - global_label_min[c]) + \
    #                                  global_label_min[c]

    # Debug: Check the min and max values after denormalization
    print(f"Predictions (min, max) after denormalization: {all_predictions.min()}, {all_predictions.max()}")
    print(f"Labels (min, max) after denormalization: {all_labels.min()}, {all_labels.max()}")

    # Now flatten the arrays for scatter plot
    all_predictions_flat = all_predictions.flatten()
    all_labels_flat = all_labels.flatten()

    # Now plot the scatter plot with denormalized values
    plot_scatter_plot(all_labels_flat, all_predictions_flat, save_path=os.path.join(plot_dir, 'scatter_plot.png'))

    return val_loss, all_labels_flat, all_predictions_flat


# For varied sizes
class VariableCollateFn:
    def __init__(self, max_height, max_width):
        self.max_height = max_height
        self.max_width = max_width

    def __call__(self, batch):
        return variable_collate_fn(batch, self.max_height, self.max_width)

def variable_collate_fn(batch, max_height, max_width):
    inputs, labels = zip(*batch)  # Unzip batch into separate lists

    # Function to pad tensors dynamically
    def pad_tensor(tensor, max_h, max_w):
        pad_h = max_h - tensor.shape[1]
        pad_w = max_w - tensor.shape[2]
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

    # Apply global padding to all tensors in the batch
    padded_inputs = [pad_tensor(x, max_height, max_width) for x in inputs]
    padded_labels = [pad_tensor(y, max_height, max_width) for y in labels]

    return torch.stack(padded_inputs), torch.stack(padded_labels)


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                       Visualisation Functions                             |
# └───────────────────────────────────────────────────────────────────────────┘


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

def plot_samples_with_annotations(loader_type, data_loader, num_samples=6, plot_dir=r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\plots"):
    """
    Iterate through the data_loader and plot samples with RGB values annotated for every 5x5 pixel block.
    Assumes variable input sizes and batch_size=1.
    """
    import os
    import matplotlib.pyplot as plt
    import torch

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for i, (features, labels) in enumerate(data_loader):
        if i >= num_samples:
            break

        # Ensure batch size is 1
        feature = features[0]
        label = labels[0]

        # Convert tensors to numpy
        feature_img = feature.permute(1, 2, 0).cpu().numpy()

        num_label_channels = label.shape[0]
        fig, axs = plt.subplots(1, num_label_channels + 1, figsize=(6 * (num_label_channels + 1), 6))
        fig.suptitle(f'Sample {i + 1} - Features and Labels with Annotations', fontsize=16)

        # Plot features
        axs[0].imshow(feature_img)
        axs[0].axis('off')
        axs[0].set_title('Features')

        for c in range(num_label_channels):
            label_img = label[c].cpu().numpy()
            axs[c + 1].imshow(label_img)
            axs[c + 1].axis('off')
            axs[c + 1].set_title(f'Label Channel {c + 1}')

            # Annotate each 5x5 pixel block
            for y in range(0, label_img.shape[0], 5):
                for x in range(0, label_img.shape[1], 5):
                    label_text = f"{label_img[y, x]:.2f}"
                    axs[c + 1].text(x, y, label_text, fontsize=8, color='white',
                                    bbox=dict(facecolor='black', alpha=0.5))

        img_path = os.path.join(plot_dir, f"debug_sample_forward_{loader_type}_{i + 1}.png")
        plt.savefig(img_path)
        plt.close()
        print(f"Saved debug plot for sample {i + 1} to {img_path}")

def plot_scatter_plot(labels, predictions, save_path):
    plt.figure(figsize=(20, 20))
    plt.scatter(labels,
                predictions,
                alpha=0.1,
                s = 1,  # Adjust the size of the dots
                c = 'teal',  # Set a uniform color (e.g., 'blue') or pass an array for varying colors
                )
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
    plt.title(
        f'Training and Validation Loss over Epochs\nFinal Loss - Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f}')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Training log plot saved to {plot_path}")

def log_global_normalized_heatmaps(gradient_map_np, title_prefix="Channel"):
    """
    Logs a heatmap for each channel in the input NumPy array to WandB.
    Normalizes the data using the global min and max across all channels.
    Uses a consistent color scale for all heatmaps and includes a shared color bar.
    Additionally, logs the average importance across all channels as another heatmap.

    Args:
        gradient_map_np (np.ndarray): The gradient map of shape (channels, height, width).
        title_prefix (str): Prefix for the title of each heatmap (default: "Channel").
    """
    num_channels, height, width = gradient_map_np.shape

    # Compute global min and max for consistent color scale
    global_min = np.min(gradient_map_np)
    global_max = np.max(gradient_map_np)

    # Avoid division by zero if the gradients are constant
    if global_max - global_min == 0:
        print("Warning: Gradient map is constant; skipping normalization.")
        normalized_map = gradient_map_np
    else:
        # Normalize globally across all channels
        normalized_map = (gradient_map_np - global_min) / (global_max - global_min)

    # Calculate the average importance for each channel
    avg_importances = np.mean(normalized_map, axis=(1, 2))

    # Create a single figure with subplots (one for each channel + average)
    fig, axs = plt.subplots(1, num_channels + 1, figsize=(5 * (num_channels + 1), 8), constrained_layout=True)
    fig.suptitle("Channel Heatmaps with Consistent Color Scale", fontsize=16)

    # Plot each channel with the same color scale
    for i in range(num_channels):
        ax = axs[i]
        channel_data = normalized_map[i]

        # Plot heatmap with consistent color scale
        cax = ax.imshow(channel_data, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f"{title_prefix} {i}")
        ax.set_xlabel(f"Avg Importance: {avg_importances[i]:.4f}")
        ax.axis('off')

        # Add average importance as subtitle
        ax.set_title(f"{title_prefix} {i}\nAvg Importance: {avg_importances[i]:.4f}", fontsize=10)

    # Add the average heatmap of all channels
    avg_heatmap = np.mean(normalized_map, axis=0)
    avg_ax = axs[-1]
    avg_cax = avg_ax.imshow(avg_heatmap, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    avg_ax.set_title("Average Heatmap")
    avg_ax.set_xlabel(f"Avg Importance: {np.mean(avg_importances):.4f}")
    avg_ax.axis('off')

    # Add a single color bar on the right side of the entire figure
    fig.colorbar(cax, ax=axs, orientation='vertical', fraction=0.01, pad=0.1)

    # Log the figure to WandB
    wandb.log({"Channel Heatmaps": wandb.Image(fig)})

    # Close the plot to free up memory
    plt.close(fig)
# ┌───────────────────────────────────────────────────────────────────────────┐
# │                             Model Class                                   |
# └───────────────────────────────────────────────────────────────────────────┘


class OurVgg16(torch.nn.Module):
    """
    Custom VGG-style model with conv-only architecture, no fully connected layers.
    Outputs a single-channel prediction (e.g., fiber orientation).
    """
    def __init__(self, dropout=0.3):
        super(OurVgg16, self).__init__()

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
        self.conv_12 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv_13 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_14 = torch.nn.Conv2d(in_channels=128, out_channels=labels_channels, kernel_size=3, padding=1)  # final output

        self.batch_norm_1 = torch.nn.BatchNorm2d(64)
        self.batch_norm_2 = torch.nn.BatchNorm2d(128)
        self.batch_norm_3 = torch.nn.BatchNorm2d(128)
        self.batch_norm_4 = torch.nn.BatchNorm2d(256)
        self.batch_norm_5 = torch.nn.BatchNorm2d(256)
        self.batch_norm_6 = torch.nn.BatchNorm2d(512)
        self.batch_norm_7 = torch.nn.BatchNorm2d(512)
        self.batch_norm_8 = torch.nn.BatchNorm2d(512)
        self.batch_norm_9 = torch.nn.BatchNorm2d(512)
        self.batch_norm_10 = torch.nn.BatchNorm2d(256)
        self.batch_norm_11 = torch.nn.BatchNorm2d(256)
        self.batch_norm_12 = torch.nn.BatchNorm2d(128)
        self.batch_norm_13 = torch.nn.BatchNorm2d(128)


        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x); x = self.batch_norm_1(x); x = self.relu(x)
        x = self.conv_2(x); x = self.batch_norm_2(x); x = self.relu(x)
        x = self.conv_3(x); x = self.batch_norm_3(x); x = self.relu(x)
        x = self.conv_4(x); x = self.batch_norm_4(x); x = self.relu(x)
        x = self.conv_5(x); x = self.batch_norm_5(x); x = self.relu(x)
        x = self.conv_6(x); x = self.batch_norm_6(x); x = self.relu(x)
        x = self.dropout(x);
        x = self.conv_7(x); x = self.batch_norm_7(x); x = self.relu(x)
        x = self.conv_8(x); x = self.batch_norm_8(x); x = self.relu(x)
        x = self.dropout(x);
        x = self.conv_9(x); x = self.batch_norm_9(x); x = self.relu(x)
        x = self.conv_10(x); x = self.batch_norm_10(x); x = self.relu(x)
        x = self.dropout(x);
        x = self.conv_11(x); x = self.batch_norm_11(x); x = self.relu(x)
        x = self.conv_12(x); x = self.batch_norm_12(x); x = self.relu(x)
        x = self.dropout(x);
        x = self.conv_13(x); x = self.batch_norm_13(x); x = self.relu(x)
        x = self.conv_14(x);
        # x = self.sigmoid(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x


class ReducedDepth(torch.nn.Module):
    """
    Custom VGG-style model with conv-only architecture, no fully connected layers.
    Outputs a single-channel prediction (e.g., fiber orientation).
    """
    def __init__(self, dropout=0.3):
        super(OurVgg16, self).__init__()

        self.conv_1 = torch.nn.Conv2d(in_channels=features_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_4 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_6 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv_7 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv_8 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_9 = torch.nn.Conv2d(in_channels=64, out_channels=labels_channels, kernel_size=3, padding=1)  # final output

        self.batch_norm_1 = torch.nn.BatchNorm2d(64)
        self.batch_norm_2 = torch.nn.BatchNorm2d(128)
        self.batch_norm_3 = torch.nn.BatchNorm2d(256)
        self.batch_norm_4 = torch.nn.BatchNorm2d(256)
        self.batch_norm_5 = torch.nn.BatchNorm2d(512)
        self.batch_norm_6 = torch.nn.BatchNorm2d(256)
        self.batch_norm_7 = torch.nn.BatchNorm2d(128)
        self.batch_norm_8 = torch.nn.BatchNorm2d(64)



        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x); x = self.batch_norm_1(x); x = self.relu(x)
        x = self.conv_2(x); x = self.batch_norm_2(x); x = self.relu(x)
        x = self.conv_3(x); x = self.batch_norm_3(x); x = self.relu(x)
        x = self.conv_4(x); x = self.batch_norm_4(x); x = self.relu(x)
        x = self.conv_5(x); x = self.batch_norm_5(x); x = self.relu(x)
        x = self.conv_6(x); x = self.batch_norm_6(x); x = self.relu(x)
        x = self.dropout(x);
        x = self.conv_7(x); x = self.batch_norm_7(x); x = self.relu(x)
        x = self.dropout(x);
        x = self.conv_8(x); x = self.batch_norm_8(x); x = self.relu(x)
        x = self.dropout(x);
        x = self.conv_9(x);
        # x = self.sigmoid(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x



# ┌───────────────────────────────────────────────────────────────────────────┐
# │                               Loss Options                                |
# └───────────────────────────────────────────────────────────────────────────┘

# Testing different loss functions
class SineCosineL1(nn.Module):
    def __init__(self):
        super(SineCosineL1, self).__init__()

    def forward(self, y_pred, y_true):
        # Embed x and y on the unit circle
        pred_embed = torch.stack((torch.cos(2 * torch.pi * y_pred), torch.sin(2 * torch.pi * y_pred)), dim=-1)
        true_embed = torch.stack((torch.cos(2 * torch.pi * y_true), torch.sin(2 * torch.pi * y_true)), dim=-1)

        # Calculate L1 distance in 2D space
        loss = torch.sum(torch.abs(pred_embed - true_embed), dim=-1)

        # Return the mean loss over the batch
        return loss.mean()  # or loss.sum() if you prefer summing over the batch

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

        # print(f"""
        # angle_loss_max_theta: {angle_loss_max_theta}, angle_loss_min_theta: {angle_loss_min_theta}
        # angle_loss_max_phi: {angle_loss_max_phi}, angle_loss_min_phi: {angle_loss_min_phi}
        # length_loss_max: {length_loss_max}, length_loss_min: {length_loss_min}
        # total_loss: {total_loss}
        # """)
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


def angular_difference(theta1, theta2):
    diff = torch.abs(theta1 - theta2)
    diff = torch.minimum(diff, 2 * torch.pi - diff).mean()  # Wrap angles to [0, π]
    # print("Angular difference input angle 1", theta1.flatten()[:2],"angle 2: ", theta2.flatten()[:2])
    # print("Computed angular difference:", diff.flatten()[:2])
    return diff

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                  Main Code                                |
# └───────────────────────────────────────────────────────────────────────────┘

#CUDA

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU")

    print(torch.__version__)
    print(torch.version.cuda)

    # Use the full configuration when not sweeping

    # Initialize WandB project
    wandb.init(project="forward_model", config={
        "dataset": dataset_name,
        "learning_rate": 0.00003,
        "epochs": 500,
        "batch_size": 64,
        "optimizer": "Adam",
        "loss_function": "L1",
        "normalization max": global_label_max,
        "normalization min": global_label_min,
        "dataset_name": dataset_name,
        "features_channels": features_channels,
        "labels_channels": labels_channels,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.1,
        "patience": 15,
        "dropout": 0.3,
        "lr_patience": 7,
        "w_theta": 1.0,
        "w_phi": 2.0,
        "w_length": 2.0
    })

    # Get last Git commit hash and message
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    commit_msg = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip()

    # Log to config or as a custom field
    wandb.config.update({
        "git_commit": commit_hash,
        "git_message": commit_msg
    })

    # Calculate global min and max values for normalization
    # global_feature_min, global_feature_max, global_label_min, global_label_max = (
    #     calculate_global_min_max(features_file, labels_file,'Labels','Features'))
    #
    #
    # # Get global values for all labels together
    # global_labels_min_all_channels = min(global_label_min)
    # global_labels_max_all_channels = max(global_label_max)

    # Initialize dataset and data loaders
    # PAY ATTENTION: the labels and feature files are flipped on purpose!
    # because this is a forward model and the files are built for inverse model

    train_dataset = FolderHDF5Data(features_file, labels_file, 'Labels', 'Features', 'Train',
                                   global_feature_min, global_feature_max, global_label_min,
                                   global_label_max)
    val_dataset = FolderHDF5Data(features_file, labels_file, 'Labels', 'Features', 'Test',
                                 global_feature_min, global_feature_max, global_label_min,
                                 global_label_max)

    # # Initialize dataset and data loaders
    # train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=8,
    #                           pin_memory=True,
    #                           drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=8,
    #                         pin_memory=True, drop_last=True)


    # Compute global max height and width from the dataset before DataLoader creation
    global_max_height = 0
    global_max_width = 0

    for inputs, labels in train_dataset:  # Iterate over dataset samples
        global_max_height = max(global_max_height, inputs.shape[1])  # Get max height
        global_max_width = max(global_max_width, inputs.shape[2])  # Get max width

    print(f"Global max height: {global_max_height}, Global max width: {global_max_width}")

    # Create collate function instance with precomputed global max dimensions
    collate_function = VariableCollateFn(global_max_height, global_max_width)

    # Define DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=True,
        num_workers=8,  # Only keep >0 if needed
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_function  # Use the callable class instead of lambda
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_function  # Use the callable class instead of lambda
    )

    # See samples(for debugging)
    plot_samples_with_annotations('train', train_loader, num_samples=4, plot_dir="plots")

    # Select Loss Function
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
    elif wandb.config.loss_function == 'SineCosineL1':
        criterion = SineCosineL1()
    elif wandb.config.loss_function == 'Orientation':
        print("using orientation loss")
        criterion = OrientationLoss(w_theta=wandb.config.w_theta, w_phi=wandb.config.w_phi,w_length=wandb.config.w_length)
    else:
        print(f"Unknown loss function: {wandb.config.loss_function}, using L1")
        criterion = nn.L1Loss()

    # Initialize model
    # model = OurVgg16().to(device)
    model = ReducedDepth().to(device)
    wandb.watch(model, log="all", log_freq=100)  # log gradients & model
    # Set Optimizer
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=wandb.config.scheduler_factor,
                                                     patience=wandb.config.lr_patience)

    # Run the training
    if train == 'yes':
        print("Training Model")

        config = wandb.config

        if os.path.exists('forward_best_model.pth'):
            os.remove('forward_best_model.pth')
            print("Deleting Old Model...")


        trained_model, training_log = train_model(model, train_loader, val_loader,
                                                                          criterion=criterion, optimizer=optimizer,
                                                                          scheduler=scheduler,
                                                                          patience=wandb.config.patience,
                                                                          num_epochs=wandb.config.epochs)

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
    val_loss, all_labels_flat, all_predictions_flat = evaluate_model(trained_model,
                                                                     val_loader,
                                                                     criterion=criterion,
                                                                     plot_dir="plots")

    # Save plots
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
