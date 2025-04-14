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
import wandb
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import subprocess
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                 Definitions                               |
# └───────────────────────────────────────────────────────────────────────────┘


seed = 42  # Set the seed for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

### Manual Definitions

# Set dataset name
dataset_name="60-701-82-83-additions_uniform_1_uv_smooth"

features_channels = 8
labels_channels = 1

# Manually insert values for normalization
global_label_max = [180.0]
global_label_min = [0.0]

# This should match the dataset inputs, if you're not sure do an analysis of the dataset using "analyse dataset" file
# in the utilities folder

# global_feature_max = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# global_feature_min = [-5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]


# global_feature_max = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
# global_feature_min = [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3]

# global_feature_max = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
# global_feature_min = [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]

global_feature_max = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
global_feature_min = [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]

# global_feature_max = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
# global_feature_min = [-0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15]
channel_list = []


# Get the script's directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Dataset directory (datasets are parallel to the code folder)
dataset_dir = project_root / "frustrated-composites-dataset"

# Defines the training files
labels_file = f"{dataset_dir}/{dataset_name}/{dataset_name}_Merged_Labels.h5"
features_file = f"{dataset_dir}/{dataset_name}/{dataset_name}_Merged_Features.h5"

# Define the path and name for saving the model
current_date = datetime.datetime.now().strftime("%Y%m%d")
model_name = f"{dataset_name}_{current_date}.pkl"

save_model_path = f"{script_dir}/saved_models/{model_name}"
load_model_path = save_model_path


train = 'yes' #If you want to load previously trained model for evaluation - set to 'no' and correct the load_model_path
is_random = 'yes'
resize_data = False #depends on model choice
scale = 6.4


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           General Functions                               |
# └───────────────────────────────────────────────────────────────────────────┘


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

    h, w = feature.shape[:2]
    new_h, new_w = int(round(h * scale)), int(round(w * scale))


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
        for c in range(label_tensor.shape[2]):
            label_tensor[:, :, c] = (label_tensor[:, :, c] - global_label_min_tensor[c]) / (
                    global_label_max_tensor[c] - global_label_min_tensor[c])


    # Reorder dimensions: from (height, width, channels) to (channels, height, width)
    label_tensor = label_tensor.permute(2, 0, 1).float()

    # Resize using nearest neighbor
    if resize_data:
        feature_tensor = TF.resize(feature_tensor, size=[new_h, new_w], interpolation=InterpolationMode.BILINEAR)
        label_tensor = TF.resize(label_tensor, size=[new_h, new_w], interpolation=InterpolationMode.NEAREST)


    return feature_tensor, label_tensor

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

            has_gradients = False
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"[NO GRAD] {name}")
                elif param.grad.abs().sum() == 0:
                    print(f"[ZERO GRAD] {name} (shape: {param.shape})")
                else:
                    # print(f"[OK] {name} grad mean: {param.grad.abs().mean().item():.2e}")
                    has_gradients = True

            if not has_gradients:
                print("❌ No gradients detected for any parameters!")


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
            torch.save(model.state_dict(), 'inverse_best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {patience} epochs of no improvement.')
            early_stop = True
            break

    if early_stop:
        print("Loading best model from checkpoint...")
        model.load_state_dict(torch.load('inverse_best_model.pth'))

    # Compute the average gradient map after training
    if input_gradients is not None:
        # Compute the average gradient map
        avg_gradient_map = input_gradients / num_samples

        # Take the absolute value (since gradients can be negative)
        avg_gradient_map = avg_gradient_map.abs()

        # Convert to NumPy and log in wandb
        avg_gradient_map_np = avg_gradient_map.cpu().numpy()
        log_global_normalized_heatmaps(avg_gradient_map_np)

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

            print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}, Label shape: {labels.shape}")

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Convert tensors to CPU before converting to numpy
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_loss /= len(val_loader)

    # Use NumPy to concatenate arrays

    errors = np.concatenate(all_predictions, axis=0).flatten() - np.concatenate(all_labels, axis=0).flatten()

    print(f'Validation Loss: {val_loss:.4f}')

    # Flatten the predictions and labels for the scatter plot
    all_predictions_flat = np.concatenate([p.reshape(-1) for p in all_predictions], axis=0)
    all_labels_flat = np.concatenate([l.reshape(-1) for l in all_labels], axis=0)

    # all_predictions_flat = np.concatenate(all_predictions, axis=0)
    # all_labels_flat = np.concatenate(all_labels, axis=0).flatten()

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

def show_random_samples(model, dataset, num_samples=6, save_path="random_samples.png"):
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

        # Convert the label tensor to a numpy array (assuming label now has only 1 channel)
        label_img = label_tensor.cpu().numpy().squeeze()
        label_img = (label_img - label_img.min()) / (label_img.max() - label_img.min())

        # Create a figure with an extra row for the label image (which is now single-channel)
        fig, axs = plt.subplots(2 + feature_tensor.shape[0], 1, figsize=(10, 25))  # Adjust the figsize as needed
        fig.suptitle(f'Sample {i + 1}', fontsize=20)

        # Display the label image (single channel) in the first row
        axs[0].imshow(label_img, cmap='plasma')
        axs[0].axis('off')
        axs[0].set_title('Ground Truth Label')

        # Plot each feature channel separately
        for c in range(feature_tensor.shape[0]):
            feature_img = feature_tensor[c, :, :].cpu().numpy().squeeze()
            feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())

            axs[c + 1].imshow(feature_img, cmap='viridis')
            axs[c + 1].axis('off')
            axs[c + 1].set_title(f'Feature Channel {c + 1}')

        # Display prediction in the last row
        prediction_img = prediction_tensor.cpu().numpy().squeeze()
        axs[-1].imshow(prediction_img, cmap='plasma')
        axs[-1].axis('off')
        axs[-1].set_title('Prediction')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.1)  # Add space between rows

        # Save the figure as an image file
        sample_save_path = save_path.replace(".png", f"inverse_sample_{i + 1}.png")
        plt.savefig(sample_save_path)
        plt.close()
        print(f"Sample {i + 1} saved to {sample_save_path}")


        # Log random sample plot to wandb
        wandb.log({f"random_sample_{i + 1}": wandb.Image(sample_save_path)})

def plot_samples_with_annotations(loader_type, data_loader, num_samples=2, plot_dir="plots"):
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

        # Normalize for visualization (optional)
        # feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())

        # Create subplots based on the number of feature channels
        num_feature_channels = feature.shape[0]
        num_label_channels = 1  # Assuming a single channel for the label in the inverse model
        fig, axs = plt.subplots(num_feature_channels + num_label_channels, 1, figsize=(10, 5 * (num_feature_channels + num_label_channels)))

        fig.suptitle(f'Sample {i + 1} - Features and Label with Annotations', fontsize=16)

        # Plot each channel of the features separately
        for c in range(num_feature_channels):
            feature_img = feature[c, :, :].cpu().numpy()
            # Optional normalization
            # feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
            axs[c].imshow(feature_img, cmap='viridis')
            axs[c].axis('off')
            axs[c].set_title(f'Feature Channel {c + 1}')

        # Plot the label image (single channel)
        label_img = label.cpu().numpy().squeeze()  # Remove the singleton dimension
        # Optional normalization
        # label_img = (label_img - label_img.min()) / (label_img.max() - label_img.min())
        axs[-1].imshow(label_img, cmap='plasma')
        axs[-1].axis('off')
        axs[-1].set_title('Ground Truth Label')

        img_path = os.path.join(plot_dir, f"debug_sample_inverse_{loader_type}_{i + 1}.png")
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
    # Log to wandb
    wandb.log({f"Scatter Plot": wandb.Image(save_path)})
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

    # Log residuals plot to wandb
    wandb.log({"residuals_plot": wandb.Image(save_path)})

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
        ax = axs[i] if num_channels > 1 else axs  # Handle case with a single channel
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

class FolderHDF5Data(Dataset):
    def __init__(self, features_file, labels_file, feature_main_group, label_main_group, category, global_feature_min, global_feature_max, global_label_min, global_label_max):
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
            feature_tensor, label_tensor = data_transform(feature, label, self.global_feature_min, self.global_feature_max, self.global_label_min, self.global_label_max)

            return feature_tensor, label_tensor

class OurVgg16(torch.nn.Module):
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
        self.conv_12 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_13 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.conv_14 = torch.nn.Conv2d(128, labels_channels, kernel_size=3, padding=1)

        # Testing FC layer
        self.conv_fc1 = torch.nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.norm_fc1 = torch.nn.BatchNorm2d(512)
        self.relu_fc1 = torch.nn.ReLU()
        self.drop_fc1 = torch.nn.Dropout(p=dropout)
        self.conv_fc2 = torch.nn.Conv2d(512, labels_channels, kernel_size=1)


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
        # self.batch_norm_14 = torch.nn.BatchNorm2d(num_features=1)


        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_7(x)
        x = self.batch_norm_7(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_8(x)
        x = self.batch_norm_8(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_9(x)
        x = self.batch_norm_9(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_10(x)
        x = self.batch_norm_10(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_11(x)
        x = self.batch_norm_11(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_12(x)
        x = self.batch_norm_12(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv_13(x)
        x = self.batch_norm_13(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # print(f"after conv13 {x.shape}")

        # x = self.conv_14(x)
        # x = self.batch_norm_14(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # print(f"after conv14 {x.shape}")

        # Testing FC layer
        x = self.conv_fc1(x)
        x = self.norm_fc1(x)
        x = self.relu_fc1(x)
        x = self.drop_fc1(x)
        x = self.conv_fc2(x)

        x = self.sigmoid(x)


        return x

class FCNVGG16(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, dropout=0.5):
        super(FCNVGG16, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /4

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /8

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /16
            #
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # /32
        )

        # Fully convolutional layers (originally fc6 and fc7)
        # Kernel size reduced from 7 to 3
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(dropout)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(dropout)

        self.score = nn.Conv2d(1024, output_channels, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]  # Save original size

        # All the blocks
        x = self.features(x)

        # "Fully connected"
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        x = self.score(x)
        x = torch.clamp(x, 0.0, 1.0)

        if resize_data:
            # print(x.size())
            x = F.interpolate(x, size=input_shape, mode='nearest')

        return x

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                               Loss Options                                |
# └───────────────────────────────────────────────────────────────────────────┘


class AngularL1Loss(nn.Module):
    def __init__(self):
        super(AngularL1Loss, self).__init__()

    def forward(self, predictions, labels):

        # Compute the absolute difference
        diff = torch.abs(predictions - labels)

        # Wrap the differences to ensure they are between 0° and 90°
        wrapped_diff = torch.minimum(diff, 1.0 - diff)

        # Take the mean (L1 loss)
        loss = wrapped_diff.mean()

        # Debugging prints
        # print(f"predictions {predictions}")
        # print(f"labels {labels}")
        # print(f"Diff: {diff}")
        # print(f"Wrapped Diff: {wrapped_diff}")
        # print(f"Loss: {loss}")

        return loss


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Main Code                                       |
# └───────────────────────────────────────────────────────────────────────────┘

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



    wandb.login()  # Add this before wandb.init()

    # Initialize wandb
    wandb.init(project="inverse_model_regression", config={
        "learning_rate": 0.00001,
        "epochs": 500,
        "batch_size": 32,
        "optimizer": "adam",  # Can be varied in sweep
        "loss_function": "L1Loss",  # Can be varied in sweep
        "normalization": "Manual",  # Can be varied in sweep
        "dropout": 0.4,  # Can be varied in sweep
        "patience": 15, # Patience for early stopping
        "dataset": dataset_name,
        "learning_rate_patience": 7,
        "global_feature_max": global_feature_max,
        "global_feature_min": global_feature_min
    })

    # Get last Git commit hash and message
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    commit_msg = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip()

    # Log to config or as a custom field
    wandb.config.update({
        "git_commit": commit_hash,
        "git_message": commit_msg
    })


    # Initialize dataset and data loaders with manual normalization
    train_dataset = FolderHDF5Data(features_file, labels_file, 'Features', 'Labels', 'Train',
                                       global_feature_min, global_feature_max, global_label_min, global_label_max)
    val_dataset = FolderHDF5Data(features_file, labels_file, 'Features', 'Labels', 'Test',
                                     global_feature_min, global_feature_max, global_label_min, global_label_max)


    # Initialize dataset and data loaders
    # train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

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

    # See samples (for debugging)
    # plot_samples_with_annotations('train',train_loader, num_samples=2, plot_dir="plots")

    # Initialize model
    model = OurVgg16().to(device)
    # model = FCNVGG16(input_channels=features_channels, output_channels=labels_channels,
    #                  dropout=wandb.config.dropout).to(device)

    # Log model name
    wandb.config.update({"model": model.__class__.__name__})

    # Watch gradients
    wandb.watch(model, log="all", log_freq=100)

    # Select the optimizer
    if wandb.config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-5)
    elif wandb.config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=0.9)


    # Select the loss function
    if wandb.config.loss_function == "L1Loss":
        print("Using L1")
        criterion = nn.L1Loss()
    elif wandb.config.loss_function == "AngularL1":
        print("Using Angular L1")
        criterion = AngularL1Loss()

    else:
        print("no criterion found. using MSE")
        criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=wandb.config.learning_rate_patience)


    # Run the training
    if train =='yes':
        # Remove old files
        if os.path.exists('inverse_best_model.pth'):
            os.remove('inverse_best_model.pth')
            print("Deleting Old Model...")
        print("Training Model...")
        trained_model, training_log = train_model(model, train_loader, val_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, patience=wandb.config.patience, num_epochs=wandb.config.epochs)

        # Save trained model
        torch.save(trained_model.state_dict(), save_model_path)
        print("Model saved to..." + save_model_path)

        # Upload to wandb
        artifact = wandb.Artifact('trained-model', type='model')
        artifact.add_file(save_model_path)
        wandb.log_artifact(artifact)


    elif train == 'load':
        print("Loading Pre-trained Model... " + load_model_path)
        model.load_state_dict(torch.load(load_model_path))

        trained_model = model
    else:
        print('not loading or training')


    # Evaluate the model
    trained_model.eval()
    val_loss, all_labels_flat, all_predictions_flat = evaluate_model(trained_model, val_loader, criterion,
                                                                          plot_dir="plots")

    # Save scatter plot and random samples
    scatter_plot_path = f"inverse_scatter_plot.png"
    random_samples_path = f"inverse_random_samples.png"


    plot_scatter_plot(all_labels_flat, all_predictions_flat, save_path=scatter_plot_path)
    show_random_samples(trained_model, val_dataset, num_samples=6, save_path=random_samples_path)


    wandb.finish()

