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
import segmentation_models_pytorch as smp
from torchvision.transforms.functional import resize
import torchvision.transforms.functional as TF
from torchvision.models import resnet34
from segmentation_models_pytorch import DeepLabV3
import torch.nn.functional as F

seed = 42  # Set the seed for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


# Set variables

## Set dataset name
og_dataset_name="171920"
dataset_name="171920_MV"

# Set dataset files.

# PAY ATTENTION: since this is a forward models the files are flipped and the labels file will be the original features
# file! and the same foe feature will be the original labels file, meant for in inverse model.
features_file = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + og_dataset_name + '/' + dataset_name + '_Labels_Reshaped.h5'
labels_file = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + og_dataset_name + '/' + dataset_name + '_Features_Reshaped.h5'

# Define the path and name for saving the model
current_date = datetime.datetime.now().strftime("%Y%m%d")
model_name = f"{dataset_name}_{current_date}.pkl"

save_model_path = 'C:/Gal_Msc/Ipublic-repo/inverse-model-frustrated-composites/saved_model/Forward/' + model_name
load_model_path = 'C:/Gal_Msc/Ipublic-repo/inverse-model-frustrated-composites/saved_model/Forward/' + model_name

features_channels = 1
labels_channels = 3

train = 'no' #If you want to load previously trained model for evaluation - set to 'no' and correct the load_model_path
train_arch = 'yes'
model_type ='ourmodel'
is_random = 'no'

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
#    print(f"Global Feature Min: {global_feature_min}")
 #   print(f"Global Feature Max: {global_feature_max}")
  #  for i in range(len(global_label_min)):
   #     print(f"Global Label Min for channel {i}: {global_label_min[i]}")
    #    print(f"Global Label Max for channel {i}: {global_label_max[i]}")

    return global_feature_min, global_feature_max, global_label_min, global_label_max

# Custom Class of Data
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

# Transform. doesn't currently resize.
def data_transform(feature, label, global_feature_min, global_feature_max, global_label_min, global_label_max):
    """
    Transform the feature and label data into the required format for the model.

    Args:
        feature (np.ndarray): The feature data array with shape (height, width, channels).
        label (np.ndarray): The label data array with shape (height, width, channels).
        global_feature_min (float): Global minimum value for feature normalization.
        global_feature_max (float): Global maximum value for feature normalization.
        global_label_min (list of float): Global minimum values for each label channel normalization.
        global_label_max (list of float): Global maximum values for each label channel normalization.

    Returns:
        tuple: Transformed feature and label tensors with shape:
               - feature_tensor: (channels, padded_height, padded_width)
               - label_tensor: (channels, padded_height, padded_width)
    """
    # Convert feature data to tensor
    feature_tensor = torch.tensor(feature, dtype=torch.float32)
    # Normalize feature data using global min and max
    feature_tensor = (feature_tensor - global_feature_min) / (global_feature_max - global_feature_min)
    # Reorder dimensions: from (height, width, channels) to (channels, height, width)
    feature_tensor = feature_tensor.permute(2, 0, 1).float()  # Should be (1, 15, 20)

    # Convert label data to tensor
    label_tensor = torch.tensor(label, dtype=torch.float32)
    # Normalize each channel of the label data using corresponding global min and max values
    for c in range(label_tensor.shape[2]):  # Assuming channels are the last dimension in the original label
        label_tensor[:, :, c] = (label_tensor[:, :, c] - global_label_min[c]) / (global_label_max[c] - global_label_min[c])
    # Reorder dimensions: from (height, width, channels) to (channels, height, width)
    label_tensor = label_tensor.permute(2, 0, 1).float()  # Should be (3, 15, 20)

    return feature_tensor, label_tensor

class OurModel(torch.nn.Module):

    def __init__(self):
        super(OurModel, self).__init__()

        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
        self.conv_3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_4 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv_5 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv_6 = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)

        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=8)
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=32)
        self.batch_norm_3 = torch.nn.BatchNorm2d(num_features=64)
        self.batch_norm_4 = torch.nn.BatchNorm2d(num_features=32)
        self.batch_norm_5 = torch.nn.BatchNorm2d(num_features=16)
        self.dropout = torch.nn.Dropout(p=0.5)

        # self.batch_norm_6 = torch.nn.BatchNorm2d(num_features=1)

        self.relu = torch.nn.ReLU()

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

        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.relu(x)

        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.relu(x)

        x = self.conv_6(x)

        return x

# Train function. set the epochs and patience here.
import torch.optim as optim

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200, patience=12):
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
            loss.backward()
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
        scheduler.step(val_loss)  # Step the scheduler

        end_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs} | Time: {end_time - start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        training_log.append((epoch + 1, train_loss, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {patience} epochs of no improvement.')
            early_stop = True
            break

    if early_stop:
        print("Loading best model from checkpoint...")
        model.load_state_dict(torch.load('best_model.pth'))

    return model, training_log


# Evaluation Function
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

            all_predictions.append(outputs.cpu().numpy())  # Move data to CPU before converting to numpy
            all_labels.append(labels.cpu().numpy())  # Move data to CPU before converting to numpy

    val_loss /= len(val_loader)

    # Calculate errors and plot histogram
    errors = np.concatenate(all_predictions, axis=0).flatten() - np.concatenate(all_labels, axis=0).flatten()
    plot_error_histogram(errors, plot_dir=plot_dir)

    # Plot heatmaps and quivers for the first few samples
    for i in range(min(3, len(all_predictions))):
        actual_sample = all_labels[i][0]  # Take the first sample in the batch
        predicted_sample = all_predictions[i][0]  # Take the first sample in the batch

        plot_heatmaps(actual_sample, predicted_sample, sample_index=i + 1, plot_dir=plot_dir)
        plot_quiver(actual_sample, predicted_sample, sample_index=i + 1, plot_dir=plot_dir)

    print(f'Validation Loss: {val_loss:.4f}')

    # Flatten the predictions and labels for the scatter plot
    all_predictions_flat = np.concatenate(all_predictions, axis=0)
    all_labels_flat = np.concatenate(all_labels, axis=0).flatten()

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
                          0.5 * abs_diff**2,
                          self.delta * (abs_diff - 0.5 * self.delta))
        return loss.mean()

class CauchyLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, input, target):
        x = torch.abs(input - target) / self.delta
        loss = self.delta * torch.log(1 + x**2)
        return loss.mean()

class TukeyBiweightLoss(nn.Module):
    def __init__(self, c=4.685):
        super().__init__()
        self.c = c

    def forward(self, input, target):
        x = torch.abs(input - target) / self.c
        x = torch.clamp(x, min=0, max=1)
        loss = self.c**2 * (1 - (1 - x**2)**3) / 6
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

def show_random_samples(model, dataset, num_samples=6, save_path="random_samples.png"):
    """
    Display random samples from the dataset with their features, ground truth labels,
    and predicted labels. All are shown as 20x15 pixel images.

    Args:
        model (nn.Module): Trained model to generate predictions.
        dataset (Dataset): Dataset object to sample data from.
        num_samples (int): Number of random samples to display.
        save_path (str): Path to save the plot.
    """
    model.eval()

    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    fig.suptitle('Features, Ground Truth, and Predictions', fontsize=16)

    for i in range(num_samples):
        if is_random=='yes':
            idx = random.randint(0, len(dataset) - 1)
        else:
            idx = i
        feature_tensor, label_tensor = dataset[idx]

        # Generate prediction
        with torch.no_grad():
            prediction_tensor = model(feature_tensor.unsqueeze(0).to(device)).squeeze(0)

        # Convert tensors to numpy arrays
        feature_img = feature_tensor.permute(1, 2, 0).cpu().numpy()
        label_img = label_tensor.permute(1, 2, 0).cpu().numpy()
        prediction_img = prediction_tensor.permute(1, 2, 0).cpu().numpy()

        # Normalize for visualization
        feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
        label_img = (label_img - label_img.min()) / (label_img.max() - label_img.min())
        prediction_img = (prediction_img - prediction_img.min()) / (prediction_img.max() - prediction_img.min())

        # Display images
        axs[i, 0].imshow(feature_img)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Sample {i + 1} - Feature')

        axs[i, 1].imshow(label_img)
        axs[i, 1].axis('off')
        axs[i, 1].set_title(f'Sample {i + 1} - Ground Truth')

        axs[i, 2].imshow(prediction_img)
        axs[i, 2].axis('off')
        axs[i, 2].set_title(f'Sample {i + 1} - Prediction')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(save_path)  # Save the figure as an image file
    plt.close()
    print(f"Random samples saved to {save_path}")

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
        label_img = label.permute(1, 2, 0).cpu().numpy()

        # Normalize for visualization
        feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
        label_img = (label_img - label_img.min()) / (label_img.max() - label_img.min())

        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(f'Sample {i + 1} - Features and Labels with Annotations', fontsize=16)

        axs[0].imshow(feature_img)
        axs[0].axis('off')
        axs[0].set_title('Features')

        axs[1].imshow(label_img)
        axs[1].axis('off')
        axs[1].set_title('Labels')

        # Annotate each 5x5 pixel block
        for y in range(0, feature_img.shape[0], 5):
            for x in range(0, feature_img.shape[1], 5):
                feature_text = f"{feature_img[y, x, 0]:.2f}"
                label_text = f"{label_img[y, x, 0]:.2f}, {label_img[y, x, 1]:.2f}, {label_img[y, x, 2]:.2f}"
                axs[0].text(x, y, feature_text, fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5))
                axs[1].text(x, y, label_text, fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5))

        img_path = os.path.join(plot_dir, f"debug_sample_{loader_type}_{i + 1}.png")
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

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Training log plot saved to {plot_path}")


# Test Architectures:
def create_model(architecture):
    layers = []
    in_channels = 1  # Assuming input has 1 channel
    for out_channels in architecture:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1))  # Final layer to match output channels
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


    # Calculate global min and max values for normalization
    global_feature_min, global_feature_max, global_label_min, global_label_max = calculate_global_min_max(features_file, labels_file, 'Labels', 'Features')

    # Initialize dataset and data loaders
    # PAY ATTENTION: the labels and feature files are flipped on purpose! because this is a forward model and the files are bult for inverse
    train_dataset = FolderHDF5Data(features_file, labels_file, 'Labels', 'Features', 'Train',
                                   global_feature_min, global_feature_max, global_label_min, global_label_max)
    val_dataset = FolderHDF5Data(features_file, labels_file, 'Labels', 'Features', 'Test',
                                 global_feature_min, global_feature_max, global_label_min, global_label_max)



    # Initialize dataset and data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)


    # See samples(for debugging)
    #plot_samples_with_annotations('train',train_loader, num_samples=3, plot_dir="plots")
    #plot_samples_with_annotations('validation',val_loader, num_samples=3, plot_dir="plots")



    # Initialize model
    if model_type == 'simplecnn':
        print(f"model selected {model_type}")
        model = SimpleCNN(features_channels, labels_channels).to(device)
    elif model_type =='unet':
        print(f"model selected {model_type}")
        # Initialize the pre-built U-Net model
        model = smp.Unet(
            encoder_name="resnet34",  # Choose encoder, e.g., resnet34, mobilenet_v2, etc.
            encoder_weights="imagenet",  # Use `imagenet` pre-trained weights for the encoder
            in_channels=features_channels,  # Model input channels (1 for grayscale images)
            classes=labels_channels  # Model output channels (number of classes for segmentation)
        ).to(device)
    elif model_type =='deeplab':
        print(f"model selected {model_type}")
        model = DeepLabV3(
            encoder_name="resnet34",  # Choose encoder, e.g., resnet34, mobilenet_v2, etc.
            encoder_weights="imagenet",  # Use `imagenet` pre-trained weights for the encoder
            in_channels=features_channels,  # Model input channels (1 for grayscale images)
            classes=labels_channels  # Model output channels (number of classes for segmentation)
        ).to(device)
    elif model_type == 'resnet34':
        print(f"model selected {model_type}")
        resnet = resnet34(pretrained=True)
        resnet.conv1 = nn.Conv2d(features_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, labels_channels * 224 * 224),  # Adjust based on your label dimensions
            nn.Unflatten(1, (labels_channels, 224, 224))  # Unflatten the output to match label dimensions
        )
        model = resnet
        model.to(device)
    elif model_type =='ourmodel':
        print(f"model selected {model_type}")
        model = OurModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    # criterion = nn.MSELoss()
    # criterion = CosineSimilarityLoss()
    criterion = nn.L1Loss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Run the training
    if train =='yes':
        print("Training Model")
        trained_model, training_log = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

        # Save trained model
        torch.save(trained_model.state_dict(), save_model_path)
        print("Model saved to..." + save_model_path)
    elif train == 'load':
        print("Loading Pre-trained Model... " + load_model_path)
        model.load_state_dict(torch.load(load_model_path))
        model.eval()  # Set the model to evaluation mode
        trained_model = model
    else:
        print('not loading or training')


    # Evaluate Performance
    try: evaluate_model(trained_model, val_loader, criterion, plot_dir=plots_dir)
    except: print("couldnt evaluate model")

    try:show_random_samples(trained_model, val_dataset, save_path=f"{plots_dir}/random_samples_{model_type}_{current_date}.png")
    except: print("coudnt show random samples")

    ### Test Architectures

    architectures = [
        [8, 16, 32, 64, 32, 16, 8],  # ResNet-like (Simplified)
        [8, 16, 32, 32, 16, 8],  # MobileNetV2-like (Simplified)
        [8, 16, 32, 64, 128, 64, 32, 16, 8],  # DenseNet-like (Simplified)
        [8, 16, 8],  # Very Shallow Network
        [8, 16, 32, 64, 32, 16, 8],  # Moderately Deep Network
        [8, 16, 32, 64, 128, 64, 32, 16, 8],  # Very Deep Network
        [16, 32, 64, 32, 16],  # Basic Convolutional Network
        [8, 16, 32, 64, 128, 256, 128, 64, 32, 16, 8],  # Increasing Complexity
        [6, 16, 120],  # LeNet-like (Simplified)
        [8, 8, 16, 16, 32, 32, 64, 64],  # VGG-like (Simplified)
        [64, 128, 256, 512, 1024],  # ResNet-like (Full)
        [32, 64, 128, 256, 512],  # MobileNetV2-like (Full)
        [32, 64, 128, 256, 512],  # DenseNet-like (Full)
        [16, 32, 64, 128, 256, 512],  # Very Deep Network (Full)
        [64, 128, 256, 512, 512],  # VGG-like (Full)
        [6, 16, 120, 84]  # LeNet-like (Full)
    ]
    """
    architectures = [
        [8, 16, 32, 64, 32, 16, 8],  # ResNet-like (Simplified)
    ]
    """
    results = []

    for idx, architecture in enumerate(architectures):
        print(f"Training architecture {idx + 1}: {architecture}")
        model = create_model(architecture).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8)
        criterion = TukeyBiweightLoss()
        model_save_path = f"saved_model_{idx + 1}.pth"

        if train_arch=='yes':
            # Train the model
            trained_model, training_log = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

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
        scatter_plot_path = f"scatter_plot_{idx + 1}.png"
        random_samples_path = f"random_samples_{idx + 1}.png"
        residuals_path = f"residuals_{idx + 1}.png"

        try: plot_scatter_plot(all_labels_flat, all_predictions_flat, save_path=scatter_plot_path)
        except: print("could not plot scatter plot")
        try: show_random_samples(trained_model, val_dataset, num_samples=6, save_path=random_samples_path)
        except: print("could not plot random samples")

        # Plot residuals
        plot_residuals(all_predictions_flat, all_labels_flat, save_path=residuals_path)
        #except: print("could not plot residuals")


        # Save training log plot
        training_log_path = f"training_log_{idx + 1}.png"
        try: plot_training_log(training_log, training_log_path)
        except: print("could not plot training log")

        # Record results
        results.append({
            "architecture": architecture,
            "val_loss": val_loss,
            "model_save_path": model_save_path,
            "scatter_plot_path": scatter_plot_path,
            "random_samples_path": random_samples_path,
            "training_log_path": training_log_path
        })


    # Print the results
    import pandas as pd

    results_df = pd.DataFrame(results)
    results_df.to_excel("model_evaluation_results.xlsx", index=False)
    print(results_df)
