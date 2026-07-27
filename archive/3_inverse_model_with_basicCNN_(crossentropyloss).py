
# Import necessary libraries
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from torchsummary import summary
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import torch.nn.functional as F
import time
import copy
import joblib
import datetime

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""## **Information to set MANUALLY:**"""

# Set starting information
num_epochs = 200
learning_rate = 0.005  # Adjust the learning rate as needed
weight_decay = 1e-4  # Regularization strength
early_stopping_patience = 10  # Number of epochs with no improvement after which training will be stopped
patience_counter = 0



og_dataset_name = '14'
dataset_name = '14_MaxCV_CNN'
train = 'yes' #If you want to load previously trained model for evaluation - set to 'no' and correct the load_model_path


patches = '_Patches'

dataset_dir = "C:/Gal_Msc/Dataset/" + og_dataset_name + '/'


features_dir = dataset_dir + '/Features'
labels_dir = dataset_dir + '/Labels'

cols = 1
rows = 1
pixels_per_patch_row = 5
pixels_per_patch_col = 5



H = cols * pixels_per_patch_col  #Height of images
W = rows * pixels_per_patch_row  #Width of images

P = H * W  #Pixel size of images
C = 3  #Number of channels

# Set dataset files
features_file = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + og_dataset_name + '/' + dataset_name + '_Features' + patches + '.h5'
labels_file = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + og_dataset_name + '/' + dataset_name + '_Labels' + patches + '.h5'

# Define the path and name for saving the model

current_date = datetime.datetime.now().strftime("%Y%m%d")
model_name = f"{dataset_name}{patches}_{current_date}.pkl"

save_model_path = 'C:/Gal_Msc/Ipublic-repo/inverse-model-frustrated-composites/saved_model/CrossEntropyLoss/' + model_name
load_model_path = 'C:/Gal_Msc/Ipublic-repo/inverse-model-frustrated-composites/saved_model/CrossEntropyLoss/' + ''

# prompt: check files exist

import os

print(features_file)
print(labels_file)

features_exist = os.path.isfile(features_file)
labels_exist = os.path.isfile(labels_file)

if features_exist and labels_exist:
    print("Both features and labels files exist.")
else:
    if not features_exist:
        print("Features file does not exist.")
    if not labels_exist:
        print("Labels file does not exist.")

# Define your original labels
original_labels = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

# Create a mapping from original labels to contiguous range
label_to_index = {label: index for index, label in enumerate(original_labels)}
index_to_label = {index: label for index, label in enumerate(original_labels)}


def map_label(labels, mapping):
    return [mapping[label] for label in labels]


"""# **Inspect Data**

* Before transforming to tensors I want to inspect the dataframe data by turning it into images of the original and the patches. This part includes:
1. A custom class for features and labels
2. Instatiating the dataset and dataloader
3. A function for showing one label and it's patches from the original data
4. A function for showing one features and it's patches from the original data
5. A function for ploting one label/feature from a shaped ndarray, used to display a batch later on.
6. A function for convert a tensor back to a numpy array

### **A Custom Class for Features and Labels**
"""


class FolderHDF5Data(Dataset):
    def __init__(self, features_file, labels_file, feature_main_group, label_main_group, category):
        """
        Initialize the dataset with the paths to the features and labels HDF5 files,
        the main groups ('Features' and 'Labels'), and the category ('Train' or 'Test').

        Args:
            features_file (str): Path to the features HDF5 file.
            labels_file (str): Path to the labels HDF5 file.
            feature_main_group (str): Main group within the features HDF5 file ('Features').
            label_main_group (str): Main group within the labels HDF5 file ('Labels').
            category (str): Subgroup within the main group ('Train' or 'Test').
        """
        self.features_file = features_file
        self.labels_file = labels_file
        self.feature_main_group = feature_main_group
        self.label_main_group = label_main_group
        self.category = category
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

            # Map the label to a contiguous range
            if label in label_to_index:
                mapped_label = label_to_index[label]

            else:
                raise ValueError(f"Label '{label}' not found in label_to_index")

            # Transform the feature and the label
            feature_tensor, label_tensor = data_transform(feature, mapped_label)

            return feature_tensor, label_tensor


import torchvision.transforms as transforms

# Create a transform function that treats the features and labels separately.
feature_transform = transforms.Compose([
    transforms.ToTensor(),
])

label_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))


def data_transform(feature, label):
    """
    Applies separate transformations to feature and label data.
    """

    if isinstance(feature, pd.DataFrame):
        feature = feature.values


    feature_tensor = feature_transform(feature)
    if isinstance(label, pd.DataFrame):
        label = label.values[0, 0]  # Assuming label is in the first cell
    label_tensor = label_transform(label)

    return feature_tensor, label_tensor


"""# **Instantiate Dataset and Data Loader**"""

train_dataset = FolderHDF5Data(features_file, labels_file, 'Features', 'Labels', 'Train')
test_dataset = FolderHDF5Data(features_file, labels_file, 'Features', 'Labels', 'Test')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)



# Iterate through the DataLoader manually
for i, (features, labels) in enumerate(train_loader):
    if i == 5:  # Display only the first 5 samples
        break
    print(f"Feature batch shape: {features.shape}")
    print(f"Label batch: {labels}")

import matplotlib.pyplot as plt


def display_batch_features_and_labels(dataloader):
    features, labels = next(iter(dataloader))
    batch_size = features.size(0)

    fig, axs = plt.subplots(batch_size, 2, figsize=(10, batch_size * 2.5))

    for i in range(batch_size):
        feature = features[i].numpy().transpose(1, 2, 0)  # Convert to HWC format for plotting
        label = labels[i].item()  # Convert tensor to integer

        # Normalize the feature to the range [0, 1]
        feature = (feature - feature.min()) / (feature.max() - feature.min())

        # Display the feature (reshaped to 5x5 image)
        try:
            axs[i, 0].imshow(feature.reshape((pixels_per_patch_row, pixels_per_patch_col, C)))
            axs[i, 0].axis('off')
            axs[i, 0].set_title(f'Feature {i + 1}')
        except:
            continue

        # Display the label as text
        axs[i, 1].text(0.5, 0.5, str(label), fontsize=18, ha='center')
        axs[i, 1].axis('off')
        axs[i, 1].set_title(f'Label {i + 1}')

    plt.tight_layout()
    plt.show()


# Example usage
# Assuming you have a DataLoader named 'train_loader'
display_batch_features_and_labels(train_loader)

"""## Display a bunch of features

# Create the Model
"""


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        super(MobileNetV3, self).__init__()
        if pretrained:
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        else:
            self.model = models.mobilenet_v3_large(weights=None)

        # Replace the classifier with a new one for our specific number of classes
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class BasicCNN(nn.Module):
    def __init__(self, num_classes=12):  # Set num_classes based on your dataset
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout = nn.Dropout(0.5)  # Dropout with a 50% rate

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)  # num_classes output neurons for classification

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # Output size: (32, 5, 5)
        x = self.pool(self.relu(self.conv2(x)))  # Output size: (64, 2, 2)
        x = self.dropout(x)  # Apply dropout after pooling
        x = self.relu(self.conv3(x))  # Output size: (128, 2, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout before the final layers
        x = self.fc2(x)  # Output logits for each class
        return x



num_classes = 12  # Change this to the number of classes in your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicCNN(num_classes = num_classes)
model = model.to(device)
# Print the type of model used
print(f"Model moved to device: {type(model).__name__}")



"""###          Training and predicting Funciton

"""

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Optimizer (Adam)

# Set up the learning rate scheduler and early stopping parameters
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
best_val_loss = float('inf')

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    global best_val_loss, patience_counter
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        start_time = time.time()

        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss = validate_model(model, val_loader, criterion)

        scheduler.step(val_loss)  # Step the scheduler based on validation loss

        # Print the learning rate
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

        end_time = time.time()
        epoch_time = end_time - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f} seconds")

    model.load_state_dict(best_model_weights)
    return model


# Predict function
def predict(model, input_tensor):
    """
    Make a prediction with the model.

    Args:
    model (torch.nn.Module): The trained model ready for making predictions.
    input_tensor (torch.Tensor): The input data tensor. Should be preprocessed as per the model's training.

    Returns:
    torch.Tensor: The model's prediction output.
    """
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Ensure no gradients are calculated
        output = model(input_tensor)

    return output


# Function to validate the model
def validate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(dataloader)
    return val_loss


"""# Train the Model"""

# Run the training
if train =='yes':
    print("Training Model")
    model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs)

    # Save trained model
    torch.save(model.state_dict(), save_model_path)
    print("Model saved to..." + save_model_path)
else:
    print("Loading Pre-trained Model... " + load_model_path)
    model.load_state_dict(torch.load(load_model_path))
    model.eval()  # Set the model to evaluation mode


print()

"""#Display Results"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode

    # Ensure the model is on the right device
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device).float()  # Ensure inputs are in float
            labels = labels.to(device).long()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Extract unique classes
    unique_classes = np.unique(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               labels=unique_classes)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=unique_classes)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("\nClassification Report:\n",
          classification_report(all_labels, all_preds, target_names=[str(index_to_label[i]) for i in unique_classes]))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[index_to_label[i] for i in unique_classes],
                yticklabels=[index_to_label[i] for i in unique_classes])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def visualize_predictions(model, dataloader, num_samples=10):
    model.eval()  # Set the model to evaluation mode

    # Ensure the model is on the right device
    model.to(device)

    # Store features, ground truth labels, and predicted labels
    features_list = []
    ground_truth_labels = []
    predicted_labels = []

    # Get random samples
    sample_indices = np.random.choice(len(dataloader.dataset), num_samples, replace=False)

    for idx in sample_indices:
        feature, true_label = dataloader.dataset[idx]
        feature = feature.unsqueeze(0).to(device).float()  # Add batch dimension at the beginning
        true_label = torch.tensor(true_label).to(device)

        with torch.no_grad():
            predicted_label = model(feature).argmax(dim=1)

        # Normalize feature for visualization
        feature = feature.cpu().squeeze().numpy()
        feature = (feature - feature.min()) / (feature.max() - feature.min())  # Normalize to [0, 1]

        # Reshape to 5x5 RGB image
        feature = feature.reshape((pixels_per_patch_row, pixels_per_patch_col, C))

        features_list.append(feature)
        ground_truth_labels.append(index_to_label[true_label.cpu().item()])
        predicted_labels.append(index_to_label[predicted_label.cpu().item()])

    # Plot the features, ground truth labels, and predicted labels
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 2.5))

    for i in range(num_samples):
        feature = features_list[i]
        true_label = ground_truth_labels[i]
        predicted_label = predicted_labels[i]

        axes[i, 0].imshow(feature)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Feature {i + 1}')

        axes[i, 1].text(0.5, 0.5, str(true_label), fontsize=18, ha='center')
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'Label {i + 1}')

        axes[i, 2].text(0.5, 0.5, str(predicted_label), fontsize=18, ha='center')
        axes[i, 2].axis('off')
        axes[i, 2].set_title(f'Prediction {i + 1}')

    plt.tight_layout()
    plt.show()


# Visualize predictions
visualize_predictions(model, test_loader, num_samples=10)


# Evaluate the model
evaluate_model(model, test_loader)





