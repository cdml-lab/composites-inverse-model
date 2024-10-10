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
excel_file_path = r"C:\Users\User\OneDrive - Technion\Documents\GitHub\public-repo\inverse-model-frustrated-composites\saved_models_for_checks\test\test1_reshaped.h5"
model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\17-24_Location_Features_RareDragon220.pkl"

# Define parameters
features_channels = 1
labels_channels = 3
added_features = 0

# Initialize wandb
wandb.init(project="test_xyz_prediction", mode="disabled")

class OurModel(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(OurModel, self).__init__()

        self.conv_1 = torch.nn.Conv2d(in_channels=features_channels + added_features, out_channels=32, kernel_size=3, padding=1)
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

# Load and preprocess Excel data
def load_excel_data(file_path):
    # Specify the engine explicitly to avoid format detection issues
    df = pd.read_excel(file_path, engine='openpyxl')  # Use 'xlrd' for .xls files
    return df

# Visualization function
def visualize_xyz(points_xyz, step, plot_name="XYZ Visualization"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_xyz[:, :, 0], points_xyz[:, :, 1], points_xyz[:, :, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(f'XYZ Points at Step {step}')

    # Log the figure to WandB
    wandb.log({plot_name: wandb.Image(fig)})

    plt.show()
    plt.close()

def create_random_sample():
    # Initialize the fiber orientation with 12 distinct orientations for the 3x4 patches
    random_orientations = torch.randint(0, 181, (3, 4), dtype=torch.float32)

    # Create the (20, 15) grid by repeating the 12 values across the 5x5 patches
    initial_fiber_orientation = torch.zeros((1, 1, 20, 15), dtype=torch.float32)

    # Loop over 3x4 patches and fill the (20x15) grid
    for i in range(3):
        for j in range(4):
            # Assign the random orientation to the corresponding 5x5 patch
            initial_fiber_orientation[:, 0, i * 5:(i + 1) * 5, j * 5:(j + 1) * 5] = random_orientations[i, j]

    return initial_fiber_orientation

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

def export_each_channel_to_excel(prediction_tensor, base_save_path="predictions_channel"):
    """
    Exports each channel from a tensor of shape (1, channels, height, width) to separate Excel files.

    :param prediction_tensor: A tensor of shape (1, channels, height, width) containing prediction data.
    :param base_save_path: Base path for saving the Excel files. A suffix with the channel number will be added.
    """
    # Remove the batch dimension by squeezing the tensor to (channels, height, width)
    if prediction_tensor.ndim == 4 and prediction_tensor.shape[0] == 1:
        prediction_tensor = prediction_tensor.squeeze(0)

    # Check that the tensor now has the shape (channels, height, width)
    if prediction_tensor.ndim != 3 or prediction_tensor.shape[0] != 3:
        raise ValueError("Expected tensor of shape (1, 3, height, width)")

    # Ensure tensor is on CPU and convert to NumPy
    prediction_np = prediction_tensor.cpu().numpy()

    for i in range(prediction_np.shape[0]):
        channel_data = prediction_np[i, :, :]  # Shape (20, 15) for each channel
        df = pd.DataFrame(channel_data)

        # Define a unique filename for each channel
        save_path = f"{base_save_path}_channel_{i + 1}.xlsx"
        df.to_excel(save_path, index=False, sheet_name=f"Channel_{i + 1}", header=False)

        print(f"Channel {i + 1} exported to {save_path}")


# data = load_excel_data(excel_file_path)
data = create_random_sample()
print(data)
print(data.size())
orientation_array = data
data = data / 180 # Normalize
print(data)

# Convert the tensor to a NumPy array and remove the extra dimensions

orientation_array = orientation_array.squeeze().numpy()

# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(orientation_array)

# Export the DataFrame to an Excel file
df.to_excel("fiber_orientation.xlsx", index=False, header=False)


# Load model
model = OurModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Make prediction
with torch.no_grad():
    predicted_xyz = model(data)
    print(predicted_xyz.size())

# Original min and max for denormalization (adjust these values as needed)
global_label_min = [-9.52, -9.52, -9.52]  # Replace with actual minimum values for each channel
global_label_max = [9.52, 9.52, 9.52]  # Replace with actual maximum values for each channel

# Global Label Min for channel 0: -7.052006
# Global Label Max for channel 0: 7.055874
# Global Label Min for channel 1: -9.528751
# Global Label Max for channel 1: 9.528675
# Global Label Min for channel 2: -1.845271
# Global Label Max for channel 2: 2.62029

# Denormalize predictions
predicted_xyz_denorm = predicted_xyz.clone()  # Clone to avoid modifying the original tensor
for c in range(labels_channels):
    predicted_xyz_denorm[:, c, :, :] = predicted_xyz_denorm[:, c, :, :] * (global_label_max[c] - global_label_min[c]) + global_label_min[c]


export_each_channel_to_excel(predicted_xyz_denorm)

predicted_xyz_denorm = torch.permute(predicted_xyz_denorm, (2, 3, 1, 0))
print(f"after permute{predicted_xyz_denorm.size}")


predicted_xyz_np = predicted_xyz_denorm.squeeze().numpy()  # Convert to NumPy for plotting

print(f"after numpy{np.shape(predicted_xyz_np)}")

# Assuming predicted_xyz is generated from your model output
visualize_xyz(predicted_xyz_np, step=0, plot_name="Predicted XYZ Visualization")

visualize_xyz_channels_2d(predicted_xyz_np, step=0)



wandb.finish()


