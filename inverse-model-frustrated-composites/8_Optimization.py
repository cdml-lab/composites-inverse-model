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

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Definitions                                     │
# └───────────────────────────────────────────────────────────────────────────┘


# Input Files
model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\30-35_Curvature_No_Length_20241105.pkl"
excel_file_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\rhino_to_model_inverse.xlsx"

features_channels = 1
labels_channels = 3

# Normalization Aspect
global_labels_min = 0.0
global_labels_max = 180.0
global_features_min = -1.0
global_features_max = 1.0

# Optimization loop
max_iterations = 10000
desired_threshold = 0.001
visualize = True
print_steps = 3000 # Once in how many steps to print the prediction


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
        raise ValueError(f"Unexpected data shape {data.shape}, expected (300, 4)")

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


def visualize_curvature_tensor(tensor):
    # Remove the batch dimension
    tensor = tensor.squeeze(0)

    # Check if the tensor has 4 channels
    if tensor.shape[0] != labels_channels:
        raise ValueError(f"Expected tensor with shape [1, {labels_channels}, 20, 15]")

    # Set up a 2x2 grid for visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Tensor Visualization ({labels_channels} Channels)")

    for i in range(labels_channels):
        # Get the channel and display it in the respective subplot
        channel = tensor[i].cpu().detach().numpy()  # Move to CPU and convert to NumPy if needed
        ax = axes[i // 2, i % 2]
        ax.imshow(channel, cmap="viridis", aspect=0.85)
        ax.set_title(f"Channel {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top to fit the suptitle
    plt.show()


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
    # Initialize an empty array to store angles, same shape as the input array except for the last dimension
    angle_array = np.zeros((vectors.shape[0], vectors.shape[1], 1))

    # Iterate over each vector in the (20, 15, 3) array
    for i in range(vectors.shape[0]):  # Iterate over rows (20)
        for j in range(vectors.shape[1]):  # Iterate over columns (15)
            x, y, z = vectors[i, j]  # Extract the vector (x, y, z)
            angle_array[i, j, 0] = angle_with_x_axis(x, y, z)  # Calculate the angle and store it

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

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Main Code                                       │
# └───────────────────────────────────────────────────────────────────────────┘


print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the surface to optimize from Excel
input_tensor, vector_df = excel_to_np_array(file_path=excel_file_path, sheet_name='Sheet1',
                                 global_features_max=global_features_max, global_features_min=global_features_min)

input_tensor = input_tensor.to(device)
print_tensor_stats(input_tensor)
if visualize:
    visualize_curvature_tensor(input_tensor)

# Define the Model
model = OurModel()
model.load_state_dict(torch.load(model_path))
model.to(device)

# ┌───────────────────────────────────────────────────────────────────────────┐
# │       Optimization Process - Matching Predicted Output to Excel Data      │
# └───────────────────────────────────────────────────────────────────────────┘


print(f"vector array  shape: {vector_df.shape}")
angles = calculate_angles(vector_df)
print(f"calculates angles array shap {angles.shape}")
print(angles)


average_patches_np = average_patches(angles, (5,5),
                                     r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\Optimization debug_Original.xlsx",
                                     r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\Optimization debug_Average.xlsx")
print("averages array shape: ", average_patches_np.shape)
num_of_patches = average_patches_np.shape[0] * average_patches_np.shape[1]
print( f"number of patches: {num_of_patches}")


# Step 1: Convert the NumPy array to a PyTorch tensor
average_patches_tensor = torch.tensor(average_patches_np, dtype=torch.float32)

# Step 2: Normalize the tensor by dividing by 180
normalized_tensor = average_patches_tensor / 180.0
print(f"shape of average degrees array {normalized_tensor.shape}")

# Export to excel for debugging purposes
fiber_orientation_to_excel(normalized_tensor, global_labels_max, "initial_fiber_orientations.xlsx")


# Clone and detach the tensor, then set requires_grad to True
initial_fiber_orientation = normalized_tensor.clone().detach().requires_grad_(True)
print(f"Initial before duplication {initial_fiber_orientation}")


loss_fn = nn.L1Loss()

# Define the optimizer - examples of different optimizers
optimizer_name = 'adam'  # Change this to switch between optimizers

if optimizer_name == 'adam':
    optimizer = optim.Adam(params=[initial_fiber_orientation], lr=0.005)
elif optimizer_name == 'sgd':
    optimizer = optim.SGD(params=[initial_fiber_orientation], lr=0.01, momentum=0.9)
elif optimizer_name == 'rmsprop':
    optimizer = optim.RMSprop(params=[initial_fiber_orientation], lr=0.01, alpha=0.99)
elif optimizer_name == 'adagrad':
    optimizer = optim.Adagrad(params=[initial_fiber_orientation], lr=0.001)



for step in range(max_iterations):
    optimizer.zero_grad()

    #Duplicate data for prediction (from 4x3 to 20x15)
    duplicate_fiber_orientation = duplicate_pixel_data(initial_fiber_orientation).to(device)

    # Forward pass
    predicted = model(duplicate_fiber_orientation)

    # Print every x steps
    if step % print_steps == 0:
        print("Fiber Orientation Tensor:")
        print_tensor_stats(duplicate_fiber_orientation)

        print("Predicted Tensor:")
        print_tensor_stats(predicted)
        visualize_curvature_tensor(predicted)

        print(f"duplicate_fiber_orientation: {duplicate_fiber_orientation}")

    # Compute loss
    # loss = sine_cosine_embedding_l1_loss(predicted, input_tensor)
    loss = loss_fn(predicted, input_tensor)

    # print((predicted.size()))
    # print(input_tensor.size())
    loss.backward()

    # Calculate and print gradient statistics
    grad_size = initial_fiber_orientation.grad
    if grad_size is not None:
        max_grad = round(grad_size.max().item(), 4)  # Maximum gradient rounded to 4 decimal places
        min_grad = round(grad_size.min().item(), 4)  # Minimum gradient rounded to 4 decimal places
        mean_grad = round(grad_size.mean().item(), 4)  # Mean gradient rounded to 4 decimal places

        print(f'Gradient stats - Max: {max_grad}, Min: {min_grad}, Mean: {mean_grad}')

    # Scale gradients for faster convergence
    # initial_fiber_orientation.grad *= 1  # Scale gradients

    optimizer.step()

    # Clamp `initial_fiber_orientation` to stay within [0, 1]
    with torch.no_grad():
        initial_fiber_orientation.clamp_(0, 1)


    # Print the loss for the current step
    print(f'Step {step + 1}, Loss: {loss.item()}')

    if loss.item() < desired_threshold:
        print('Desired threshold reached. Stopping optimization.')
        break


# Convert the optimized fiber orientation tensor to a 2D DataFrame and save to Excel
final_fiber_orientation_final = initial_fiber_orientation.detach()

fiber_orientation_to_excel(final_fiber_orientation_final, global_labels_max)

print("Optimization complete. Result saved to Excel.")
