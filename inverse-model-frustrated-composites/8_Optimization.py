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

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Definitions                                     │
# └───────────────────────────────────────────────────────────────────────────┘


# Input Files
model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\forward_best_model.pth"
excel_file_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\rhino_to_model_inverse.xlsx"

features_channels = 1
labels_channels = 8

num_of_cols = 3
num_of_rows = 4

channels_to_keep = [0,1,2,3,4]

# Normalization Aspect
global_labels_min = 0.0
global_labels_max = 180.0
global_features_max = [10.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
global_features_min = [-10.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5]

# Optimization loop
max_iterations = 2
desired_threshold = 0.001
visualize = True
print_steps = 3000 # Once in how many steps to print the prediction


optimizer_type = 'nl-opt' # basic, nl-opt


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


def average_patches_for_gradients(data, patch_size):
    """
    Divide the input 2D NumPy array into patches and calculate the average value in each patch.

    Parameters:
        data (np.ndarray): The input 2D array of shape (height, width).
        patch_size (tuple): The size of each patch as (patch_height, patch_width).

    Returns:
        np.ndarray: The averaged patches as a 2D array with shape determined by the input and patch size.
    """
    # Ensure the input is a NumPy array
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    h, w = data.shape  # Original dimensions
    ph, pw = patch_size  # Patch dimensions

    # Ensure input dimensions are divisible by patch size
    assert h % ph == 0, "Height is not divisible by patch height."
    assert w % pw == 0, "Width is not divisible by patch width."

    # Reshape to split into patches and calculate the mean
    reshaped = data.reshape(h // ph, ph, w // pw, pw)
    averaged_patches = reshaped.mean(axis=(1, 3))  # Average over the patch height and width dimensions

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


def visualize_curvature_tensor(tensor, labels_channels):
    # Remove the batch dimension
    tensor = tensor.squeeze(0)

    # Check if the tensor has the correct number of channels
    if tensor.shape[0] != labels_channels:
        raise ValueError(f"Expected tensor with shape [1, {labels_channels}, 20, 15], but got {tensor.shape}")

    # Calculate the grid size needed for visualization
    grid_size = math.ceil(math.sqrt(labels_channels))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 8))
    fig.suptitle(f"Tensor Visualization ({labels_channels} Channels)")

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


def objective_function(x, grad):

    """
    NLopt objective function to minimize the loss between predicted and input tensors.

    Parameters:
    - x (np.ndarray): The current parameter values (flattened initial fiber orientation).
    - grad (np.ndarray): Gradient array (used for L-BFGS, updated here).

    Returns:
    - float: The computed loss value.
    """
    # Zero out any previous gradients for the model
    model.zero_grad()

    global call_count
    call_count += 1

    # Reshape the input to the desired shape (12 parameters being optimized)
    reshaped_fiber_orientations = x.reshape(num_of_rows, num_of_cols, 1)

    # Duplicate data for prediction
    duplicate_fiber_orientation = duplicate_pixel_data(reshaped_fiber_orientations).to(device)

    # Ensure the tensor has requires_grad=True and use clone().detach() to avoid warnings
    fiber_orientation = torch.tensor(duplicate_fiber_orientation, dtype=torch.float32).to(device)

    # Forward pass
    predicted = model(fiber_orientation)

    predicted.retain_grad()

    # Keep only channels to optimize on
    predicted = cull_channels(predicted, channels_to_keep)

    # Compute the loss
    loss = loss_fn(predicted, input_tensor)

    # Perform backpropagation to calculate the gradients
    loss.backward()

    if grad.size > 0:
        print(f"gradients shape: {fiber_orientation.grad.shape}")


        grad_flat = average_patches_for_gradients(fiber_orientation.grad.cpu().numpy().squeeze(),
                                                  patch_size=(5, 5)).flatten()
        # Get only the gradients of the 12 parameters being optimized (not the full 300)
        # print(f"gradient after averaging: {grad_flat}, shape: {grad_flat.shape}")
        grad[:] = grad_flat

    print(f"Iteration: {call_count} Current Loss: {loss.item()}")

    return loss.item()


def new_objective_function(x, grad):
    global call_count
    call_count += 1

    print(f"Iteration: {call_count}")
    print(f"Input x shape: {x.shape}, size: {x.size}")
    print(f"Expected reshape dimensions: ({num_of_rows}, {num_of_cols}, 1)")

    # Reshape input
    if x.size != (num_of_rows * num_of_cols * 1):
        raise ValueError(f"Cannot reshape array of size {x.size} into shape ({num_of_rows}, {num_of_cols}, 1)")
    reshaped_fiber_orientations = x.reshape(num_of_rows, num_of_cols, 1)
    print(f"Reshaped fiber orientations shape: {reshaped_fiber_orientations.shape}")

    # Duplicate data for prediction
    duplicate_fiber_orientation = duplicate_pixel_data(reshaped_fiber_orientations).to(device)
    print(f"Duplicated fiber orientations shape: {duplicate_fiber_orientation.shape}")
    fiber_orientation = torch.tensor(duplicate_fiber_orientation, dtype=torch.float32, requires_grad=True).to(device)

    # Forward pass
    predicted = model(fiber_orientation)
    print(f"Predicted tensor shape: {predicted.shape}")
    print(f"Predicted grad: {predicted.grad}")

    # Retain gradients for the output tensor
    predicted.retain_grad()

    # Keep only channels to optimize on
    predicted = cull_channels(predicted, channels_to_keep)
    print(f"Filtered predicted tensor shape after culling channels: {predicted.shape}")

    # Retain gradients for the filtered tensor
    predicted.retain_grad()


    # Compute confidence-based weights
    if call_count == 1:  # Handle the first iteration
        print("Skipping confidence calculation on first iteration to compute initial loss.")
        confidence_scores = torch.ones_like(predicted)  # Dummy confidence scores for the first iteration
    else:
        try:
            confidence_scores = calculate_confidence(predicted)  # Ensure this doesn't require `.grad` of `predicted`
        except RuntimeError as e:
            print(f"Error calculating confidence: {e}")
            raise

    print(f"Confidence scores shape: {confidence_scores.shape}")

    # Compute loss with combined weights

    patch_distance_weights = calculate_patch_weights(predicted.shape[2:], patch_size=(5, 5)).to(device)
    print(f"Patch distance weights shape: {patch_distance_weights.shape}")
    print(f"Patch Distance Ww")
    combined_weights = confidence_scores * patch_distance_weights
    print(f"Combined weights shape: {combined_weights.shape}")
    loss = torch.mean(combined_weights * (predicted - input_tensor) ** 2)
    print(f"Loss value: {loss.item()}")

    # Backward pass
    model.zero_grad()
    loss.backward()

    print(f"Fiber orientation Grad shape after backward pass: {fiber_orientation.grad.shape} and grad: {fiber_orientation.grad}")

    # Debugging: Gradients after backward pass
    if fiber_orientation.grad is not None:
        print(f"Fiber orientation Grad shape: {fiber_orientation.grad.shape}")
    else:
        print("Fiber orientation Grad is None.")

    if grad.size > 0:
        grad_flat = average_patches_for_gradients(
            fiber_orientation.grad.cpu().numpy().squeeze(), patch_size=(5, 5)
        ).flatten()
        grad[:] = grad_flat
        print(f"Flattened gradient shape: {grad_flat.shape}")

    print(f"Iteration: {call_count} | Current Weighted Loss: {loss.item()}")
    return loss.item()

# def calculate_patch_weights(image_size, patch_size, grid_shape=(4, 3)):
#     height, width = image_size
#     patch_h, patch_w = patch_size
#     rows, cols = grid_shape
#
#     # Calculate patch centers based on grid positions
#     patch_centers_y = torch.linspace(patch_h / 2, height - patch_h / 2, rows)
#     patch_centers_x = torch.linspace(patch_w / 2, width - patch_w / 2, cols)
#
#     # Create a grid of distances to the center of each patch
#     y, x = torch.meshgrid(
#         torch.arange(0, height, dtype=torch.float32),
#         torch.arange(0, width, dtype=torch.float32),
#         indexing='ij'
#     )
#
#     # Calculate distance to the closest patch center
#     distances = torch.full((height, width), float('inf'))
#
#     for center_y in patch_centers_y:
#         for center_x in patch_centers_x:
#             distance_to_patch_center = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
#             distances = torch.minimum(distances, distance_to_patch_center)
#
#     # Normalize distances and convert to weights (closer = higher weight)
#     normalized_distance = 1 - (distances / distances.max())
#     print("normalized weight of distance:")
#     print(normalized_distance)
#
#     return normalized_distance

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

    print(f"Input tensor requires_grad: {tensor.requires_grad}")


    # Validate input dimensions
    if len(tensor.shape) != 4:
        raise ValueError("Input tensor must have shape (batch, channels, x, y).")

    # Validate channel indices
    num_channels = tensor.shape[1]
    if any(ch < 0 or ch >= num_channels for ch in channels_to_keep):
        raise ValueError("Channels to keep must be within the range of available channels.")

    # Select the specified channels
    culled_tensor = tensor.index_select(1, torch.tensor(channels_to_keep, device=tensor.device))


    print(f"Output tensor requires_grad: {culled_tensor.requires_grad}")

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

# Import the surface to optimize from Excel
input_tensor, vector_df = excel_to_np_array(file_path=excel_file_path, sheet_name='Sheet1',
                                 global_features_max=global_features_max, global_features_min=global_features_min)

input_tensor = input_tensor.to(device)
print_tensor_stats(input_tensor)
if visualize:
    visualize_curvature_tensor(input_tensor, labels_channels)

# Keep only certain channels to optimize to
input_tensor = cull_channels(input_tensor,channels_to_keep)

if visualize:
    visualize_curvature_tensor(input_tensor, len(channels_to_keep))

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



average_patches_np = average_patches(angles, (5,5),
                                     r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\Optimization debug_Original.xlsx",
                                     r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\Optimization debug_Average.xlsx")

print("averages array shape: ", average_patches_np.shape)
num_of_patches = average_patches_np.shape[0] * average_patches_np.shape[1]
print( f"number of patches: {num_of_patches}")


# Convert the NumPy array to a PyTorch tensor
average_patches_tensor = torch.tensor(average_patches_np, dtype=torch.float32)

# Normalize the tensor by dividing by 180
normalized_tensor = average_patches_tensor / 180.0
print(f"shape of average degrees array {normalized_tensor.shape}")

# Export to excel for debugging purposes
fiber_orientation_to_excel(normalized_tensor, global_labels_max, "initial_fiber_orientations.xlsx")

# Clone and detach the tensor, then set requires_grad to True
initial_fiber_orientation = normalized_tensor.clone().detach().requires_grad_(True)
print(f"Initial before duplication {initial_fiber_orientation}")

# Define the loss
loss_fn = nn.L1Loss()

# Define the optimizer - examples of different optimizers
optimizer = optim.Adam(params=[initial_fiber_orientation], lr=0.005)



if optimizer_type == 'basic':
    print(f"Using basic optimizer")
    for step in range(max_iterations):
        optimizer.zero_grad()

        # Duplicate data for prediction (from 4x3 to 20x15)
        duplicate_fiber_orientation = duplicate_pixel_data(initial_fiber_orientation).to(device)

        # Forward pass
        predicted = model(duplicate_fiber_orientation)

        # Keep only channels to optimize on
        predicted = cull_channels(predicted, channels_to_keep)

        # Print every x steps
        if step % print_steps == 0:
            print("Fiber Orientation Tensor:")
            print_tensor_stats(duplicate_fiber_orientation)

            print("Predicted Tensor:")
            print_tensor_stats(predicted)
            visualize_curvature_tensor(predicted, len(channels_to_keep))

            print(f"duplicate_fiber_orientation: {duplicate_fiber_orientation}")

        # Compute loss
        loss = loss_fn(predicted, input_tensor)

        # Compute Gradients
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
        # with torch.no_grad():
        #     initial_fiber_orientation.clamp_(0, 1)


        # Print the loss for the current step
        print(f'Step {step + 1}, Loss: {loss.item()}')

        if loss.item() < desired_threshold:
            print('Desired threshold reached. Stopping optimization.')
            break

    # Convert the optimized fiber orientation tensor to a 2D DataFrame and save to Excel
    final_fiber_orientation_final = initial_fiber_orientation.detach()


elif optimizer_type == 'nl-opt':
    print("Using nl-opt optimization")

    # Initialize NLopt optimizer
    # try to 0 with direct
    opt = nlopt.opt(nlopt.GN_DIRECT_L, initial_fiber_orientation.numel())

    # Initialize the counter
    call_count = 0

    # Set the lower and upper bounds (clamp between 0 and 1)
    opt.set_lower_bounds(0.0)
    opt.set_upper_bounds(1.0)

    # Set the objective function
    # opt.set_min_objective(objective_function)

    # Set the objective function
    opt.set_min_objective(new_objective_function)

    # Set stopping criteria
    opt.set_maxeval(max_iterations)  # Maximum number of iterations
    opt.set_ftol_rel(1e-8)  # Relative tolerance on the function value
    opt.set_xtol_rel(1e-8)  # Relative tolerance on the parameters

    # Flatten initial fiber orientation for NLopt
    x0 = initial_fiber_orientation.cpu().numpy().flatten()
    print(f"Initial x0 values: {x0}")

    # Run the optimizer
    optimized_x = opt.optimize(x0)


    print("Optimization completed.")
    # print(f"Final loss: {objective_function(optimized_x, optimized_x.grad)}")
    # print(f"Shape of optimized: {optimized_x.shape}")

    final_fiber_orientation_final = optimized_x.reshape(num_of_rows, num_of_cols)


fiber_orientation_to_excel(final_fiber_orientation_final, global_labels_max)

print("Optimization complete. Result saved to Excel.")
