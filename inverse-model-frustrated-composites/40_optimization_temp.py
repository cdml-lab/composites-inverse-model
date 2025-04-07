# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Imports                                         │
# └───────────────────────────────────────────────────────────────────────────┘


import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Definitions                                     │
# └───────────────────────────────────────────────────────────────────────────┘


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

            # Forward pass
            predicted = model(duplicate_fiber_orientation)
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




fiber_orientation_to_excel(final_fiber_orientation, global_labels_max)

print("Optimization complete. Result saved to Excel.")
