import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from numpy.ma.extras import average
import matplotlib.pyplot as plt



# Input Files

model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\30-33_MaxCV_Forward_20241027.pkl"
excel_file_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\rhino_to_model_inverse.xlsx"

features_channels = 1
labels_channels = 4

# Normalization Aspect
global_labels_min = 0.0
global_labels_max = 180.0
global_features_min = -1.043418
global_features_max = 1.949431


# Model Architecture
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

#Functions
def export_each_channel_to_excel(prediction_np, base_save_path="predictions_channel"):

    df = pd.DataFrame(prediction_np)

    # Define a unique filename for each channel
    save_path = f"{base_save_path}.xlsx"
    df.to_excel(save_path, index=False, sheet_name=f"Channel_1", header=False)

    print(f"predictions exported to {save_path}")

def excel_to_np_array(file_path, sheet_name='Sheet1', global_features_max=10, global_features_min=-10):
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

    # Check if the data has the correct shape (300, 4)
    if data.shape != (300, 4):
        raise ValueError(f"Unexpected data shape {data.shape}, expected (300, 4)")

    # Reshape each channel (column) from 300 to (20, 15) using Fortran order (column-major)
    reshaped_data = [np.reshape(data[:, i], (20, 15), order='F') for i in range(4)]

    # Stack the reshaped arrays along the third axis (channels)
    final_array = np.stack(reshaped_data, axis=-1)


    print(f"from excel{final_array.shape}")


    for c in range(4):
        img = final_array[:,:,c]
        df_img = pd.DataFrame(img)
        df_img.to_excel(f'reshape_debug_channel_{c}.xlsx', index=False)
        # plt.imshow(img, cmap='gray')
        # plt.show()

    # Convert the NumPy array to a PyTorch tensor
    print(data)
    data = torch.from_numpy(final_array).unsqueeze(0).float()


    print(f"after converting to tensor {data.size()}")
    data = torch.permute(data, dims=(0,3,1,2))



    # Normalize the features using the global min and max
    normalized_data = (data - global_features_min) / (global_features_max - global_features_min)


    return normalized_data


# Main

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test From Excel
input_from_excel = excel_to_np_array(file_path=excel_file_path, sheet_name='Sheet1', global_features_max=global_features_max, global_features_min=global_features_min)


# Make prediction using model
model = OurModel()
model.load_state_dict(torch.load(model_path))
model.to(device)

# ==========================================================
# Optimization Process - Matching Predicted Output to Excel Data
# ==========================================================

input_tensor = input_from_excel
input_tensor = input_tensor.to(device)
print(f"np shape {input_tensor.size()}")

# Initialize the fiber orientation with all zeros and move to the device
initial_fiber_orientation = torch.zeros((1, 1, 20, 15), dtype=torch.float32, device=device, requires_grad=True)
print(initial_fiber_orientation)

# Set up the optimizer (SGD)
optimizer = optim.SGD([initial_fiber_orientation], lr=0.1)

# Define the loss function
loss_fn = nn.L1Loss()

# Optimization loop
max_iterations = 10000
desired_threshold = 0.01

for step in range(max_iterations):
    optimizer.zero_grad()

    # Forward pass
    predicted_images = model(initial_fiber_orientation)

    # Compute loss
    loss = loss_fn(predicted_images, input_tensor)
    print((predicted_images.size()))
    print(input_tensor.size())
    loss.backward()

    # Calculate and print gradient statistics
    grad_size = initial_fiber_orientation.grad
    if grad_size is not None:
        max_grad = round(grad_size.max().item(), 4)  # Maximum gradient rounded to 4 decimal places
        min_grad = round(grad_size.min().item(), 4)  # Minimum gradient rounded to 4 decimal places
        mean_grad = round(grad_size.mean().item(), 4)  # Mean gradient rounded to 4 decimal places

        print(f'Gradient stats - Max: {max_grad}, Min: {min_grad}, Mean: {mean_grad}')

    # Scale gradients for faster convergence
    initial_fiber_orientation.grad *= 20  # Scale gradients by 10

    optimizer.step()

    # Clamp the values to ensure they stay within [0, 1]
    with torch.no_grad():
        initial_fiber_orientation.clamp_(0, 1)

    # Print the loss for the current step
    print(f'Step {step + 1}, Loss: {loss.item()}')


    if loss.item() < desired_threshold:
        print('Desired threshold reached. Stopping optimization.')
        break

# Save the optimized fiber orientation to Excel
# Convert the optimized fiber orientation tensor to a 2D DataFrame and save to Excel
optimized_fiber_orientation_df = pd.DataFrame(initial_fiber_orientation.detach().to("cpu").numpy().squeeze(0).squeeze(0))
optimized_fiber_orientation_df.to_excel('optimized_fiber_orientation.xlsx', index=False)

print("Optimization complete. Result saved to Excel.")


# # Output the prediction to excel file
# predicted_fiber_orientations_denorm = predicted_fiber_orientations.clone()  # Clone to avoid modifying the original tensor
# for c in range(labels_channels):
#     predicted_fiber_orientations_denorm[:, c, :, :] = predicted_fiber_orientations_denorm[:, c, :, :] * (global_labels_max - global_labels_min) + global_labels_min
#
#
