# Imports
import torch
import numpy as np
import h5py
import pandas as pd
from torch import nn
# Input Files

model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\inverse_autumn_smoke_114.pth"

# new_samples_file_path_features = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\100\100_MaxCV_Features_Reshaped.h5"
# new_samples_file_path_labels = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\100\100_MaxCV_Labels_Reshaped.h5"
excel_file_path = r"/rhino_to_model_inverse.xlsx"


features_channels = 8
labels_channels = 1
height = 20
width = 15

channels_to_use = [0,1,2,3,4,5,6,7]

features_main_group = 'Features'
labels_main_group = 'Labels'
category = 'Train'
compute_certainty = True

x=1 # Random sample selection

# Normalization Aspect
global_labels_max = 180.0
global_labels_min = 0.0


#
global_features_max = [10.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
global_features_min = [-10.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5]
# global_features_max = 10.0
# global_features_min = -10.0



# Model Architecture

class OurVgg16t(torch.nn.Module):
    def __init__(self, dropout=0.3, height = height, width = width):
        super(OurVgg16t, self).__init__()

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
        self.conv_14 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)



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
        self.batch_norm_14 = torch.nn.BatchNorm2d(num_features=64)
        self.batch_norm_15 = torch.nn.BatchNorm2d(num_features=64)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.fc1 = nn.Linear(512 * height * width, 512)  # Output size adjusted for 1 channel with resolution 20x15
        # self.fc2 = nn.Linear(512, labels_channels * height * width)  # Output size adjusted for 1 channel with resolution 20x15
        self.fc3 = nn.Linear(64 * height * width, labels_channels * height * width)
        self.upsample = torch.nn.Upsample(size=(height, width), mode='nearest')
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

        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.relu(x)
        # x = self.dropout(x)  # Dropout after every 3 layers

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

        x = self.conv_14(x)
        x = self.batch_norm_14(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # print(f"after conv14 {x.shape}")


        # Flatten and pass through the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"after flatten {x.shape}")

        # x = self.fc1(x)
        # print(f"after fc1 {x.shape}")
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        # Reshape and upsample
        x = x.view(x.size(0), labels_channels, height, width)
        x = self.upsample(x)
        return x


#Functions
def export_each_channel_to_excel(prediction_np, base_save_path="predictions_channel"):

    df = pd.DataFrame(prediction_np)

    # Define a unique filename for each channel
    save_path = f"{base_save_path}.xlsx"
    df.to_excel(save_path, index=False, sheet_name=f"Channel_1", header=False)

    print(f"predictions exported to {save_path}")

def excel_to_np_array(file_path, sheet_name='Sheet1', global_features_max=10, global_features_min=-10):
    f"""
    Reads an Excel file with X amount of columns and 300 rows and converts it into a NumPy array
    of shape (20, 15, features channels), with each column representing a channel and reorganizing the
    rows using Fortran order.

    Parameters:
    - file_path (str): Path to the Excel file.
    - sheet_name (str): Name of the sheet in the Excel file. Default is 'Sheet1'.

    Returns:
    - np.ndarray: A NumPy array of shape (20, 15, features_channels) with the data organized by (height, width, channels).
    """
    # Read the xlsx file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Drop the header and convert to NumPy array
    data = df.to_numpy()

    # Check if the data has the correct shape (300, 4)
    if data.shape != (300, features_channels):
        raise ValueError(f"Unexpected data shape {data.shape}, expected (300, {features_channels})")

    # Reshape each channel (column) from 300 to (20, 15) using Fortran order (column-major)
    reshaped_data = [np.reshape(data[:, i], (20, 15), order='F') for i in range(features_channels)]

    # Stack the reshaped arrays along the third axis (channels)
    final_array = np.stack(reshaped_data, axis=-1)


    print(f"from excel{final_array.shape}")


    for c in range(features_channels):
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

    # Convert global_features_min and global_features_max to PyTorch tensors
    global_features_min = torch.tensor(global_features_min, dtype=torch.float32).view(1, features_channels, 1, 1)
    global_features_max = torch.tensor(global_features_max, dtype=torch.float32).view(1, features_channels, 1, 1)

    # Normalize the features
    normalized_data = (data - global_features_min) / (global_features_max - global_features_min)

    return normalized_data




# Main

# Test From Excel
input_from_excel = excel_to_np_array(file_path=excel_file_path, sheet_name='Sheet1', global_features_max=global_features_max, global_features_min=global_features_min)


input_curvature = input_from_excel


# Make prediction using model
# model = OurModel()
model = OurVgg16t(height = 20, width=15)
model.load_state_dict(torch.load(model_path))


# Make prediction
model.eval()
with torch.no_grad():
    predicted_fiber_orientations = model(input_curvature)
    print(f"Predicted Fiber Orientations datatype: {predicted_fiber_orientations.dtype} Size: {predicted_fiber_orientations.size()}")




# Output the prediction to excel file
predicted_fiber_orientations_denorm = predicted_fiber_orientations.clone()  # Clone to avoid modifying the original tensor
for c in range(labels_channels):
    predicted_fiber_orientations_denorm[:, c, :, :] = predicted_fiber_orientations_denorm[:, c, :, :] * (global_labels_max - global_labels_min) + global_labels_min

predicted_fiber_orientations_denorm_np = predicted_fiber_orientations_denorm.squeeze().numpy()  # Convert to NumPy for plotting

print(f"after numpy{np.shape(predicted_fiber_orientations_denorm_np)}")

export_each_channel_to_excel(prediction_np=predicted_fiber_orientations_denorm_np, base_save_path="predicted_fiber_orientation_inverse")

# Load corresponding labels

gt_fiber_orientation = load_labels_h5_data(new_samples_file_path_labels, labels_main_group, category)
print(f"gt fiber orientation: {gt_fiber_orientation.shape}")
gt_fiber_orientation = gt_fiber_orientation[x:x+1,:, :, :].squeeze()
print(f"After selecting 1 sample {gt_fiber_orientation.shape}")

export_each_channel_to_excel(prediction_np=gt_fiber_orientation, base_save_path="gt_fiber_orientation_inverse")







