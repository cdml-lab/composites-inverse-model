
### This Code is meant for 3 dimensional data and will not work for more


# Import Libraries
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


og_dataset_name = '14_16small'
dataset_name = '14_16small_MaxCV_overlap0'
patches = '_Patches'


# Set dataset files
features_file_path = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + og_dataset_name + '/' + dataset_name + '_Features' + patches + '.h5'
labels_file_path = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + og_dataset_name + '/' + dataset_name + '_Labels' + patches + '.h5'

import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import os


def normalize_to_hsl(data):
    """
    Normalize data to fit HSL format.

    Args:
    data (numpy array): The data to normalize.

    Returns:
    numpy array: Normalized data.
    """
    data_min = -2
    data_max = 2
    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data


def display_patch(patch, patch_id):
    """
    Display the patch using HSL.

    Args:
    patch (numpy array): The patch to display.
    patch_id (int): The ID of the patch for labeling.
    """
    normalized_patch = normalize_to_hsl(patch)

    plt.figure(figsize=(6, 6))
    plt.imshow(normalized_patch, cmap='viridis', vmin=0, vmax=1)
    plt.title(f'Patch {patch_id}')
    plt.colorbar(label='Normalized Value')
    plt.show()


def check_hdf5_data_with_patches(hdf5_file_path, num_patches):
    """
    Check the HDF5 file and display the shape of arrays with examples of random patches.

    Args:
    hdf5_file_path (str): Path to the HDF5 file to check.
    num_patches (int): Number of example patches to display.
    """
    if not os.path.exists(hdf5_file_path):
        print(f"Error: File '{hdf5_file_path}' does not exist.")
        return

    with h5py.File(hdf5_file_path, 'r') as h5file:
        all_patches = []
        for category in h5file:
            main_group = h5file[category]
            for folder_suffix in main_group:
                sub_group = main_group[folder_suffix]
                for dataset_name in sub_group:
                    dataset = sub_group[dataset_name]
                    data = dataset[:]
                    all_patches.append((category, folder_suffix, dataset_name, data))

        # Randomly select the requested number of patches
        selected_patches = random.sample(all_patches, min(num_patches, len(all_patches)))

        for idx, selected_patch in enumerate(selected_patches):
            category, folder_suffix, dataset_name, data = selected_patch
            print(f'Category: {category}')
            print(f'  Folder: {folder_suffix}')
            print(f'    Dataset: {dataset_name}')
            print(f'      Shape: {data.shape}')


            display_patch(data,idx + 1)

# Prompt the user for the number of example patches
num_patches = int(input("How many example patches do you want to see? "))

# Main Run
check_hdf5_data_with_patches(features_file_path, num_patches)
