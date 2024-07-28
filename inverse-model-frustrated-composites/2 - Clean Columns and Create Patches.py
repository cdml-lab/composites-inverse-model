

# Imports

import torch
import torchvision
import pandas as pd
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random


print("GPU available: {}".format(torch.cuda.is_available()))

# Set starting information

##############

dataset_name="14"
new_dataset_name="14_All_overlap1"
model_type = 'overlap' # use CNN or RF. changes the data structure output.


###############

cols = 3
rows = 4

pixels_per_patch_row = 5
pixels_per_patch_col = 5

overlap = 1

###############

dataset_dir = 'C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/' + dataset_name + '/'
new_dataset_dir = 'C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset' + dataset_name + '/' + new_dataset_name

file_path = hdf5_file_path = dataset_dir + dataset_name + '.h5'


features_dir = dataset_dir + '/Features'
labels_dir = dataset_dir + '/Labels'

features_patches_dir = new_dataset_dir + '/Features'
labels_patches_dir = new_dataset_dir + '/Labels'

features_file_path = dataset_dir + new_dataset_name + '_Features.h5'
labels_file_path  = dataset_dir + new_dataset_name + '_Labels.h5'

output_features_file_path = dataset_dir + new_dataset_name + '_Features_Patches.h5'
output_labels_file_path = dataset_dir + new_dataset_name + '_Labels_Patches.h5'


H= cols * pixels_per_patch_col #Height of images
W= rows * pixels_per_patch_row #Width of images

P=H*W #Pixel size of images
C=3  #Number of channels

##############

# Columns to Use
preserve_columns_features = ['Movement Vector Direction', 'Max Curvature Direction', 'Min Curvature Direction', 'Movement Vector Length', 'Max Curvature Length', 'Min Curvature Length', 'Location X', 'Location Y', 'Location Z']  # columns to preserve for features
preserve_columns_labels = ['Top Angle']  # columns to preserve for labels

# Columns to split and their new column names
split_columns_features = {
    'Movement Vector Direction': ['MVD-X', 'MVD-Y', 'MVD-Z'],
    'Max Curvature Direction': ['MaCD-X', 'MaCD-Y', 'MaCD-Z'],
    'Min Curvature Direction': ['MiCD-X', 'MiCD-Y', 'MiCD-Z'],
}

split_columns_labels = {}  # Assuming no split columns for Labels

remove_split_columns = []

suffixes = ['Train', 'Test']

"""
This is all of the options to copy from:

preserve_columns_features = ['Movement Vector Direction', 'Max Curvature Direction', 'Min Curvature Direction', 'Movement Vector Length', 'Max Curvature Length', 'Min Curvature Length', 'Location X', 'Location Y', 'Location Z']  # columns to preserve for features
preserve_columns_labels = ['Top Angle']  # columns to preserve for labels
preserve_columns_features = ['Max Curvature Direction', 'Max Curvature Length']  # columns to preserve for features


 Columns to split and their new column names
split_columns_features = {
    'Movement Vector Direction': ['MVD-X', 'MVD-Y', 'MVD-Z'],
    'Max Curvature Direction': ['MaCD-X', 'MaCD-Y', 'MaCD-Z'],
    'Min Curvature Direction': ['MiCD-X', 'MiCD-Y', 'MiCD-Z'],
}

 remove_split_columns = ['MaCD-Z']
"""

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU")


print(torch.__version__)
print(torch.version.cuda)



# Clean unnecessary columns and save to new files

def clean_hdf5_data(category, preserve_columns, split_columns, suffixes, hdf5_file_path, new_hdf5_file_path):
    """
    Cleans unnecessary columns from HDF5 datasets and saves the cleaned datasets to a new HDF5 file.
    If a dataset is found to be problematic, it skips copying it to the new file.

    Parameters:
    - category (str): Main group name in the HDF5 file (e.g., 'Features' or 'Labels').
    - preserve_columns (list of str): List of columns to preserve in the cleaned datasets.
    - split_columns (dict): Dictionary mapping columns to be split into multiple new columns.
    - suffixes (list of str): List of suffixes for subgroups (e.g., ['train', 'test']).
    - hdf5_file_path (str): Path to the input HDF5 file.
    - new_hdf5_file_path (str): Path to the output cleaned HDF5 file.

    Returns:
    - None
    """
    with h5py.File(hdf5_file_path, 'r') as h5file, h5py.File(new_hdf5_file_path, 'w') as new_h5file:
        main_group = h5file[category]  # Access the group (e.g., "Features" or "Labels")
        print(f"Main group '{category}' contains: {list(main_group.keys())}")

        # Recreate the main group
        if category in new_h5file:
            del new_h5file[category]
        new_main_group = new_h5file.create_group(category)

        for folder_suffix in suffixes:
            print(f"Trying to access subgroup '{folder_suffix}' in main group '{category}'")
            if folder_suffix not in main_group:
                print(f"Subgroup '{folder_suffix}' not found in main group '{category}'")
                continue

            sub_group = main_group[folder_suffix]  # Access the "train" or "test" subgroup within the category
            print(f"Subgroup '{folder_suffix}' contains: {list(sub_group.keys())}")

            # Recreate the subgroup only if it doesn't already exist
            if folder_suffix in new_main_group:
                new_sub_group = new_main_group[folder_suffix]
            else:
                new_sub_group = new_main_group.create_group(folder_suffix)

            for sheet in list(sub_group):  # Iterate through datasets within the subgroup
                if isinstance(sub_group[sheet], h5py.Group):
                    continue  # Skip if it's a group, not a dataset

                dataset = sub_group[sheet]
                print(f"Attributes for dataset '{sheet}': {list(dataset.attrs.keys())}")
                if 'columns' not in dataset.attrs:
                    print(f"Warning: Dataset '{sheet}' does not have 'columns' attribute. Skipping.")
                    continue

                columns_metadata = dataset.attrs['columns']
                df = pd.DataFrame(dataset[:], columns=columns_metadata)

                # Print all column names before processing
                print(f'Columns in {sheet} before processing: {list(df.columns)}')

                # Process specified columns to remove curly brackets and split into multiple columns
                for col, new_cols in split_columns.items():
                    if col in df.columns:
                        print(f"Processing column '{col}' in {sheet}")
                        # Decode bytes to strings
                        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                        df[col] = df[col].str.replace('{', '', regex=True).str.replace('}', '', regex=True)
                        df[new_cols] = df[col].str.split(',', expand=True)
                        df = df.drop(col, axis=1)

                # Determine which columns to keep (those that exist in the DataFrame)
                keep_columns = [col for col in preserve_columns if col in df.columns] + \
                               [new_col for cols in split_columns.values() for new_col in cols]

                # Remove Split Columns if needed
                keep_columns = [col for col in keep_columns if col not in remove_split_columns]

                print(f'Keeping columns in {sheet}: {keep_columns}')

                # Ensure the new columns exist in the DataFrame
                missing_columns = [col for col in keep_columns if col not in df.columns]
                if missing_columns:
                    print(f"Warning: The following columns are missing in '{sheet}' and will be skipped: {missing_columns}")
                    continue

                # Keep only the specified columns that exist
                df = df[keep_columns]

                # Convert all values to float
                df = df.astype(float)

                # Save the cleaned dataset to the new HDF5 file
                new_dataset = new_sub_group.create_dataset(sheet, data=df.to_numpy())
                new_dataset.attrs['columns'] = keep_columns
                print(f'Cleaned and saved {sheet} to new HDF5 file')

# prompt: check if file in file_path exits
if os.path.exists(file_path):
    print(file_path)
    print("File exists")
else:
    print(file_path)
    print("File does not exist")

# Run the Clean

## Features
try: clean_hdf5_data('Features', preserve_columns_features, split_columns_features, suffixes, file_path, features_file_path)
except: print( "Couldn't clean Feature file")

## Labels
try: clean_hdf5_data('Labels', preserve_columns_labels, split_columns_labels, suffixes, file_path, labels_file_path)
except: print( "Couldn't clean Labels file")


# Reconstruct Patches Functions

## New Functions for a 5x5x Z layout

def plot_image(image, label=None, title="Image"):
    """
    Normalize and plot an RGB image with text annotations.

    Args:
    image (numpy array): The image to plot.
    label (int): The corresponding label.
    title (str): The title of the plot.
    """
    # Normalize the image to [0, 1] range
    normalized_image = (image - (-10)) / (10 - (-10))

    fig, ax = plt.subplots()
    ax.imshow(normalized_image)
    ax.set_title(f"{title} - Label: {label}" if label is not None else title)
    ax.axis('off')

    # Add text annotations to about half the pixels
    rows, cols, _ = image.shape
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            text = f"{image[i, j, 0]:.1f}, {image[i, j, 1]:.1f}, {image[i, j, 2]:.1f}"
            ax.text(j, i, text, ha='center', va='center', color='white', fontsize=6,
                    bbox=dict(facecolor='black', alpha=0.5))

    plt.show()

def extract_patches(arr, num_row_patches, num_col_patches):
    """
    Extract patches from the reshaped array.

    Args:
    arr (numpy array): Reshaped array.
    num_row_patches (int): Number of patches along rows.
    num_col_patches (int): Number of patches along columns.

    Returns:
    list: List of patches.
    """
    patches = []
    rows, cols, _ = arr.shape

    row_step = rows // num_row_patches
    col_step = cols // num_col_patches

    for j in range(num_col_patches):
        for i in range(num_row_patches):
            start_row = i * row_step
            end_row = (i + 1) * row_step

            start_col = j * col_step
            end_col = (j + 1) * col_step

            patch = arr[start_row:end_row, start_col:end_col, :]
            patches.append(patch)
 #           plot_image(patch, title=f"Patch ({j}, {i})") #For Debugging

    return patches

def reconstruct_patches_features_new(hdf5_file_path, output_hdf5_file_path, category, patches_per_row, patches_per_col,
                                     pixels_per_patch_row, pixels_per_patch_col):
    """
    Process the HDF5 file, reshape datasets, extract patches, and save to a new HDF5 file.

    Args:
    hdf5_file_path (str): Path to the input HDF5 file.
    output_hdf5_file_path (str): Path to the output HDF5 file.
    category (str): Category of datasets to process.
    patches_per_row (int): Number of patches per row.
    patches_per_col (int): Number of patches per column.
    pixels_per_patch_row (int): Number of pixels per patch row.
    pixels_per_patch_col (int): Number of pixels per patch column.
    """
    with h5py.File(hdf5_file_path, 'r') as h5file, h5py.File(output_hdf5_file_path, 'w') as new_h5file:
        main_group = h5file[category]

        if category not in new_h5file:
            new_main_group = new_h5file.create_group(category)
        else:
            new_main_group = new_h5file[category]

        for folder_suffix in main_group:
            sub_group = main_group[folder_suffix]

            if folder_suffix not in new_main_group:
                new_sub_group = new_main_group.create_group(folder_suffix)
            else:
                new_sub_group = new_main_group[folder_suffix]

            for dataset_name in sub_group:
                dataset = sub_group[dataset_name]
                if dataset.size == 0:
                    print(f"Skipping empty dataset {dataset_name} in {folder_suffix}")
                    continue
                dataset = sub_group[dataset_name]
                original_data = dataset[:]
                num_columns = original_data.shape[1]

                # Reshape the array
                reshaped_array = original_data.reshape(
                    ( patches_per_col * pixels_per_patch_col, patches_per_row * pixels_per_patch_row, num_columns),
                    order='F')

                #plot_image(reshaped_array, title="Reshaped Array") # For Debugging

                # Extract patches
                patches = extract_patches(reshaped_array, patches_per_col, patches_per_row )



                # Save patches to the new HDF5 file
                for patch_row in range(patches_per_row):
                    for patch_col in range(patches_per_col):
                        patch_dataset_name = f'{dataset_name}_patch_{patch_col}_{patch_row}'
                        patch = patches[patch_row * patches_per_col + patch_col]
                        new_sub_group.create_dataset(patch_dataset_name, data=patch)

                        #plot_image(patch, title=patch_dataset_name)  # For Debugging
                        print(f'Patch {patch_dataset_name} saved in {folder_suffix}')

def reconstruct_patches_labels_new(hdf5_file_path, category, patches_per_row, patches_per_col,
                                                pixels_per_patch_row, pixels_per_patch_col, output_hdf5_file_path):
    """
    Process the HDF5 file, reshape datasets, extract label patches, and save to a new HDF5 file.

    Args:
    hdf5_file_path (str): Path to the input HDF5 file.
    output_hdf5_file_path (str): Path to the output HDF5 file.
    category (str): Category of datasets to process.
    patches_per_row (int): Number of patches per row.
    patches_per_col (int): Number of patches per column.
    pixels_per_patch_row (int): Number of pixels per patch row.
    pixels_per_patch_col (int): Number of pixels per patch column.
    """
    with h5py.File(hdf5_file_path, 'r') as h5file, h5py.File(output_hdf5_file_path, 'w') as new_h5file:
        main_group = h5file[category]

        if category not in new_h5file:
            new_main_group = new_h5file.create_group(category)
        else:
            new_main_group = new_h5file[category]

        for folder_suffix in main_group:
            sub_group = main_group[folder_suffix]

            if folder_suffix not in new_main_group:
                new_sub_group = new_main_group.create_group(folder_suffix)
            else:
                new_sub_group = new_main_group[folder_suffix]

            for dataset_name in sub_group:
                dataset = sub_group[dataset_name]
                if dataset.size == 0:
                    print(f"Skipping empty dataset {dataset_name} in {folder_suffix}")
                    continue

                original_data = dataset[:]

                # Reshape the array
                reshaped_array = original_data.reshape(
                    (patches_per_col * pixels_per_patch_col, patches_per_row * pixels_per_patch_row),
                    order='F')

                reshaped_array = reshaped_array[:, :, np.newaxis]  # Add a new axis to make it compatible

                # Extract patches
                patches = extract_patches(reshaped_array,patches_per_col, patches_per_row)

                # Save label patches to the new HDF5 file
                for patch_row in range(patches_per_row):
                    for patch_col in range(patches_per_col):
                        patch = patches[patch_row * patches_per_col + patch_col]
                        label = int(patch[0, 0, 0])  # Take the first value as the label
                        patch_dataset_name = f'{dataset_name}_patch_{patch_col}_{patch_row}'
                        new_sub_group.create_dataset(patch_dataset_name, data=label)
                        print(f'Label {label} saved to {patch_dataset_name} in {folder_suffix}')


import h5py
import numpy as np

import h5py
import numpy as np

import h5py
import numpy as np


def reconstruct_patches_features_with_overlap(hdf5_file_path, output_hdf5_file_path, category, patches_per_row,
                                              patches_per_col,
                                              pixels_per_patch_row, pixels_per_patch_col, overlap_size):
    """
    Process the HDF5 file, reshape datasets, extract patches with surrounding pixels, and save to a new HDF5 file.

    Args:
    hdf5_file_path (str): Path to the input HDF5 file.
    output_hdf5_file_path (str): Path to the output HDF5 file.
    category (str): Category of datasets to process.
    patches_per_row (int): Number of patches per row.
    patches_per_col (int): Number of patches per column.
    pixels_per_patch_row (int): Number of pixels per patch row.
    pixels_per_patch_col (int): Number of pixels per patch column.
    overlap_size (int): Number of additional pixels around each patch.
    """
    with h5py.File(hdf5_file_path, 'r') as h5file, h5py.File(output_hdf5_file_path, 'w') as new_h5file:
        main_group = h5file[category]

        if category not in new_h5file:
            new_main_group = new_h5file.create_group(category)
        else:
            new_main_group = new_h5file[category]

        for folder_suffix in main_group:
            sub_group = main_group[folder_suffix]

            if folder_suffix not in new_main_group:
                new_sub_group = new_main_group.create_group(folder_suffix)
            else:
                new_sub_group = new_main_group[folder_suffix]

            for dataset_name in sub_group:
                dataset = sub_group[dataset_name]
                if dataset.size == 0:
                    print(f"Skipping empty dataset {dataset_name} in {folder_suffix}")
                    continue
                original_data = dataset[:]

                # Ensure the data is 2D before reshaping
                if original_data.ndim != 2:
                    print(
                        f"Unexpected data dimensions for {dataset_name} in {folder_suffix}. Expected 2D, got {original_data.ndim}D.")
                    continue

                num_rows, num_columns = original_data.shape

                # Reshape the array to 3D using num_columns
                reshaped_array = original_data.reshape(
                    (patches_per_col * pixels_per_patch_col, patches_per_row * pixels_per_patch_row, num_columns),
                    order='F')

                # Debugging plot for reshaped array
                #plot_image(reshaped_array, title="Reshaped Array")  # For Debugging

                num_rows, num_cols, num_channels = reshaped_array.shape

                # Extract patches with surrounding pixels
                for patch_row in range(patches_per_col):
                    for patch_col in range(patches_per_row):
                        start_row = patch_row * pixels_per_patch_row
                        start_col = patch_col * pixels_per_patch_col
                        end_row = start_row + pixels_per_patch_row
                        end_col = start_col + pixels_per_patch_col

                        # Add overlap
                        extended_start_row = max(0, start_row - overlap_size)
                        extended_start_col = max(0, start_col - overlap_size)
                        extended_end_row = min(num_rows, end_row + overlap_size)
                        extended_end_col = min(num_cols, end_col + overlap_size)

                        # Extract the patch with its surroundings
                        patch_with_surrounding = reshaped_array[extended_start_row:extended_end_row,
                                                 extended_start_col:extended_end_col, :]

                        # Handle cases where the patch extends beyond the original data boundaries
                        extended_patch = np.full((pixels_per_patch_row + 2 * overlap_size,
                                                  pixels_per_patch_col + 2 * overlap_size, num_channels),
                                                 999.9, dtype=np.float32)
                        insert_row_start = overlap_size - (
                                    start_row - extended_start_row) if start_row - overlap_size < 0 else 0
                        insert_col_start = overlap_size - (
                                    start_col - extended_start_col) if start_col - overlap_size < 0 else 0
                        insert_row_end = insert_row_start + patch_with_surrounding.shape[0]
                        insert_col_end = insert_col_start + patch_with_surrounding.shape[1]

                        extended_patch[insert_row_start:insert_row_end, insert_col_start:insert_col_end,
                        :] = patch_with_surrounding

                        patch_dataset_name = f'{dataset_name}_patch_{patch_row}_{patch_col}' #This seems flipped but it's right
                        new_sub_group.create_dataset(patch_dataset_name, data=extended_patch)

                        # Debugging plot for each patch
                        #plot_image(extended_patch, title=patch_dataset_name)  # For Debugging

                        print(f'Patch {patch_dataset_name} saved in {folder_suffix}')


## Run the Patches

patches_per_row = cols
patches_per_col = rows

if model_type == 'CNN':
    print("CNN Style data-structure")
    reconstruct_patches_features_new(features_file_path, output_features_file_path, 'Features', patches_per_row, patches_per_col, pixels_per_patch_row, pixels_per_patch_col)
    try: reconstruct_patches_labels_new(labels_file_path, 'Labels', patches_per_row, patches_per_col,
                                                pixels_per_patch_row, pixels_per_patch_col, output_labels_file_path)
    except: print("Couldn't patchify labels file")

elif model_type == 'RF':
    print("RF Style data-structure isn't operative")
    # Labels
    #reconstruct_patches_labels_and_save_to_hdf5(labels_file_path, 'Labels', patches_per_row, patches_per_col, pixels_per_patch_row, pixels_per_patch_col, output_labels_file_path)

    #reconstruct_patches_features_and_save_to_hdf5(features_file_path, 'Features', patches_per_row, patches_per_col, pixels_per_patch_col, output_features_file_path)
elif model_type=='overlap':
    print(f"CNN Style data-structure with {overlap} pixel overlap")
    reconstruct_patches_features_with_overlap(features_file_path, output_features_file_path, 'Features', patches_per_row, patches_per_col, pixels_per_patch_row, pixels_per_patch_col, overlap)

    try: reconstruct_patches_labels_new(labels_file_path, 'Labels', patches_per_row, patches_per_col,
                                                pixels_per_patch_row, pixels_per_patch_col, output_labels_file_path)
    except: print("Couldn't patchify labels file")

# Display Functions

def display_label_image(data):
    # Ensure the data has 25 elements
    if len(data) != 25:
        raise ValueError("Input data must have 25 elements.")
    #convert to numpy
    data = np.array(data)

    # Reshape the data to a 5x5 grid
    reshaped_data = data.reshape((5, 5))

    # Display the image
    plt.imshow(reshaped_data, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.show()
def display_feature_image(data, x, y):
    """
    Display a single feature image.
    """
    data = np.array(data)
    if len(data) != x * y * 3:
        print(f"Input data is {len(data)} elements")
        return

    data = (data + 1) / 2  # Normalize the values
    reshaped_data = data.reshape((x, y, 3))

    plt.imshow(reshaped_data)
    plt.axis('off')  # Turn off axis labels

def display_feature_image_full(data,x,y):

    #convert to numpy
    data = np.array(data)
        # Ensure the data has 900 elements
    if len(data) != 900:
        print("Input data is ", len(data), " elements")

    # Normalize the values
    data = (data + 1)/2

    # Reshape the data to a  grid
    reshaped_data = data.reshape((x, y, 3))
    #print(reshaped_data)

    # Display the image
    plt.imshow(reshaped_data)
    plt.axis('off')  # Turn off axis labels
    plt.show()
def display_feature_images_grid(data_list, x, y, cols=5):
    """
    Displays a grid of feature images.

    Parameters:
    - data_list (list): List of feature data arrays.
    - x (int): Number of pixel rows in each patch.
    - y (int): Number of pixel columns in each patch.
    - cols (int): Number of columns in the grid.
    """
    # Number of images
    num_images = len(data_list)
    rows = (num_images + cols - 1) // cols  # Calculate the number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for i, data in enumerate(data_list):
        data = np.array(data)
        if len(data) != x * y * 3:
            print(f"Input data {i} is {len(data)} elements")
            continue

        data = (data + 1) / 2  # Normalize the values
        reshaped_data = data.reshape((x, y, 3))

        axes[i].imshow(reshaped_data)
        axes[i].axis('off')

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# Inspect Results (maybe move this to a different code)

# Open the HDF5 files
with h5py.File(output_features_file_path, 'r') as f_features, h5py.File(output_labels_file_path, 'r') as f_labels:
    # Initialize counters for displayed features and labels
    displayed_count = 0
    max_display = 10

    # Iterate through each subgroup (train, test) in the 'Features' group
    for group_name in f_features['Features'].keys():
        if displayed_count >= max_display:
            break
        features_group = f_features['Features'][group_name]
        try: labels_group = f_labels['Labels'][group_name]
        except: print("Could not open Labels file")

        # Iterate through each dataset in the subgroup
        for dataset_name in features_group.keys():
            if displayed_count >= max_display:
                break

            try:
                # Get the actual data from the dataset
                feature_data = features_group[dataset_name][()]
                label_data = labels_group[dataset_name][()]

                # Print the features and labels using the display functions
                print(f"Features for {group_name}/{dataset_name}:")
                display_feature_image(feature_data, 5, 5)  # Use feature_data
                print(f"Labels for {group_name}/{dataset_name}:{label_data}")

                displayed_count += 1
            except KeyError as e:
                print(f"Error: Dataset '{group_name}/{dataset_name}' not found in one of the HDF5 files. {e}")
            except Exception as e:
                print(f"An error occurred while processing dataset '{group_name}/{dataset_name}': {e}")

"""## Search for specific labels"""

def search_hdf5_for_number(hdf5_file_path, category, target_number):
    """
    Searches for a target number in the first cell of each dataset within an HDF5 file.

    Parameters:
    - hdf5_file_path (str): Path to the HDF5 file.
    - category (str): Main group name in the HDF5 file (e.g., 'Features' or 'Labels').
    - target_number (int or float): The target number to search for.

    Returns:
    - matching_datasets (list): List of dataset names that contain the target number in the first cell.
    """
    matching_datasets = []

    with h5py.File(hdf5_file_path, 'r') as h5file:
        main_group = h5file[category]

        for folder_suffix in main_group:
            sub_group = main_group[folder_suffix]

            for dataset_name in sub_group:
                dataset = sub_group[dataset_name]
                if dataset.size == 0:
                    print(f"Skipping empty dataset {dataset_name} in {folder_suffix}")
                    continue
                data = dataset[()]

                if data.size > 0 and data.flat[0] == target_number:  # Check the first cell
                    matching_datasets.append(f"{folder_suffix}/{dataset_name}")

    return matching_datasets

# Example usage
target_number = 105
matching_datasets = search_hdf5_for_number(output_labels_file_path, 'Labels', target_number)

print("Datasets containing the target number:")
#for ds in matching_datasets:
    #print(ds)

"""##Search for corresponding features:"""

def get_corresponding_datasets(dataset_list, hdf5_file_path, category):
    """
    Retrieves the corresponding datasets from an HDF5 file based on a list of dataset names.

    Parameters:
    - dataset_list (list): List of dataset names to retrieve.
    - hdf5_file_path (str): Path to the HDF5 file.
    - category (str): Main group name in the HDF5 file (e.g., 'Features' or 'Labels').

    Returns:
    - corresponding_datasets (list): List of corresponding datasets.
    """
    corresponding_datasets = []

    with h5py.File(hdf5_file_path, 'r') as h5file:
        main_group = h5file[category]

        for dataset_name in dataset_list:
            try:
                folder_suffix, ds_name = dataset_name.split('/')
                sub_group = main_group[folder_suffix]
                dataset = sub_group[ds_name]
                corresponding_datasets.append(dataset)
            except KeyError:
                print(f"Dataset {dataset_name} not found in {category}")

    return corresponding_datasets

print(output_features_file_path)
print(output_labels_file_path)

def display_samples_from_matching_datasets(matching_datasets, features_file_path, labels_file_path, num_samples=10):
    """
    Displays samples from matching datasets by printing labels and displaying features.

    Parameters:
    - matching_datasets (list): List of matching dataset names.
    - features_file_path (str): Path to the HDF5 file containing features.
    - labels_file_path (str): Path to the HDF5 file containing labels.
    - num_samples (int): Number of samples to display.
    """
    if len(matching_datasets) < num_samples:
        print(f"Not enough matching datasets. Found {len(matching_datasets)} datasets.")
        return

    selected_datasets = random.sample(matching_datasets, num_samples)
    feature_data_list = []

    with h5py.File(features_file_path, 'r') as f_features, h5py.File(labels_file_path, 'r') as f_labels:
        for dataset_name in selected_datasets:
            try:
                print(f"Processing dataset: {dataset_name}")

                feature_dataset_path = f"Features/{dataset_name}"
                label_dataset_path = f"Labels/{dataset_name}"

                if feature_dataset_path not in f_features or label_dataset_path not in f_labels:
                    print(f"Dataset '{dataset_name}' not found in one of the files.")
                    continue

                features_dataset = f_features[feature_dataset_path]
                labels_dataset = f_labels[label_dataset_path]

                feature_data = features_dataset[:]
                if labels_dataset.shape == ():
                    label_data = labels_dataset[()]
                else:
                    label_data = labels_dataset[:]

                print(f"Label for {dataset_name}: {label_data}")
                feature_data_list.append(feature_data)  # Collect feature data

            except KeyError as e:
                print(f"An error occurred while processing dataset '{dataset_name}': {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing dataset '{dataset_name}': {e}")

    # Display the collected feature data in a grid
    display_feature_images_grid(feature_data_list, 5, 5)

# Example usage
display_samples_from_matching_datasets(matching_datasets, output_features_file_path, output_labels_file_path, num_samples=10)
