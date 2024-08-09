
"""
This is a cleaning and reshaping code
set the needed columns and the new shape
new shape must match the number of columns you keep and the original shape

"""
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

dataset_name="17-21"
new_dataset_name="17-21_All"

clean = 'yes'

reshape = 'yes'

# (rows, columns, channels)
new_shape_features = (20,15,15)
new_shape_labels = (20,15,1)

##############

dataset_dir = 'C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/' + dataset_name + '/'
new_dataset_dir = 'C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset' + dataset_name + '/' + new_dataset_name

file_path = hdf5_file_path = dataset_dir + dataset_name + '.h5'


features_dir = dataset_dir + '/Features'
labels_dir = dataset_dir + '/Labels'

features_patches_dir = new_dataset_dir + '/Features'
labels_patches_dir = new_dataset_dir + '/Labels'

features_file_path = dataset_dir + new_dataset_name + '_Features.h5'
labels_file_path  = dataset_dir + new_dataset_name + '_Labels.h5'

reshaped_features_file_path = dataset_dir + new_dataset_name + '_Features_Reshaped.h5'
reshaped_labels_file_path  = dataset_dir + new_dataset_name + '_Labels_Reshaped.h5'




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


# Clean unnecessary columns and save to new files function
def clean_hdf5_data(category, preserve_columns, split_columns, remove_split_columns, suffixes, hdf5_file_path, new_hdf5_file_path):
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

def reshape_hdf5_data(category, new_shape, hdf5_file_path, reshaped_hdf5_file_path):
    """
    Reshapes datasets in an HDF5 file and saves them to a new HDF5 file.

    Parameters:
    - category (str): Main group name in the HDF5 file (e.g., 'Features' or 'Labels').
    - new_shape (tuple): The new shape for reshaping the data (rows, cols, channels).
    - hdf5_file_path (str): Path to the input HDF5 file.
    - reshaped_hdf5_file_path (str): Path to the output reshaped HDF5 file.

    Returns:
    - None
    """
    with h5py.File(hdf5_file_path, 'r') as h5file, h5py.File(reshaped_hdf5_file_path, 'w') as new_h5file:
        main_group = h5file[category]
        print(f"Main group '{category}' contains: {list(main_group.keys())}")

        # Recreate the main group
        if category in new_h5file:
            del new_h5file[category]
        new_main_group = new_h5file.create_group(category)

        for folder_suffix in main_group:
            print(f"Trying to access subgroup '{folder_suffix}' in main group '{category}'")
            if folder_suffix not in main_group:
                print(f"Subgroup '{folder_suffix}' not found in main group '{category}'")
                continue

            sub_group = main_group[folder_suffix]
            print(f"Subgroup '{folder_suffix}' contains: {list(sub_group.keys())}")

            # Recreate the subgroup only if it doesn't already exist
            if folder_suffix in new_main_group:
                new_sub_group = new_main_group[folder_suffix]
            else:
                new_sub_group = new_main_group.create_group(folder_suffix)

            for sheet in list(sub_group):
                if isinstance(sub_group[sheet], h5py.Group):
                    continue  # Skip if it's a group, not a dataset

                dataset = sub_group[sheet]
                print(f"Attributes for dataset '{sheet}': {list(dataset.attrs.keys())}")

                # Check if the dataset is empty
                if dataset.size == 0:
                    print(f"Dataset '{sheet}' in subgroup '{folder_suffix}' is empty. Skipping.")
                    continue

                data = dataset[:]

                # Reshape the data
                reshaped_data = data.reshape(new_shape, order='F')

                # Save the reshaped dataset to the new HDF5 file
                new_dataset = new_sub_group.create_dataset(sheet, data=reshaped_data)
                print(f'Reshaped and saved {sheet} to new HDF5 file')


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



########### MAIN CODE ###############

# cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU")

print(torch.__version__)
print(torch.version.cuda)


# check if file in file_path exits
if os.path.exists(file_path):
    print(file_path)
    print("File exists")
else:
    print(file_path)
    print("File does not exist")

# Run the Clean
if clean == 'yes':
    ## Features
    clean_hdf5_data('Features', preserve_columns_features, split_columns_features, remove_split_columns, suffixes, file_path, features_file_path)

    ## Labels
    clean_hdf5_data('Labels', preserve_columns_labels, split_columns_labels, remove_split_columns, suffixes, file_path, labels_file_path)





if reshape == 'yes':
    reshape_hdf5_data('Labels', new_shape_labels, labels_file_path, reshaped_labels_file_path)
    reshape_hdf5_data('Features', new_shape_features, features_file_path, reshaped_features_file_path)











