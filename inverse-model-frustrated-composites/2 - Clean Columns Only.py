
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
import time



print("GPU available: {}".format(torch.cuda.is_available()))

# Set starting information

##############

dataset_name="30-35"
new_dataset_name="30-35_Normal"

clean = 'yes'
reshape = 'yes'


# (rows, columns, channels)
new_shape_features = (20,15,3)
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

# Max and Min Curvature:
# preserve_columns_features = ['Max Curvature Length','Max Curvature Direction', 'Min Curvature Length', 'Min Curvature Direction',]  # columns to preserve for features
# Max Curvature - no length
# preserve_columns_features = ['Max Curvature Direction']  # columns to preserve for features
# All:
# preserve_columns_features = ['Movement Vector Direction', 'Max Curvature Direction', 'Min Curvature Direction', 'Movement Vector Length', 'Max Curvature Length', 'Min Curvature Length', 'Location X', 'Location Y', 'Location Z', 'Normal Vector', 'U Vector', 'V Vector']  # columns to preserve for features
# XYZ
# preserve_columns_features = ['Location X', 'Location Y', 'Location Z']
# All usefull(able to calculate for inverse):
preserve_columns_features = ['Max Curvature Direction', 'Min Curvature Direction', 'Max Curvature Length', 'Min Curvature Length', 'Normal Vector', 'U Vector', 'V Vector']  # columns to preserve for features
# Normal only
preserve_columns_features = ['Normal Vector']  # columns to preserve for features

preserve_columns_labels = ['Top Angle']  # columns to preserve for labels

# Columns to split and their new column names
split_columns_features = {
    'Movement Vector Direction': ['MVD-X', 'MVD-Y', 'MVD-Z'],
    'Max Curvature Direction': ['MaCD-X', 'MaCD-Y', 'MaCD-Z'],
    'Min Curvature Direction': ['MiCD-X', 'MiCD-Y', 'MiCD-Z'],
    'Normal Vector' : ['No-X', 'No-Y', 'No-Z'],
    'U Vector' : ['U-X', 'U-Y', 'U-Z'],
    'V Vector' : ['V-X', 'V-Y', 'V-Z']
}

split_columns_labels = {}  # Assuming no split columns for Labels
# Max Curvature:
# remove_split_columns = ['MVD-X', 'MVD-Y', 'MVD-Z','MiCD-X', 'MiCD-Y', 'MiCD-Z', 'No-X', 'No-Y', 'No-Z','U-X', 'U-Y', 'U-Z','V-X', 'V-Y', 'V-Z'] # Curvature
# Max and Min Curvature:
# remove_split_columns = ['MVD-X', 'MVD-Y', 'MVD-Z', 'No-X', 'No-Y', 'No-Z','U-X', 'U-Y', 'U-Z','V-X', 'V-Y', 'V-Z'] # Curvature
# remove_split_columns = [] # All
# Location:
# remove_split_columns = ['MaCD-X', 'MaCD-Y', 'MaCD-Z', 'MVD-X', 'MVD-Y', 'MVD-Z','MiCD-X', 'MiCD-Y', 'MiCD-Z', 'No-X', 'No-Y', 'No-Z','U-X', 'U-Y', 'U-Z','V-X', 'V-Y', 'V-Z'] # All
# All useful(able to calculate for inverse):
# remove_split_columns = ['MVD-X', 'MVD-Y', 'MVD-Z'] # Curvature
# Normal only
remove_split_columns = ['MaCD-X', 'MaCD-Y', 'MaCD-Z', 'MVD-X', 'MVD-Y', 'MVD-Z','MiCD-X', 'MiCD-Y', 'MiCD-Z' ,'U-X', 'U-Y', 'U-Z','V-X', 'V-Y', 'V-Z'] # All


suffixes = ['Train', 'Test']



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

        new_h5file.close()
        time.sleep(0.1)  # Adding a delay to ensure file closure before next operation

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


class HDF5DataAugmentor:
    def __init__(self, features_file_path, labels_file_path, rotation_degree=180, apply_mirroring=True):
        self.features_file_path = features_file_path
        self.labels_file_path = labels_file_path
        self.rotation_degree = rotation_degree
        self.apply_mirroring = apply_mirroring

    def _mirror_data(self, data, labels):
        """
        Mirrors the data and labels across the vertical axis without changing the shape.

        Parameters:
        - data (np.array): The feature data to be mirrored.
        - labels (np.array): The label data to be mirrored.

        Returns:
        - mirrored_data (np.array): The mirrored feature data.
        - mirrored_labels (np.array): The mirrored label data.
        """
        print(f"Original data shape: {data.shape}, Original labels shape: {labels.shape}")

        # Mirror the data and labels across the vertical axis (axis 2)
        mirrored_data = np.flip(data, axis=2)
        mirrored_labels = np.flip(labels, axis=2)

        print(f"Mirrored data shape: {mirrored_data.shape}, Mirrored labels shape: {mirrored_labels.shape}")
        return mirrored_data, mirrored_labels

    def _rotate_data(self, data, labels, rotation_increment=90):
        """
        Rotates the data and labels by a specified angle without changing the shape.

        Parameters:
        - data (np.array): The feature data to be rotated.
        - labels (np.array): The label data to be rotated.
        - rotation_increment (int): The angle by which to rotate the data. Default is 90 degrees.

        Returns:
        - rotated_data (np.array): The rotated feature data.
        - rotated_labels (np.array): The rotated label data.
        """
        rotations = []
        rotated_labels = []
        num_rotations = 360 // rotation_increment  # Number of rotations to perform

        for i in range(1, num_rotations):
            # Rotate the data and labels by the specified increment
            rotated_data = np.rot90(data, k=i, axes=(1, 2))

            # Ensure labels rotation maintains the expected shape
            rotated_label = np.rot90(labels, k=i, axes=(1, 2))

            # If rotating the labels changes their shape, swap axes to maintain the original shape
            if rotated_label.shape != labels.shape:
                rotated_label = np.swapaxes(rotated_label, 1, 2)

            print(
                f"Rotation {i}: rotated data shape: {rotated_data.shape}, rotated labels shape: {rotated_label.shape}")

            rotations.append(rotated_data)
            rotated_labels.append(rotated_label)

        rotated_data = np.concatenate(rotations, axis=0)
        rotated_labels = np.concatenate(rotated_labels, axis=0)
        return rotated_data, rotated_labels

    def augment_data(self, data, labels):
        """
        Augments the data by mirroring and rotating, then returns the augmented data and labels.

        Parameters:
        - data (np.array): The feature data to be augmented.
        - labels (np.array): The label data to be augmented.

        Returns:
        - augmented_data (np.array): The augmented feature data.
        - augmented_labels (np.array): The augmented label data.
        """
        print(f"Initial data shape: {data.shape}, Initial labels shape: {labels.shape}")

        augmented_data = data.copy()
        augmented_labels = labels.copy()

        # Apply mirroring
        mirrored_data, mirrored_labels = self._mirror_data(data, labels)
        augmented_data = np.concatenate((augmented_data, mirrored_data), axis=0)
        augmented_labels = np.concatenate((augmented_labels, mirrored_labels), axis=0)
        print(f"After mirroring: data shape: {augmented_data.shape}, labels shape: {augmented_labels.shape}")

        # Apply rotation
        rotated_data, rotated_labels = self._rotate_data(data, labels)
        augmented_data = np.concatenate((augmented_data, rotated_data), axis=0)
        augmented_labels = np.concatenate((augmented_labels, rotated_labels), axis=0)
        print(f"After rotation: data shape: {augmented_data.shape}, labels shape: {augmented_labels.shape}")

        return augmented_data, augmented_labels

    def remove_duplicates(self, data, labels):
        """ Remove duplicate samples based on labels. """
        flat_labels = labels.reshape(labels.shape[0], -1)
        _, unique_indices = np.unique(flat_labels, axis=0, return_index=True)
        unique_data = data[unique_indices]
        unique_labels = labels[unique_indices]
        return unique_data, unique_labels

    def save_augmented_data(self):
        """ Save the augmented features and labels to new HDF5 files. """
        features_augmented_path = self.features_file_path.replace('.h5', '_aug.h5')
        labels_augmented_path = self.labels_file_path.replace('.h5', '_aug.h5')

        # Process 'Features' and 'Labels' separately
        self._process_and_save(self.features_file_path, features_augmented_path, self.labels_file_path,
                               labels_augmented_path)

        print(f"Augmented data saved to {features_augmented_path} and {labels_augmented_path}")

    def _process_and_save(self, features_file_path, features_augmented_path, labels_file_path, labels_augmented_path):
        """
        Processes and saves the augmented data for 'Features' and 'Labels' separately.

        Parameters:
        - features_file_path (str): Path to the input HDF5 file for features.
        - features_augmented_path (str): Path to the output augmented HDF5 file for features.
        - labels_file_path (str): Path to the input HDF5 file for labels.
        - labels_augmented_path (str): Path to the output augmented HDF5 file for labels.
        """
        with h5py.File(features_file_path, 'r') as f_features, h5py.File(labels_file_path, 'r') as f_labels:
            with h5py.File(features_augmented_path, 'w') as f_features_aug, h5py.File(labels_augmented_path,
                                                                                      'w') as f_labels_aug:
                for category, h5file, new_h5file in [('Features', f_features, f_features_aug),
                                                     ('Labels', f_labels, f_labels_aug)]:
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
                            if category == 'Features':
                                labels_data = f_labels['Labels'][folder_suffix][sheet][:]
                                features_data = data
                            else:
                                labels_data = data
                                features_data = f_features['Features'][folder_suffix][sheet][:]

                            # Augment data while keeping resolution
                            aug_features_data, aug_labels_data = self.augment_data(features_data, labels_data)

                            # Remove duplicates
                            unique_features, unique_labels = self.remove_duplicates(aug_features_data, aug_labels_data)

                            # Concatenate original and augmented data
                            combined_features = np.concatenate((features_data, unique_features), axis=0)
                            combined_labels = np.concatenate((labels_data, unique_labels), axis=0)

                            # Save the combined original and augmented data to the new HDF5 file
                            if category == 'Features':
                                new_sub_group.create_dataset(sheet, data=combined_features)
                            else:
                                new_sub_group.create_dataset(sheet, data=combined_labels)

                            print(f"Augmented and saved '{sheet}' to new HDF5 file in '{category}' category.")

                print(f"Augmented data saved to {features_augmented_path} and {labels_augmented_path}")


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

# # Instantiate and run the augmentor
# if augmentation == 'yes':
#     augmentor = HDF5DataAugmentor(features_file_path=reshaped_features_file_path,
#                                   labels_file_path=reshaped_labels_file_path,
#                                   rotation_degree=180, apply_mirroring=True)
#     augmentor.save_augmented_data()






