import h5py
import os
import pandas as pd
import numpy as np
import time

# Set directories and dataset names
dataset_dir = 'C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/17-24/'
output_dir = dataset_dir + 'Combined_Features/'

h = 20
w = 15

reshape = 'yes'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Path to the original HDF5 file (renamed to 17-24_All_Features.h5)
hdf5_file_path = dataset_dir + '17-24_All_Features.h5'

# Feature groups to create separate files for each group
feature_groups = {
    # 'Movement_Features': ['MVD-X', 'MVD-Y', 'MVD-Z', 'Movement Vector Length'],
    # 'Curvature_Features': ['MaCD-X', 'MaCD-Y', 'MaCD-Z', 'Max Curvature Length', 'Min Curvature Length'],
    # 'Location_Features': ['Location X', 'Location Y', 'Location Z']
    'Curvature_Max_Length_X_Y_Movement_Z' : ['Max Curvature Length','MaCD-X', 'MaCD-Y','MVD-Z'],
    }

# Function to map the columns in the dataset to the correct feature names
def extract_features_from_dataset(dataset, feature_names):
    """Extract features from the dataset based on columns metadata."""
    if 'columns' not in dataset.attrs:
        print(f"Warning: Dataset does not have 'columns' attribute. Skipping.")
        return None

    # Retrieve columns metadata and convert to a DataFrame
    columns_metadata = dataset.attrs['columns']
    df = pd.DataFrame(dataset[:], columns=columns_metadata)

    # Print all column names before processing
    print(f'Columns in dataset before processing: {list(df.columns)}')

    # Extract the relevant features based on feature_names
    extracted_data = {}
    for feature in feature_names:
        if feature in df.columns:
            extracted_data[feature] = df[feature].to_numpy()
        else:
            print(f"Warning: Feature '{feature}' not found in dataset columns.")

    return extracted_data

# Function to combine features from each iteration into separate HDF5 files for each feature group
def combine_features_to_files(hdf5_file_path, output_dir, feature_groups):
    with h5py.File(hdf5_file_path, 'r') as h5file:
        # Assuming the main group for features is 'Features'
        main_group = h5file['Features']
        print(f"Main group 'Features' contains: {list(main_group.keys())}")

        # Create separate HDF5 files for each feature group
        for group_name, features in feature_groups.items():
            output_file_path = os.path.join(output_dir, f'{group_name}.h5')

            # Create a new HDF5 file for each feature group
            with h5py.File(output_file_path, 'w') as new_h5file:
                new_group = new_h5file.create_group('Features')

                for subgroup in main_group:  # 'Test', 'Train'
                    sub_group = main_group[subgroup]
                    print(f"Processing subgroup: {subgroup}")

                    # Create the 'Train' or 'Test' subgroup in the new HDF5 file
                    if subgroup not in new_group:
                        new_sub_group = new_group.create_group(subgroup)
                    else:
                        new_sub_group = new_group[subgroup]

                    for iteration in sub_group:  # '17_Iteration 10', '17_Iteration 108', etc.
                        iteration_group = sub_group[iteration]

                        # Check if iteration_group is a dataset
                        if isinstance(iteration_group, h5py.Dataset):
                            process_dataset(iteration_group, new_sub_group, subgroup, iteration, iteration, features, group_name, output_file_path)
                        else:
                            print(f"Skipping non-dataset item: {iteration} in subgroup {subgroup}")

            if reshape == 'yes':
                new_shape_features = (h, w, len(features))
                reshaped_features_file_path = os.path.join(output_dir, f'{group_name}_Reshaped.h5')
                reshape_hdf5_data('Features', new_shape_features, output_file_path, reshaped_features_file_path)

    print(f"Feature combining complete. Separate files saved in {output_dir}.")

# Helper function to process datasets
def process_dataset(dataset, new_group, subgroup, iteration, dataset_name, features, group_name, output_file_path):
    """Helper function to process and save datasets."""
    print(f"Attributes for dataset '{dataset_name}': {list(dataset.attrs.keys())}")

    # Extract and process the features based on the columns metadata
    feature_data = extract_features_from_dataset(dataset, features)
    if feature_data is None:
        return  # Skip if no valid feature data was extracted

    combined_data = None

    for feature_name in features:
        if feature_name in feature_data:
            data = feature_data[feature_name]
            # Combine the features along the last axis
            if combined_data is None:
                combined_data = data[:, np.newaxis]  # Add new axis for concatenation
            else:
                combined_data = np.concatenate((combined_data, data[:, np.newaxis]), axis=-1)
        else:
            print(f'Feature {feature_name} not found in dataset {dataset_name} of iteration {iteration}')

    # Save the combined feature data directly as a dataset
    if combined_data is not None:
        # Ensure the dataset is created in the correct Train/Test group
        new_group.create_dataset(f'{iteration}', data=combined_data)
        print(f'Saved combined features {features} from {dataset_name} in iteration {iteration} into {output_file_path}')

# Function to reshape datasets
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

                # Save the reshaped dataset to the new HDF5 file in the main group
                new_sub_group.create_dataset(sheet, data=reshaped_data)
                print(f'Reshaped and saved {sheet} to new HDF5 file')

        new_h5file.close()
        time.sleep(0.1)  # Adding a delay to ensure file closure before the next operation

# Run the feature combining
combine_features_to_files(hdf5_file_path, output_dir, feature_groups)
