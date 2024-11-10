

import h5py
import pandas as pd
import numpy as np
import xlrd
import os
import time
import random
import torch

"""## ***Manually insert file names!***"""

# Dataset Name and Paths
dataset_name = "30-35"

input_files = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_" + dataset_name + ".xlsx"
output_files = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_" + dataset_name + ".xlsx"

input_files_list = [
    # # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_24.xlsx"
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_17.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_18.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_19.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_20.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_21.xlsx"
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_22.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_23.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_24.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_100.xlsx"
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_27.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_28.xlsx"
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_29.xlsx"
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_30.xlsx",
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_31.xlsx",
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_32.xlsx",
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_33.xlsx",
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_34.xlsx",
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Input_35.xlsx"
    # Add more file paths as needed
]
output_files_list = [
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_24.xlsx"
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_17.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_18.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_19.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_20.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_21.xlsx"
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_22.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_23.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_24.xlsx"
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_TEST2.xlsx"
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_100.xlsx"
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_27.xlsx",
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_28.xlsx"
    # "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_29.xlsx"
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_30.xlsx",
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_31.xlsx",
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_32.xlsx",
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_33.xlsx",
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_34.xlsx",
    "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/Dataset_Output_35.xlsx"
    # Add more file paths as needed
]

base_dir = "C:/Gal_Msc/Ipublic-repo/frustrated-composites-dataset/" + dataset_name + "/"

hdf5_file_path = base_dir + dataset_name + '.h5'

# Size of Samples
size_x = 15
size_y = 20
shape_data = size_x * size_y

# Training Split
split_percentages = [90, 10]  # percentages to split, modify as needed. accommodates different splits
print(f"train percentages: {split_percentages[0]}")
print(f"test percentages: {split_percentages[1]}")

suffixes = ['Train', 'Test']  # suffixes for file names

"""## Separate to Train / Validate / Test functions
* a function for splitting
* runs on inputs and then outputs
"""


# Run on CUDA

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU")


def verify_matching_files_and_sheets(input_files, output_files):
    if len(input_files) != len(output_files):
        print("Error: The number of input files and output files does not match.")
        return False

    for input_file, output_file in zip(input_files, output_files):
        input_file_name = os.path.basename(input_file)
        output_file_name = os.path.basename(output_file)

        # Check if files exist
        if not (os.path.isfile(input_file) and os.path.isfile(output_file)):
            print(f"Error: One of the files '{input_file_name}' or '{output_file_name}' does not exist.")
            return False

        # Load Excel sheets and compare sheet names
        input_sheets = pd.ExcelFile(input_file).sheet_names
        output_sheets = pd.ExcelFile(output_file).sheet_names

        if input_sheets != output_sheets:
            print(f"Error: Sheet names do not match between '{input_file_name}' and '{output_file_name}'.")
            print(f"  Input sheets: {input_sheets}")
            print(f"  Output sheets: {output_sheets}")
            return False
        else:
            print(f"'{input_file_name}' and '{output_file_name}' have matching sheet names.")

    print("All files and sheets match.")
    return True


def split_excel_to_hdf5_multiple(file_paths, split_indices, suffixes, base_dir, category, hdf5_file_path):
    """
    Splits multiple Excel files into specified percentages and saves each split as a group in an HDF5 file.

    Parameters:
    - file_paths (list of str): List of paths to the input Excel files.
    - split_indices (list of list of int): List of indices for each split.
    - suffixes (list of str): List of suffixes for naming the subgroups in HDF5 file (e.g., ['Train', 'Test']).
    - base_dir (str): Base directory for saving the HDF5 file.
    - category (str): Main group name in the HDF5 file (e.g., 'Features' or 'Labels').
    - hdf5_file_path (str): Path to the output HDF5 file.

    Returns:
    - None
    """
    combined_worksheets = []

    # Load all Excel files and append worksheets with file identifiers
    for file_path in file_paths:
        xlsx = pd.ExcelFile(file_path)
        worksheets = xlsx.sheet_names
        for sheet in worksheets:
            df = pd.read_excel(xlsx, sheet_name=sheet)
            df = df.astype(str)
            combined_worksheets.append((file_path, sheet, df))

    # Ensure the directory for the HDF5 file exists
    os.makedirs(os.path.dirname(hdf5_file_path), exist_ok=True)

    # Create or open the HDF5 file
    with h5py.File(hdf5_file_path, 'a') as h5file:
        # Create the main group (e.g., 'Features' or 'Labels')
        main_group = h5file.create_group(category)
        for idx, indices in enumerate(split_indices):
            # Create the subgroup (e.g., 'Train' or 'Test')
            sub_group = main_group.create_group(suffixes[idx])
            for i in indices:
                file_path, sheet_name, df = combined_worksheets[i]

                file_base_name = os.path.basename(file_path)
                # Extract the number after the last underscore and before the file extension
                number_part = file_base_name.rsplit('_', 1)[-1].split('.')[0]
                unique_sheet_name = f"{number_part}_{sheet_name}"

                dataset = sub_group.create_dataset(unique_sheet_name, data=df.to_numpy())
                dataset.attrs['columns'] = df.columns.astype(str).to_list()
            print(f"Saved sheets in group {category}/{suffixes[idx]} : {indices}")


def generate_shuffled_indices(total_sheets, split_percentages):
    indices = list(range(total_sheets))
    random.shuffle(indices)

    split_counts = [int(total_sheets * (split / 100)) for split in split_percentages]
    if sum(split_counts) != total_sheets:
        split_counts[-1] += total_sheets - sum(split_counts)

    split_indices = []
    start = 0
    for count in split_counts:
        end = start + count
        split_indices.append(indices[start:end])
        start = end

    return split_indices


"""## Run the split"""
# Verify that the input files match the output files.
# If they don't, it's likely that the input has a last worksheet that doesn't exist in output
# and should be deleted in Excel
verify_matching_files_and_sheets(input_files_list, output_files_list)

# Combine all worksheets from all files and generate the split indices once
combined_total_sheets = sum(len(pd.ExcelFile(file).sheet_names) for file in output_files_list)
split_indices = generate_shuffled_indices(combined_total_sheets, split_percentages)


# Run the split for both inputs and outputs
for category, files_list in zip(['Labels', 'Features'], [input_files_list, output_files_list]):
    if all(os.path.isfile(file) and file.endswith('.xlsx') for file in files_list):  # Check if all are Excel files
        # Run the split
        split_excel_to_hdf5_multiple(files_list, split_indices, suffixes, base_dir, category, hdf5_file_path)

def count_datasets(group):
    count = 0
    for item in group.values():
        if isinstance(item, h5py.Dataset):
            count += 1
        elif isinstance(item, h5py.Group):
            count += count_datasets(item)
    return count

# Read the HDF5 file and display its structure
with h5py.File(hdf5_file_path, 'r') as h5file:
    # Print the structure of the HDF5 file
    print("Structure:")
    for key in h5file.keys():
        print(f"- {key}")

    # Print the metadata of each group
    print("\nMetadata:")
    for group_name, group in h5file.items():
        print(f"- Group: {group_name}")
        print(f"  Attributes:")
        for attr_name, attr_value in group.attrs.items():
            print(f"    - {attr_name}: {attr_value}")
        print(f"  Datasets:")
        for dataset_name in group:
            print(f"    - {dataset_name}")

    # Print the number of datasets in each group and subgroup
    print("\nNumber of datasets in each group and subgroup:")

    for group_name, group in h5file.items():
        dataset_count = count_datasets(group)
        print(f"- {group_name}: {dataset_count} datasets")