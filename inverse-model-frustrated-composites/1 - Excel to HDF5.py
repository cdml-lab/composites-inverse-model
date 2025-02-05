# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                 Imports                                   |
# └───────────────────────────────────────────────────────────────────────────┘


import h5py
import pandas as pd
import numpy as np
import xlrd
import os
import time
import random
import torch
from pathlib2 import Path



# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                 Definitions                               |
# └───────────────────────────────────────────────────────────────────────────┘


# Dataset Name (Modify as needed)
dataset_name = "40-49"

# Training Split
split_percentages = [90, 10]  # percentages to split, modify as needed. accommodates different splits

suffixes = ['Train', 'Test']  # suffixes for file names


# Get the script's directory
script_dir = Path(__file__).resolve().parent

# Move up one level to the project root and define dataset location
project_root = script_dir.parent

# Base directory (datasets are parallel code folder)
base_dir = project_root / "frustrated-composites-dataset"
hdf5_file_path = base_dir / dataset_name / f"{dataset_name}.h5"

# Manually specify input files
input_files_list = [
    base_dir / "Dataset_Input_40.xlsx",
    base_dir / "Dataset_Input_41.xlsx",
    base_dir / "Dataset_Input_42.xlsx",
    base_dir / "Dataset_Input_43.xlsx",
    base_dir / "Dataset_Input_44.xlsx",
    base_dir / "Dataset_Input_45.xlsx",
    base_dir / "Dataset_Input_46.xlsx",
    base_dir / "Dataset_Input_47.xlsx",
    base_dir / "Dataset_Input_48.xlsx",
    base_dir / "Dataset_Input_49.xlsx",
]

# Manually specify output files 
output_files_list = [
    base_dir / "Dataset_Output_40.xlsx",
    base_dir / "Dataset_Output_41.xlsx",
    base_dir / "Dataset_Output_42.xlsx",
    base_dir / "Dataset_Output_43.xlsx",
    base_dir / "Dataset_Output_44.xlsx",
    base_dir / "Dataset_Output_45.xlsx",
    base_dir / "Dataset_Output_46.xlsx",
    base_dir / "Dataset_Output_47.xlsx",
    base_dir / "Dataset_Output_48.xlsx",
    base_dir / "Dataset_Output_49.xlsx",
]

# Convert paths to strings if necessary
input_files_list = [str(file) for file in input_files_list]
output_files_list = [str(file) for file in output_files_list]


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                               Functions                                   |
# └───────────────────────────────────────────────────────────────────────────┘


def count_datasets(group):
    count = 0
    for item in group.values():
        if isinstance(item, h5py.Dataset):
            count += 1
        elif isinstance(item, h5py.Group):
            count += count_datasets(item)
    return count


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

    print("All files and sheets match. No action needed.")
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


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Main Code                                       |
# └───────────────────────────────────────────────────────────────────────────┘

# Run on CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU")

print(f"train percentage: {split_percentages[0]}")
print(f"test percentage: {split_percentages[1]}")


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