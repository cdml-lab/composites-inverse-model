# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                 Imports                                   |
# └───────────────────────────────────────────────────────────────────────────┘

from pathlib import Path
from s1_convert_excel_to_h5 import s1_convert_excel_to_h5
from s2_clean_and_reshape_h5 import s2_clean_and_reshape_h5
from s3_merge_h5_files import s3_merge_h5_files
import torch
import os

PINK = "\033[95m"  # Light magenta (pink)
RESET = "\033[0m"   # Reset color to default


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                 Definitions                               |
# └───────────────────────────────────────────────────────────────────────────┘

# Define input parameters
datasets = {
    "40": (20, 20, 8),
    "41": (20, 20, 8),
    "42": (30, 20, 8),
    "43": (30, 20, 8),
    "44": (30, 30, 8),
    "45": (30, 30, 8),
    "46": (40, 30, 8),
    "47": (40, 30, 8),
    "48": (40, 40, 8),
    "49": (40, 40, 8)
}

dataset_name = "40-49"

num_of_labels = 1

convert = True
clean = True
reshape = True
merge = True
clean_or_reshape = True
delete_unused = True

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                               Functions                                   |
# └───────────────────────────────────────────────────────────────────────────┘

def delete_unwanted_files(base_folder, files_to_keep):
    """
    Deletes all files in the specified folder except for the provided list of files.

    Parameters:
    - base_folder (str or Path): The folder where files should be deleted.
    - files_to_keep (list of str or Path): List of file paths to be preserved.

    Returns:
    - None
    """
    base_folder = Path(base_folder)  # Ensure base_folder is a Path object
    files_to_keep = {Path(file).resolve() for file in files_to_keep}  # Convert to set of resolved Path objects

    if not base_folder.exists() or not base_folder.is_dir():
        print(f"Error: {base_folder} is not a valid directory.")
        return

    # Iterate over all files in the directory
    for file in base_folder.iterdir():
        if file.is_file() and file.resolve() not in files_to_keep:
            try:
                file.unlink()  # Delete file
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Main Code                                       |
# └───────────────────────────────────────────────────────────────────────────┘

print(f"prints from the master file will be in {PINK}PINK")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU")



# Get the script's directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Base directory (datasets are parallel to the code folder)
base_dir = project_root / "frustrated-composites-dataset"

h5_files = []

print(f"{PINK}Datasets: {datasets}")
print(f"Base Directory: {base_dir}{RESET}")


# Step 1: Convert Excel to HDF5
if convert:
    for name, shape in datasets.items():
        print(f"{PINK}Processing {name} with shape {shape}...{RESET}")  # ✅ Debugging Step

        # Define input and output files
        input_files_list = [f"{base_dir}/Dataset_Input_{name}.xlsx"]
        output_files_list = [f"{base_dir}/Dataset_Output_{name}.xlsx"]


        h5_path = s1_convert_excel_to_h5(name, base_dir, input_files_list, output_files_list, [90,10], ['Train', 'Test'], dataset_name)
        h5_files.append(h5_path)

    print(f"{PINK} h5 files: {h5_files}")
    print(f"{PINK} Converting to h5 successful {RESET}")

# Step 2: Clean and Reshape HDF5
all_labels =[]
all_features = []

if clean_or_reshape:
    for i, (name, shape) in enumerate(datasets.items()):
        file = h5_files[i]
        features_file_path = f"{base_dir}/{dataset_name}/{name}_Features.h5"
        labels_file_path = f"{base_dir}/{dataset_name}/{name}_Labels.h5"
        reshaped_h5_features, reshaped_h5_labels = s2_clean_and_reshape_h5(base_dir, file, clean= clean, reshape= reshape,
                                                    features_file_path=features_file_path, labels_file_path=labels_file_path,
                                                    new_shape_labels=(*shape[:-1], num_of_labels),new_shape_features=shape,
                                                    suffixes=['Train', 'Test'])
        all_labels.append(reshaped_h5_labels)
        all_features.append(reshaped_h5_features)
    print(f"{PINK} Cleaning and reshaping successful {RESET}")


print(all_labels)
print(all_features)
# Step 3: Merge HDF5 Files
if merge:
    merged_labels_path = f"{base_dir}/{dataset_name}/{dataset_name}_Merged_Labels.h5"
    merged_features_path = f"{base_dir}/{dataset_name}/{dataset_name}_Merged_Features.h5"

    merged_labels = s3_merge_h5_files(all_labels, merged_labels_path)
    merged_features = s3_merge_h5_files(all_features, merged_features_path)

    print(f"{PINK} Merge Successful {RESET}")

if delete_unused:
    base_folder = f"{base_dir}/{dataset_name}"
    files_to_keep = [
        merged_labels_path,
        merged_features_path
    ]

    print(f"{PINK} Deleting everything in folder {base_folder} except {files_to_keep}")
    delete_unwanted_files(base_folder, files_to_keep)
    print("Delete successful")

