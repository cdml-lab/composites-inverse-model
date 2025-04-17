# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                 Imports                                   |
# └───────────────────────────────────────────────────────────────────────────┘

from pathlib import Path
from modules.smooth_surface_and_compute_curvature import smooth_surface_and_compute_curvature
from modules.smooth_surface_and_compute_curvature import smooth_surface_and_compute_normal
from modules.s3_merge_h5_files import s3_merge_h5_files
from modules.s1_convert_excel_to_h5 import s1_convert_excel_to_h5
from modules.s2_clean_and_reshape_h5 import s2_clean_and_reshape_h5
import torch
import time

PINK = "\033[95m"
RESET = "\033[0m"


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                 Definitions                               |
# └───────────────────────────────────────────────────────────────────────────┘

# # Define input parameters - Y size(bigger size) should be first
# datasets = {
#     "60": (30, 10, 8),
#     "61": (30, 10, 8),
#     "611": (30,10,8), # 1 patch
#     "62": (30, 20, 8),
#     "63": (30, 20, 8),
#     "631": (30, 20, 8),
#     "632": (30, 20, 8),
#     "633": (30, 20, 8), # 1 patch
#     "634": (30, 20, 8),
#     "64": (30, 30, 8),
#     "65": (30, 30, 8),
#     "66": (40, 30, 8),
#     "661": (40, 30, 8),
#     "662": (40, 30, 8),
#     "67": (40, 30, 8),
#     # "68": (40, 40, 8),
#     "69": (40, 40, 8),
#     "70": (40, 20, 8),
#     "701": (40, 20, 8),
#     # "71": (40, 20, 8),
#     # "72": (50, 20, 8),
#     # "73": (50, 20, 8),
#     # "74": (50, 30, 8),
#     # "75": (50, 30, 8),
#     # "76": (50, 40, 8),
#     # "77": (50, 40, 8),
#     # "78": (50, 50, 8),
#     # "79": (50, 50, 8),
#     "82": (30, 20, 8),
#     "83": (30, 20, 8)
# }


# Noraml / XYZ
datasets = {
    "60": (30, 10, 3),
    "61": (30, 10, 3),
    "611": (30,10,3), # 1 patch
    "62": (30, 20, 3),
    "63": (30, 20, 3),
    "631": (30, 20, 3),
    "632": (30, 20, 3),
    "633": (30, 20, 3), # 1 patch
    "634": (30, 20, 3),
    "64": (30, 30, 3),
    "65": (30, 30, 3),
    "66": (40, 30, 3),
    "661": (40, 30, 3),
    "662": (40, 30, 3),
    "67": (40, 30, 3),
    # "68": (40, 40, 3),
    "69": (40, 40, 3),
    "70": (40, 20, 3),
    "701": (40, 20, 3),
    # "71": (40, 20, 3),
    # "72": (50, 20, 3),
    # "73": (50, 20, 3),
    # "74": (50, 30, 3),
    # "75": (50, 30, 3),
    # "76": (50, 40, 3),
    # "77": (50, 40, 3),
    # "78": (50, 50, 3),
    # "79": (50, 50, 3),
    "82": (30, 20, 3),
    "83": (30, 20, 3)
}


# Define input parameters - Y size(bigger size) should be first
# datasets = {
#     "60": (30, 10, 3),
#     "61": (30, 10, 3),
#     "611": (30,10,3), # 1 patch
#     "62": (30, 20, 3),
#     "63": (30, 20, 3),
#     "631": (30, 20, 3),
#     "632": (30, 20, 3),
#     "633": (30, 20, 3), # 1 patch
#     "634": (30, 20, 3),
#     "64": (30, 30, 3),
#     "65": (30, 30, 3),
#     "66": (40, 30, 3),
#     "661": (40, 30, 3),
#     "662": (40, 30, 3),
#     "67": (40, 30, 3),
#     # "68": (40, 40, 8),
#     "69": (40, 40, 3),
#     "70": (40, 20, 3),
#     "701": (40, 20, 3),
#     # "71": (40, 20, 8),
#     # "72": (50, 20, 8),
#     # "73": (50, 20, 8),
#     # "74": (50, 30, 8),
#     # "75": (50, 30, 8),
#     # "76": (50, 40, 8),
#     # "77": (50, 40, 8),
#     # "78": (50, 50, 8),
#     # "79": (50, 50, 8),
#     "82": (30, 20, 3),
#     "83": (30, 20, 3)
# }

# Test
# datasets = {
#     "62": (30, 20, 3),
#     "67": (40, 30, 3)
# }

dataset_name = "62-83-no_smooth_xyz"

num_of_labels = 1

# Only if recalculating curvature
# smoothing_method = 'rebuild' #'savgol' 'bilateral' 'anisotropic' 'uniform' 'gaussian'
# smoothing_methods = [None, 'rebuild', 'gaussian']
smoothing_methods = [None]
# smoothing_methods = [None, 'uniform']
sigma = 1.0
grid_divide = 5 # for rebuild resolution, has no other effect
# Set flags. If set to False it may require adaptations to the code.

recalculate_curvature = False
recalculate_normal = False
convert = True
clean_or_reshape = True
clean = True
reshape = True
merge = True
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

# Record start time
start_time = time.time()

print(f"prints from the master file will be in {PINK}PINK.")
print(f"start time: {start_time}")
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


for name, shape in datasets.items():
    for smoothing_method in smoothing_methods:

        # Skip the raw input variant — no smoothing or curvature needed
        if smoothing_method is None:
            continue

        suffix = smoothing_method
        print(f"{PINK}Recalculating Curvature for {name} [{suffix}]...{RESET}")

        # Always use the original unsmoothed input file
        input_files_list = [f"{base_dir}/Dataset_Output_{name}.xlsx"]

        # Output path includes smoothing suffix (e.g. "_rebuild", "_gaussian")
        output_files_list = [f"{base_dir}/Dataset_Output_{suffix}_{name}.xlsx"]

        # Define surface resolution and smoothing grid
        grid_shape = (shape[0], shape[1])
        rebuild_shape = (grid_shape[0] // grid_divide, grid_shape[1] // grid_divide)

        # Run smoothing + curvature computation
        if recalculate_curvature:
            smooth_surface_and_compute_curvature(
                base_dir,
                input_files_list,
                grid_shape,
                rebuild_shape,
                smoothing_method=smoothing_method,
                sigma=sigma,
                suffix=suffix
            )
        if recalculate_normal:
            smooth_surface_and_compute_normal(
                base_dir,
                input_files_list,
                grid_shape,
                rebuild_shape,
                smoothing_method=smoothing_method,
                sigma=sigma,
                suffix=suffix
            )

# Step 1: Convert Excel to HDF5
if convert:
    for name, shape in datasets.items():
        print(f"{PINK}Processing {name} with shape {shape}...{RESET}")  # ✅ Debugging Step

        split_indices = None  # Will store and reuse the split across all variants

        for i, smoothing_method in enumerate(smoothing_methods):
            print(f"{PINK}Processing {name} [{smoothing_method}]...{RESET}")

            # Define the input file (raw always the same)
            input_files_list = [f"{base_dir}/Dataset_Input_{name}.xlsx"]
            if smoothing_method is None:
                output_files_list = [f"{base_dir}/Dataset_Output_{name}.xlsx"]
            else:
                output_files_list = [f"{base_dir}/Dataset_Output_{smoothing_method}_{name}.xlsx"]

            # Use 'original' to generate the split, reuse it for others
            if i == 0:
                h5_path, split_indices = s1_convert_excel_to_h5(
                    name, base_dir, input_files_list, output_files_list,
                    [90, 10], ['Train', 'Test'], dataset_name,
                    split_indices=None,
                    smoothing_method=smoothing_method
                )
            else:
                h5_path, _ = s1_convert_excel_to_h5(
                    name, base_dir, input_files_list, output_files_list,
                    [90, 10], ['Train', 'Test'], dataset_name,
                    split_indices=split_indices,
                    smoothing_method=smoothing_method
                )

            h5_files.append((name, smoothing_method, h5_path))

    print(f"{PINK} h5 files: {h5_files}")
    print(f"{PINK} Converting to h5 successful {RESET}")

# Step 2: Clean and Reshape HDF5
all_labels =[]
all_features = []

if clean_or_reshape:
    for name, shape in datasets.items():
        for smoothing_method in smoothing_methods:
            # Find the corresponding file
            matches = [(n, sm, f) for (n, sm, f) in h5_files if n == name and sm == smoothing_method]
            if not matches:
                continue  # skip if no match found
            _, _, file = matches[0]

            suffix = smoothing_method if smoothing_method else "original"
            features_file_path = f"{base_dir}/{dataset_name}/{name}_{suffix}_Features.h5"
            labels_file_path = f"{base_dir}/{dataset_name}/{name}_{suffix}_Labels.h5"

            reshaped_h5_features, reshaped_h5_labels = s2_clean_and_reshape_h5(
                base_dir, file, clean=clean, reshape=reshape,
                features_file_path=features_file_path,
                labels_file_path=labels_file_path,
                new_shape_labels=(*shape[:-1], num_of_labels),
                new_shape_features=shape,
                suffixes=['Train', 'Test']
            )

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

# Record end time
end_time = time.time()

# Calculate total time
total_time = end_time - start_time

print(f"{PINK}Start Time: {time.ctime(start_time)}")
print(f"End Time: {time.ctime(end_time)}")
print(f"Total Time: {total_time/60:.2f} minutes{RESET}")
