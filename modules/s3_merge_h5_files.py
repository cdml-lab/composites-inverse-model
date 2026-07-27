import h5py
from pathlib import Path


def s3_merge_h5_files(h5_file_paths, output_file_path):
    """
    Merges multiple HDF5 files into a single file while preserving their original dataset structure.

    Parameters:
    - h5_file_paths (list of str): List of input HDF5 file paths to be merged.
    - output_file_path (str): Path for the merged output HDF5 file.

    Returns:
    - str: Filepath of the merged HDF5 file.
    """

    output_file_path = Path(output_file_path)

    # ✅ Ensure the directory for the merged file exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ Create an empty HDF5 file before writing
    with h5py.File(output_file_path, 'w') as merged_h5:
        pass  # Just to ensure the file is created

    with h5py.File(output_file_path, 'a') as merged_h5:  # Open in append mode
        for file_path in h5_file_paths:
            file_path = Path(file_path)  # Ensure paths are valid
            if not file_path.exists():
                raise FileNotFoundError(f"Error: HDF5 file not found - {file_path}")

            with h5py.File(file_path, 'r') as h5f:
                def copy_group(src_group, dest_group):
                    """
                    Recursively copy groups and datasets while preserving structure.
                    """
                    for key, item in src_group.items():
                        if isinstance(item, h5py.Group):  # If it's a group, create and recurse
                            if key not in dest_group:
                                new_group = dest_group.create_group(key)
                            else:
                                new_group = dest_group[key]
                            copy_group(item, new_group)
                        elif isinstance(item, h5py.Dataset):  # If it's a dataset, copy it
                            if key not in dest_group:  # Prevent overwriting existing datasets
                                dest_group.create_dataset(key, data=item[...])

                # Copy all top-level groups
                copy_group(h5f, merged_h5)

    return str(output_file_path)

# Example usage
if __name__ == "__main__":
    print("merge")
    # input_files = ["dataset1.h5", "dataset2.h5", "dataset3.h5"]
    # merged_file = "merged_dataset.h5"
    #
    # result_path = s3_merge_h5_files(input_files, merged_file)
    # print(f"Merged file saved at: {result_path}")
