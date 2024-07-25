import h5py
from collections import defaultdict
from colorama import Fore, Style, init

init(autoreset=True)


def find_duplicates(h5file):
    def visit_func(name, node):
        if isinstance(node, h5py.Dataset):
            dataset_name = name.split('/')[-1]
            if node.shape == ():  # Check if the dataset is scalar
                dataset_content = node[()]
            else:
                dataset_content = node[:]
            dataset_key = dataset_content.tobytes() if not node.dtype == 'object' else str(dataset_content)
            if dataset_key in datasets_dict:
                duplicates[dataset_key].append(name)
            else:
                datasets_dict[dataset_key] = name
                duplicates[dataset_key] = [name]

    with h5py.File(h5file, 'r') as f:
        datasets_dict = {}
        duplicates = defaultdict(list)
        f.visititems(visit_func)

    return duplicates


def print_duplicates(duplicates):
    total_datasets = 0
    unique_datasets = 0
    total_duplicates = 0

    for dataset_key, names in duplicates.items():
        total_datasets += len(names)
        if len(names) > 1:
            total_duplicates += len(names) - 1
        else:
            unique_datasets += 1

    unique_datasets = len(duplicates)

    if total_duplicates == 0:
        print(Fore.GREEN + Style.BRIGHT + "No duplicates found.")
    else:
        print(Fore.RED + Style.BRIGHT + "Duplicate datasets found:")
        for dataset_key, names in duplicates.items():
            if len(names) > 1:
                print(f"Datasets: {Fore.YELLOW + ', '.join(names)} are duplicates")

    print("\n" + Fore.CYAN + Style.BRIGHT + f"Total datasets: {total_datasets}")
    print(Fore.CYAN + Style.BRIGHT + f"Total unique datasets: {unique_datasets}")
    print(Fore.CYAN + Style.BRIGHT + f"Total duplicates: {total_duplicates}")



##############

dataset_name="14"
new_dataset_name="14_All_CNN"

###############

dataset_dir = "C:/Gal_Msc/Dataset/" + dataset_name + '/'

features_file_path = dataset_dir + new_dataset_name + '_Features.h5'
labels_file_path = dataset_dir + new_dataset_name + '_Labels.h5'



# Replace 'your_file.h5' with the path to your HDF5 file
feature_duplicates = find_duplicates(features_file_path)
label_duplicates = find_duplicates(labels_file_path)
print("Features:")
print_duplicates(feature_duplicates)
print("Labels:")
print_duplicates(label_duplicates)