import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import os


# Function to read datasets from H5 file
def read_datasets(h5_file, groups):
    data = []
    column_names = []
    for group in groups:
        for subgroup in h5_file[group]:
            for dataset_name in h5_file[group][subgroup]:
                print(f"Reading {group}/{subgroup}/{dataset_name} ...")
                dataset = h5_file[group][subgroup][dataset_name]
                print(f"Dataset shape: {dataset.shape}")
                if dataset.shape[1] == 15:  # Check for expected shape
                    data.append(dataset[:])
                else:
                    print(f"Skipping dataset due to incompatible size: {dataset.shape}")
        if 'columns' in h5_file[group].attrs:
            column_names = list(h5_file[group].attrs['columns'])
    return data, column_names


# Function to perform analysis on the original data
def analyze_data(data):
    print(f"Number of datasets: {len(data)}")
    combined_data = np.vstack(data)
    print(f"Combined data shape: {combined_data.shape}")
    median_data = np.median(combined_data, axis=0)
    average_data = np.mean(combined_data, axis=0)
    maximum_data = np.max(combined_data, axis=0)
    abs_median_data = np.median(np.abs(combined_data), axis=0)
    abs_average_data = np.mean(np.abs(combined_data), axis=0)
    abs_maximum_data = np.max(np.abs(combined_data), axis=0)
    return median_data, average_data, maximum_data, abs_median_data, abs_average_data, abs_maximum_data


# Function to reshape data for visualization
def reshape_data_for_visualization(data, x, y):
    if data.size == x * y:
        reshaped_data = data.reshape(x, y, order='F')
        return reshaped_data
    else:
        print(f"Cannot reshape data of size {data.size} to shape ({x}, {y})")
        return None


# Function to display and save results with numbers on pixels
def display_and_save_results(median_data, average_data, maximum_data, abs_median_data, abs_average_data,
                             abs_maximum_data, x, y, title_prefix="", colormap='viridis'):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    titles = [f'{title_prefix} Median', f'{title_prefix} Average', f'{title_prefix} Maximum',
              f'{title_prefix} Abs Median', f'{title_prefix} Abs Average', f'{title_prefix} Abs Maximum']
    data = [median_data, average_data, maximum_data, abs_median_data, abs_average_data, abs_maximum_data]

    for ax, title, img_data in zip(axs.flatten(), titles, data):
        ax.set_title(title)
        reshaped_img_data = reshape_data_for_visualization(img_data, x, y)
        if reshaped_img_data is not None:
            # Normalize data for display
            img_data_normalized = (reshaped_img_data - np.min(reshaped_img_data)) / (
                        np.max(reshaped_img_data) - np.min(reshaped_img_data))
            ax.imshow(img_data_normalized, cmap=colormap, aspect='auto')

            # Add numbers to pixels
            for i in range(0, reshaped_img_data.shape[0], 5):
                for j in range(0, reshaped_img_data.shape[1], 5):
                    ax.text(j, i, f'{reshaped_img_data[i, j]:.2f}', ha='center', va='center', color='white', fontsize=6)
        else:
            ax.text(0.5, 0.5, 'Cannot reshape data', horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(f'{title_prefix}_analysis.png')
    plt.close()


# Function to display and save 5 random datasets with reshaping
def display_and_save_random_datasets(data, column_names, x, y):
    selected_indices = random.sample(range(len(data)), 5)
    for idx in selected_indices:
        dataset = data[idx]
        fig, axs = plt.subplots(3, 5, figsize=(15, 10))
        for col in range(dataset.shape[1]):
            column_data = dataset[:, col]
            reshaped_column_data = reshape_data_for_visualization(column_data, x, y)
            ax = axs[col // 5, col % 5]
            if reshaped_column_data is not None:
                ax.imshow(reshaped_column_data, cmap='gray', aspect='auto')
                # Add numbers to pixels
                for i in range(0, reshaped_column_data.shape[0], 5):
                    for j in range(0, reshaped_column_data.shape[1], 5):
                        ax.text(j, i, f'{reshaped_column_data[i, j]:.2f}', ha='center', va='center', color='white',
                                fontsize=6)
            else:
                ax.text(0.5, 0.5, f'Cannot reshape column {col}', horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
            ax.set_title(column_names[col].decode('utf-8') if column_names else f'Column {col + 1}')
        fig.suptitle(f'Dataset {idx}')
        plt.tight_layout()
        plt.savefig(f'Dataset_{idx}_samples.png')
        plt.close()


# Main function
def main(features_file, x, y, c):
    # Read features
    with h5py.File(features_file, 'r') as h5_file:
        groups = ['Features']
        features_data, column_names = read_datasets(h5_file, groups)

    # Display and save 5 random datasets
    display_and_save_random_datasets(features_data, column_names, x, y)

    # Perform and display analysis for each column separately
    for col in range(features_data[0].shape[1]):
        title_prefix = column_names[col].decode('utf-8') if column_names else f'Column {col + 1}'
        column_data = np.vstack([d[:, col] for d in features_data])
        print(f"Column {col} combined data shape: {column_data.shape}")
        median_data, average_data, maximum_data, abs_median_data, abs_average_data, abs_maximum_data = analyze_data(
            [column_data])
        display_and_save_results(median_data, average_data, maximum_data, abs_median_data, abs_average_data,
                                 abs_maximum_data, x, y, title_prefix=title_prefix, colormap='gray')


# Example usage
og_dataset_name = "14"
dataset_name = "14_All_Features"

features_file = f'C:/Gal_Msc/Dataset/{og_dataset_name}/{dataset_name}.h5'
x = 20
y = 15
c = 3
main(features_file, x, y, c)
