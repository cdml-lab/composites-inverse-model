# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Imports                                         │
# └───────────────────────────────────────────────────────────────────────────┘

import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import time
from scipy.stats import qmc

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Functions                                       │
# └───────────────────────────────────────────────────────────────────────────┘

def load_h5_labels(labels_file_path):
    # Read Labels HDF5 File
    with h5py.File(labels_file_path, 'r') as labels_file:
        labels_train = labels_file['Labels/Train']
        labels_test = labels_file['Labels/Test']

        # Ensure all elements are at least 1D arrays
        labels_train_data = [np.atleast_1d(labels_train[key][()]) for key in labels_train.keys()]
        labels_test_data = [np.atleast_1d(labels_test[key][()]) for key in labels_test.keys()]

        # print("Initial train data shapes:", [data.shape for data in labels_train_data])
        # print("Initial test data shapes:", [data.shape for data in labels_test_data])

        # Convert labels from (20, 15, 1) to (4, 3) by selecting the middle pixel of each 5x5 patch
        def extract_middle_pixels(data):
            # Data should be of shape (20, 15, 1)
            new_data = np.zeros((4, 3), dtype=data.dtype)
            # print(f"\nConverting {data.shape} to {new_data.shape}")

            for i in range(4):
                for j in range(3):
                    row_idx = i * 5 + 2
                    col_idx = j * 5 + 2
                    # print(f"Extracting middle pixel from patch ({i},{j}) at position ({row_idx},{col_idx})")

                    # Display patch and selected pixel for debugging
                    # print(f"Patch values:\n{data[i * 5:i * 5 + 5, j * 5:j * 5 + 5, 0]}")
                    # print(f"Selected middle pixel: {data[row_idx, col_idx, 0]}")

                    # Extract the middle pixel of each 5x5 patch
                    new_data[i, j] = data[row_idx, col_idx, 0]

            # print("Resulting 4x3 grid:\n", new_data)  # Print the transformed grid for debugging
            return new_data

        # Apply the function to transform both train and test data
        labels_train_transformed = np.array([extract_middle_pixels(data) for data in labels_train_data])
        labels_test_transformed = np.array([extract_middle_pixels(data) for data in labels_test_data])

        # Concatenate transformed train and test data along the first axis (samples)
        labels_all = np.concatenate((labels_train_transformed, labels_test_transformed), axis=0)

        # print("\nFinal shape of all labels:", labels_all.shape)
        return labels_all


def sample_possibility_space(n_samples=10000, rows=4, cols=3, options=180):
    """
    Generates samples where each sheet is represented as a 3x4 grid.

    Parameters:
    - n_samples: Number of samples to generate
    - rows: Number of rows in the patch grid (3 in this case)
    - cols: Number of columns in the patch grid (4 in this case)
    - options: Range of possible values for each patch (0 to 179)

    Returns:
    - A 3D numpy array of shape (n_samples, 3, 4) where each 3x4 grid represents a sheet.
    """
    # Each patch in the 3x4 grid can take a value from 0 to 179
    return np.random.randint(0, options, size=(n_samples, rows, cols))


# Step 3: Apply K-means on the sampled possibility space to define clusters (classes)
def find_clusters(possible_space_sample, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(possible_space_sample)
    return kmeans


def sample_possibility_space_lhs(n_samples=10000, rows=4, cols=3, options=180):
    """
    Generates samples using Latin Hypercube Sampling where each sheet is represented as a 3x4 grid.

    Parameters:
    - n_samples: Number of samples to generate
    - rows: Number of rows in the patch grid (4 in this case)
    - cols: Number of columns in the patch grid (3 in this case)
    - options: Range of possible values for each patch (0 to 179)

    Returns:
    - A 3D numpy array of shape (n_samples, 4, 3) where each 4x3 grid represents a sheet.
    """
    # Create an LHS sampler for rows * cols dimensions
    total_patches = rows * cols
    sampler = qmc.LatinHypercube(d=total_patches)

    # Generate the samples and scale them to the range [0, options)
    lhs_sample = sampler.random(n=n_samples)
    lhs_sample_scaled = qmc.scale(lhs_sample, 0, options)

    # Reshape each sample to a (rows, cols) grid
    lhs_sample_reshaped = lhs_sample_scaled.reshape(n_samples, rows, cols)

    return lhs_sample_reshaped.astype(int)




# Step 4: Cluster the actual dataset to see its distribution in these "classes"
def classify_data(data, kmeans):
    # Assign each sample in the dataset to the nearest cluster center
    labels = kmeans.predict(data)
    return labels


# Step 5: Analyze diversity by comparing cluster distributions
def check_diversity(actual_labels, kmeans):
    # Count the number of occurrences in each cluster for the actual data
    unique, counts = np.unique(actual_labels, return_counts=True)
    actual_distribution = dict(zip(unique, counts))

    # Expected distribution if fully balanced
    total_samples = len(actual_labels)
    expected_count = total_samples / kmeans.n_clusters
    expected_distribution = {i: expected_count for i in range(kmeans.n_clusters)}

    # Display actual vs. expected distribution
    print("Cluster\tActual Count\tExpected Count")
    for i in range(kmeans.n_clusters):
        actual_count = actual_distribution.get(i, 0)
        print(f"{i}\t{actual_count}\t\t{expected_count:.2f}")

    # Optionally plot the distribution
    labels, counts = zip(*actual_distribution.items())
    plt.bar(labels, counts, label="Actual")
    plt.hlines(expected_count, xmin=min(labels), xmax=max(labels), color='red', label="Expected")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Cluster Distribution in Dataset")
    plt.show()


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Definitions                                     │
# └───────────────────────────────────────────────────────────────────────────┘

h5_file = r"C:\Gal_Msc\Ipublic-repo\frustrated-composites-dataset\30-35\30-35_MaxMinCurvature_Labels_Reshaped.h5"
n_samples_space = 100000000
n_clusters = 3

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Main Code                                       │
# └───────────────────────────────────────────────────────────────────────────┘


# Load the actual dataset
start_time = time.time()
labels_all = load_h5_labels(h5_file)
end_time = time.time()
print(f"Data shape: {labels_all.shape}")
print(f"Time taken to load dataset: {end_time - start_time:.2f} seconds")

# Sample the full possibility space
start_time = time.time()
# possible_space_sample = sample_possibility_space(n_samples=n_samples_space) #NP Random
possible_space_sample = sample_possibility_space_lhs(n_samples=n_samples_space, rows=4, cols=3, options=180) # Using LHS
print(f"First Sample: {possible_space_sample[0]}")  # Print the first sample to check
end_time = time.time()
print(f"possible_space_sample shape: {possible_space_sample.shape}")
print(f"Time taken to sample possibility space: {end_time - start_time:.2f} seconds")



# Reshape the possible_space_sample and labels_all to be 2D arrays
start_time = time.time()
possible_space_sample_flat = possible_space_sample.reshape(possible_space_sample.shape[0], -1)
labels_all_flat = labels_all.reshape(labels_all.shape[0], -1)
end_time = time.time()
print("Flattened possible_space_sample shape:", possible_space_sample_flat.shape)
print("Flattened labels_all shape:", labels_all_flat.shape)
print(f"Time taken to reshape data: {end_time - start_time:.2f} seconds")

# Find clusters in the full space sample
start_time = time.time()
kmeans = find_clusters(possible_space_sample_flat, n_clusters=n_clusters)
end_time = time.time()
print(f"Time taken to find clusters: {end_time - start_time:.2f} seconds")

# Classify actual data based on the clusters in the full space
start_time = time.time()
actual_labels = classify_data(labels_all_flat, kmeans)
end_time = time.time()
print(f"Time taken to classify data: {end_time - start_time:.2f} seconds")

# Check diversity and balancing
start_time = time.time()
check_diversity(actual_labels, kmeans)
end_time = time.time()
print(f"Time taken to check diversity and balancing: {end_time - start_time:.2f} seconds")
