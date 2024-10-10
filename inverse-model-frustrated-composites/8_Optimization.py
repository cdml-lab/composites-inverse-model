import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from numpy.ma.extras import average
from findiff import FinDiff
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

# ==========================================================
# Initial Definitions
# ==========================================================
model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\17-24_Location_Features_20241001.pkl"
labels_channels = 3
features_channels = 1
add_curvature_channels = 0

# File Paths
excel_file_path = r'C:\Users\user\OneDrive - Technion\Documents\GitHub\public-repo\inverse-model-frustrated-composites\saved_models_for_checks\Dataset_Output_Test1.xlsx'
excel_file_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\Dataset_Output_Test1.xlsx"
hdf5_file_path = r'C:\Users\user\OneDrive - Technion\Documents\GitHub\public-repo\inverse-model-frustrated-composites\saved_models_for_checks\test\test1.h5'
reshaped_hdf5_file_path = r"C:\Users\User\OneDrive - Technion\Documents\GitHub\public-repo\inverse-model-frustrated-composites\saved_models_for_checks\test\test1_reshaped.h5"

# Columns to preserve and split
preserve_columns_features = ['Location X', 'Location Y', 'Location Z'
                             # 'MaCD-X'
                             #, 'MaCD-Y', 'MaCD-Z', 'Max Curvature Length', 'Min Curvature Length'
                             ]
split_columns_features = {
    # 'Max Curvature Direction': ['MaCD-X', 'MaCD-Y', 'MaCD-Z'],

}
remove_split_columns = []
new_shape = (20, 15, labels_channels)  # Adjust based on your requirements

# Control flags
is_convert = False
is_clean = False
use_local_frame = True  # Set this flag to choose between local or global curvature calculation
is_optimize = True # Turn the optimization on and off


# ==========================================================
# Functions and Classes
# ==========================================================

# Load Excel as DataFrame
def load_excel_as_dataframe(file_path):
    return pd.read_excel(file_path)

def visualize_xyz_and_derivatives(points_xyz, tangent_u, tangent_v, normal_vector, step, plot_name):
    """
    Visualize the XYZ points, tangent vectors (derivatives), and normal vectors.
    :param points_xyz: The grid of XYZ points (rows, cols, 3)
    :param tangent_u: The tangent vectors in the u direction
    :param tangent_v: The tangent vectors in the v direction
    :param normal_vector: The normal vectors at each point
    :param step: The current step number (for logging)
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface points
    ax.scatter(points_xyz[:, :, 0], points_xyz[:, :, 1], points_xyz[:, :, 2], color='b', label='XYZ Points')

    # Calculate the magnitude of the tangents and normals
    tangent_u_magnitude = np.linalg.norm(tangent_u, axis=-1)
    tangent_v_magnitude = np.linalg.norm(tangent_v, axis=-1)
    normal_magnitude = np.linalg.norm(normal_vector, axis=-1)

    print(f"Tangent U magnitudes: {average(tangent_u_magnitude)}")
    print(f"Tangent V magnitudes: {average(tangent_v_magnitude)}")
    print(f"Normal magnitudes: {average(normal_magnitude)}")

    # Scaling factor to increase the size of the vectors for visualization
    scale_factor = 0.5  # Increase this value for a more visible visualization


    # Plot tangent and normal vectors as arrows
    for i in range(tangent_u.shape[0]):  # Loop over the rows (18)
        for j in range(tangent_u.shape[1]):  # Loop over the columns (13)
            # Tangent U direction
            ax.quiver(
                points_xyz[i+1, j+1, 0], points_xyz[i+1, j+1, 1], points_xyz[i+1, j+1, 2],  # Starting point (adjusted for indexing)
                tangent_u[i, j, 0], tangent_u[i, j, 1], tangent_u[i, j, 2],  # U-direction components
                color='r', length=float(tangent_u_magnitude[i, j]) * scale_factor, normalize=True, label='Tangent U' if i == 0 and j == 0 else ""
            )
            # Tangent V direction
            ax.quiver(
                points_xyz[i+1, j+1, 0], points_xyz[i+1, j+1, 1], points_xyz[i+1, j+1, 2],  # Starting point (adjusted for indexing)
                tangent_v[i, j, 0], tangent_v[i, j, 1], tangent_v[i, j, 2],  # V-direction components
                color='g', length=float(tangent_v_magnitude[i, j]) * scale_factor, normalize=True, label='Tangent V' if i == 0 and j == 0 else ""
            )
            # Normal vector direction
            ax.quiver(
                points_xyz[i+1, j+1, 0], points_xyz[i+1, j+1, 1], points_xyz[i+1, j+1, 2],  # Starting point (adjusted for indexing)
                normal_vector[i, j, 0], normal_vector[i, j, 1], normal_vector[i, j, 2],  # Normal vector components
                color='y', length=0.1 * scale_factor, normalize=True, label='Normal' if i == 0 and j == 0 else ""
            )

    ax.legend()
    plt.title(f'XYZ Points and Derivatives (Tangents & Normals) at Step {step}')

    # Log the figure to wandb
    wandb.log({f"{plot_name} Step {step}": wandb.Image(fig)})

    # Show and close the plot
    plt.show()
    plt.close()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_xyz_with_permutations(points_xyz, tangent_u, tangent_v, normal_vector, step, plot_name, axis_orders=None):
    """
    Visualize the XYZ points, tangent vectors, and normal vectors for all permutations of axes.

    :param points_xyz: The grid of XYZ points (rows, cols, 3)
    :param tangent_u: The tangent vectors in the u direction
    :param tangent_v: The tangent vectors in the v direction
    :param normal_vector: The normal vectors at each point
    :param step: The current step number (for logging)
    :param plot_name: Base name for the plot
    :param axis_orders: List of tuples, each representing a permutation of (x, y, z) axes
    """
    if axis_orders is None:
        axis_orders = [(0, 1, 2), (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 0, 1), (2, 1, 0)]

    fig = plt.figure(figsize=(15, 10))

    for idx, axis_order in enumerate(axis_orders):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')

        # Permute points and vectors based on axis_order
        permuted_xyz = points_xyz[:, :, axis_order]
        permuted_tangent_u = tangent_u[:, :, axis_order]
        permuted_tangent_v = tangent_v[:, :, axis_order]
        permuted_normal = normal_vector[:, :, axis_order]

        # Plot the surface points
        ax.scatter(permuted_xyz[:, :, 0], permuted_xyz[:, :, 1], permuted_xyz[:, :, 2], color='b', label='XYZ Points')

        # Calculate the magnitudes of the vectors
        tangent_u_magnitude = np.linalg.norm(permuted_tangent_u, axis=-1)
        tangent_v_magnitude = np.linalg.norm(permuted_tangent_v, axis=-1)
        normal_magnitude = np.linalg.norm(permuted_normal, axis=-1)

        # Scaling factor for visualization
        scale_factor = 0.5  # You can adjust this for better visualization

        # Plot tangent and normal vectors as arrows
        for i in range(permuted_tangent_u.shape[0]):
            for j in range(permuted_tangent_u.shape[1]):
                ax.quiver(
                    permuted_xyz[i, j, 0], permuted_xyz[i, j, 1], permuted_xyz[i, j, 2],
                    permuted_tangent_u[i, j, 0], permuted_tangent_u[i, j, 1], permuted_tangent_u[i, j, 2],
                    color='r', length=(tangent_u_magnitude[i, j] * scale_factor), normalize=True
                )
                ax.quiver(
                    permuted_xyz[i, j, 0], permuted_xyz[i, j, 1], permuted_xyz[i, j, 2],
                    permuted_tangent_v[i, j, 0], permuted_tangent_v[i, j, 1], permuted_tangent_v[i, j, 2],
                    color='g', length=(tangent_v_magnitude[i, j] * scale_factor), normalize=True
                )
                ax.quiver(
                    permuted_xyz[i, j, 0], permuted_xyz[i, j, 1], permuted_xyz[i, j, 2],
                    permuted_normal[i, j, 0], permuted_normal[i, j, 1], permuted_normal[i, j, 2],
                    color='y', length=(normal_magnitude[i, j] * scale_factor), normalize=True
                )

        ax.set_title(f'Axis Order: {axis_order}')
        ax.legend()

    plt.suptitle(f'XYZ Points and Derivatives at Step {step} for All Axis Permutations')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the figure and log to wandb
    save_path = f"{plot_name}_step_{step}.png"
    plt.savefig(save_path)
    wandb.log({f"{plot_name}": wandb.Image(save_path)})
    plt.close()

    print(f"Visualization saved: {save_path}")


def calculate_tangent_normal_xyz(points_xyz):
        """
        Calculate tangent vectors and normal vectors from XYZ coordinates.
        :param points_xyz: A 3D tensor of shape (rows, cols, 3) containing XYZ coordinates
        :return: Tangent vectors (T_u, T_v) and normal vectors (N) in XYZ space
        """
        try:
            # print("Inside calculate_tangent_normal_xyz")
            # print(f"points_xyz shape: {points_xyz.shape}")

            assert points_xyz.shape[
                       -1] == 3, "Last dimension of points_xyz must have 3 components (for XYZ coordinates)."

            # Calculate tangent vectors using finite differences (adapted for 3D tensor)
            T_u = points_xyz[2:, 1:-1, :] - points_xyz[:-2, 1:-1, :]  # Tangent in the row (u) direction
            T_v = points_xyz[1:-1, 2:, :] - points_xyz[1:-1, :-2, :]  # Tangent in the column (v) direction

            # print(f"T_u shape: {T_u.shape}, T_v shape: {T_v.shape}")

            # Cross product to get the normal vector at each point
            N = torch.cross(T_u, T_v, dim=-1)
            N_unit = N / torch.norm(N, dim=-1, keepdim=True)

            # print(f"N shape: {N.shape}, N_unit shape: {N_unit.shape}")
            return T_u, T_v, N_unit
        except Exception as e:
            print(f"Error in calculate_tangent_normal_xyz: {e}")
            raise


def aggregate_patch_gradients(initial_fiber_orientation_patches, initial_fiber_orientation):
    """
    Aggregate the gradients of the full fiber orientation grid (20x15) into the gradients of the 12 patches (3x4).
    """
    # Reshape gradients for the full grid (20x15)
    gradients_full = initial_fiber_orientation.grad.view(4, 20, 15)

    # Initialize a tensor to store the gradients for the patches
    patch_gradients = torch.zeros_like(initial_fiber_orientation_patches.grad)

    # Aggregate gradients over each 5x5 patch
    for i in range(3):  # 3 patches in the row direction
        for j in range(4):  # 4 patches in the column direction
            patch_gradients[:, i, j] = gradients_full[:, i * 5:(i + 1) * 5, j * 5:(j + 1) * 5].mean(
                dim=(1, 2))  # Mean over the 5x5 patch

    # Assign the aggregated gradients to the patch tensor
    initial_fiber_orientation_patches.grad = patch_gradients


# Convert DataFrame to HDF5
def convert_dataframe_to_hdf5(df, output_hdf5_path):
    with h5py.File(output_hdf5_path, 'w') as hdf:
        hdf.create_dataset('data', data=df.values)


# Function to split vector columns and convert Excel data to HDF5
def excel_to_hdf5(excel_file_path, hdf5_file_path, category, split_columns):
    print(f"Starting conversion from Excel to HDF5 for file: {excel_file_path}")
    xlsx = pd.ExcelFile(excel_file_path)
    worksheets = xlsx.sheet_names
    print(f"Worksheets found: {worksheets}")

    with h5py.File(hdf5_file_path, 'w') as h5file:
        main_group = h5file.create_group(category)
        print(f"Created HDF5 group: {category}")

        for sheet in worksheets:
            print(f"Processing sheet: {sheet}")
            df = pd.read_excel(xlsx, sheet_name=sheet)
            print(f"Loaded sheet with shape: {df.shape}")

            # Split specified columns
            for col, new_cols in split_columns.items():
                if col in df.columns:
                    print(f"Splitting column '{col}' into {new_cols}")
                    df[col] = df[col].astype(str).str.replace('{', '', regex=True).str.replace('}', '', regex=True)
                    df[new_cols] = df[col].str.split(',', expand=True)
                    df = df.drop(col, axis=1)
                    print(f"After splitting '{col}', DataFrame now has columns: {df.columns.tolist()}")

            # Convert all columns to numeric (and drop any non-numeric columns)
            df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
            print(f"Numeric columns found: {df_numeric.columns.tolist()}")

            if df_numeric.empty:
                print(f"Warning: No numeric data found in sheet '{sheet}', skipping.")
                continue

            # Save to HDF5
            dataset = main_group.create_dataset(sheet, data=df_numeric.to_numpy())
            dataset.attrs['columns'] = df_numeric.columns.astype(str).to_list()
            print(f"Saved dataset for sheet: {sheet} with shape: {df_numeric.shape}")

    print(f"Finished converting Excel to HDF5. HDF5 saved at: {hdf5_file_path}")


# Function to clean and reshape the HDF5 data
def clean_and_reshape_hdf5(category, preserve_columns, split_columns, remove_split_columns, new_shape, hdf5_file_path,
                           reshaped_hdf5_file_path):
    print(f"Starting cleaning and reshaping of HDF5 data from file: {hdf5_file_path}")

    with h5py.File(hdf5_file_path, 'r') as h5file, h5py.File(reshaped_hdf5_file_path, 'w') as new_h5file:
        main_group = h5file[category]
        new_main_group = new_h5file.create_group(category)
        print(f"Processing HDF5 group: {category}")

        for sheet in main_group:
            print(f"Processing dataset: {sheet}")
            dataset = main_group[sheet]

            if dataset.size == 0:
                print(f"Warning: Dataset '{sheet}' is empty, skipping.")
                continue

            # Get columns and create DataFrame
            columns_metadata = dataset.attrs['columns']
            df = pd.DataFrame(dataset[:], columns=columns_metadata)
            print(f"Loaded dataset into DataFrame with shape: {df.shape}")

            # Split specified columns
            for col, new_cols in split_columns.items():
                if col in df.columns:
                    print(f"Splitting column '{col}' into {new_cols}")
                    df[col] = df[col].astype(str).str.replace('{', '', regex=True).str.replace('}', '', regex=True)
                    df[new_cols] = df[col].str.split(',', expand=True)
                    df = df.drop(col, axis=1)

            # Keep only needed columns
            keep_columns = [col for col in preserve_columns if col in df.columns]
            print(f"Columns to keep: {keep_columns}")
            df = df[keep_columns]

            # Convert all values to numeric
            df = df.apply(pd.to_numeric, errors='coerce')

            # Reshape the data
            reshaped_data = df.to_numpy().reshape(new_shape, order='F')
            print(f"Reshaped data to: {reshaped_data.shape}")

            # Save reshaped data to HDF5
            new_sub_group = new_main_group.create_group(sheet)
            new_dataset = new_sub_group.create_dataset(sheet, data=reshaped_data)
            new_dataset.attrs['columns'] = keep_columns
            print(f"Saved reshaped dataset for '{sheet}' with new shape: {reshaped_data.shape}")

    print(f"Finished cleaning and reshaping HDF5 data. New HDF5 saved at: {reshaped_hdf5_file_path}")


# Function to visualize the fiber angles as a heatmap
def visualize_fiber_orientation_heatmap(fiber_orientation, step, plot_name):
    """
    Visualize fiber orientation as a 2D heatmap.
    :param fiber_orientation: Numpy array of shape (4, 20, 15) representing the angles.
    :param step: Current optimization step.
    :param plot_name: The name for the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # No need to detach or use `.cpu()` since fiber_orientation is already a numpy array.
    # If it's in tensor format, you would convert it. Here it's a numpy array already.
    fiber_orientation_np = fiber_orientation.squeeze(0)  # (4, 20, 15)

    # For simplicity, we can visualize the first channel (you can extend to others if needed)
    angle_channel = fiber_orientation_np[0, :, :]  # Shape (20, 15)

    # Create a heatmap from the angle channel
    sns.heatmap(angle_channel, cmap="coolwarm", annot=False, cbar=True, ax=ax)

    # Title and logging
    plt.title(f'Fiber Orientation Heatmap at Step {step} - {plot_name}')

    # Log the plot to wandb
    wandb.log({f"Fiber Orientation Heatmap {plot_name} Step {step}": wandb.Image(fig)})

    # Show the plot
    # plt.show()
    plt.close()


# Define the same model architecture as the one used for training
class OurModel(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(OurModel, self).__init__()

        self.conv_1 = torch.nn.Conv2d(in_channels=features_channels + add_curvature_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_8 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_9 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_10 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_11 = torch.nn.Conv2d(in_channels=512, out_channels=labels_channels, kernel_size=3, padding=1)

        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=32)
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=64)
        self.batch_norm_3 = torch.nn.BatchNorm2d(num_features=64)
        self.batch_norm_4 = torch.nn.BatchNorm2d(num_features=128)
        self.batch_norm_5 = torch.nn.BatchNorm2d(num_features=128)
        self.batch_norm_6 = torch.nn.BatchNorm2d(num_features=256)
        self.batch_norm_7 = torch.nn.BatchNorm2d(num_features=256)
        self.batch_norm_8 = torch.nn.BatchNorm2d(num_features=512)
        self.batch_norm_9 = torch.nn.BatchNorm2d(num_features=512)
        self.batch_norm_10 = torch.nn.BatchNorm2d(num_features=512)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)

        x = self.dropout(x)  # Dropout after every 3 layers

        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.relu(x)

        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.relu(x)

        x = self.dropout(x)  # Dropout

        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.relu(x)

        x = self.conv_7(x)
        x = self.batch_norm_7(x)
        x = self.relu(x)

        x = self.conv_8(x)
        x = self.batch_norm_8(x)
        x = self.relu(x)

        x = self.dropout(x)  # Dropout

        x = self.conv_9(x)
        x = self.batch_norm_9(x)
        x = self.relu(x)

        x = self.conv_10(x)

        x = self.batch_norm_10(x)
        x = self.relu(x)

        x = self.conv_11(x)
        # Don't apply ReLU if this is a regression problem, so no activation on the final layer

        # Constrain output values to the label range (0, 1)
        x = torch.sigmoid(x)
        return x

# ==========================================================
# Loss Functions
# ==========================================================
def dot_product_loss(predicted, target, epsilon=1e-8):
    # Normalize the vectors
    predicted_norm = torch.norm(predicted, dim=-1, keepdim=True)
    target_norm = torch.norm(target, dim=-1, keepdim=True)

    # Clamp the norms to avoid division by zero or very small values
    predicted_norm = torch.clamp(predicted_norm, min=epsilon)
    target_norm = torch.clamp(target_norm, min=epsilon)

    predicted_normalized = predicted / predicted_norm
    target_normalized = target / target_norm

    # Compute dot product
    dot_product = torch.sum(predicted_normalized * target_normalized, dim=-1)

    # Return the negative dot product (to minimize in optimization)
    return -torch.mean(dot_product)


import torch.nn.functional as F
def cosine_similarity_loss(predicted, target):
    """
    Cosine similarity loss between predicted and target tangent vectors.
    Values close to 0 indicate high similarity.
    """
    # Normalize the vectors to get unit vectors (direction only)
    predicted_normalized = F.normalize(predicted, p=2, dim=-1)
    target_normalized = F.normalize(target, p=2, dim=-1)

    # Compute cosine similarity
    cosine_similarity = F.cosine_similarity(predicted_normalized, target_normalized, dim=-1)

    # Loss is 1 - cosine similarity, so a loss of 0 means perfect alignment
    loss = 1 - cosine_similarity.mean()
    return loss


def geodesic_loss(predicted, target, eps=1e-6):
    """
    Geodesic loss between two sets of vectors, normalized between 0 and 1.
    :param predicted: Predicted tensor of shape (batch_size, rows, cols, 3)
    :param target: Target tensor of shape (batch_size, rows, cols, 3)
    :param eps: Small value to avoid division by zero or undefined arccos
    :return: Normalized geodesic loss (between 0 and 1)
    """
    # Normalize both predicted and target vectors
    predicted_normalized = F.normalize(predicted, dim=-1, p=2)
    target_normalized = F.normalize(target, dim=-1, p=2)

    # Compute dot product between predicted and target
    dot_product = torch.sum(predicted_normalized * target_normalized, dim=-1)

    # Clamp the dot product to avoid values slightly outside [-1, 1] due to numerical precision
    dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)

    # Compute the geodesic distance using arccos
    geodesic_distance = torch.acos(dot_product)

    # Normalize the geodesic distance to [0, 1] by dividing by π
    normalized_geodesic_distance = geodesic_distance / torch.pi

    # Return the mean normalized geodesic distance as the loss
    return torch.mean(normalized_geodesic_distance)

def normalize_data(data, min_val=0, max_val=1):
    """
    Normalizes the input data to the range [min_val, max_val].

    :param data: Tensor containing the input data.
    :param min_val: Minimum value of the target normalization range.
    :param max_val: Maximum value of the target normalization range.
    :return: Normalized tensor.
    """
    data_min = data.min()
    data_max = data.max()

    # Perform Min-Max normalization
    normalized_data = (data - data_min) / (data_max - data_min)
    normalized_data = normalized_data * (max_val - min_val) + min_val
    return normalized_data


# ==========================================================
# Main Code - Load Model and Data, Perform Optimization
# ==========================================================

# Initialize wandb for experiment tracking
wandb.init(project="optimization")

# Track hyperparameters and initial settings
wandb.config.update({
    "learning_rate": 100000,
    "max_iterations": 10000,
    "loss": "Cosine", # Options: "MSE", "L1", "Geodesic", "Cosine", "DotProduct"
    "curvature_calculation": "tangents_u_v",
    "derivative": "first_order"
})

# Load the trained model
model = OurModel()
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device('cuda'))
    device = 'cuda'
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    device = 'cpu'

model.eval()

if is_convert:
    # Convert Excel to HDF5, with splitting of vector columns
    excel_to_hdf5(excel_file_path, hdf5_file_path, 'Features', split_columns_features)
else:
    print("Not Converting to HDF5")

if is_clean:
    # Clean columns and reshape
    clean_and_reshape_hdf5('Features', preserve_columns_features, split_columns_features, [], new_shape, hdf5_file_path,
                           reshaped_hdf5_file_path)
else:
    print("Not Cleaning File")

# Read dataset from the file
with h5py.File(reshaped_hdf5_file_path, 'r') as hdf:
    dataset_name = [None]


    def find_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset found: {name}")
            dataset_name[0] = name
            return


    hdf.visititems(find_dataset)

    if dataset_name[0]:
        reshaped_data = hdf[dataset_name[0]][:]
        print(f"Loaded reshaped data from dataset '{dataset_name[0]}' with shape: {reshaped_data.shape}")
    else:
        print("No dataset found in the HDF5 file.")

# ==========================================================
# Optimization Process - Matching Predicted Output to Excel Data
# ==========================================================

# Initialize the fiber orientation with 12 distinct orientations for the 3x4 patches
random_orientations = torch.randint(0, 181, (3, 4), dtype=torch.float32)

# Create the (20, 15) grid by repeating the 12 values across the 5x5 patches
initial_fiber_orientation = torch.zeros((1, 1, 20, 15), dtype=torch.float32)

# Loop over 3x4 patches and fill the (20x15) grid
for i in range(3):
    for j in range(4):
        # Assign the random orientation to the corresponding 5x5 patch
        initial_fiber_orientation[:, 0, i*5:(i+1)*5, j*5:(j+1)*5] = random_orientations[i, j]

initial_fiber_orientation = normalize_data(initial_fiber_orientation)



initial_fiber_orientation = initial_fiber_orientation.to(device).requires_grad_(True)


# Log the initial fiber orientation (removing the batch dimension for logging, but keeping it for the model)
fiber_orientation_step_0 = initial_fiber_orientation.detach().cpu().numpy()  # Detach for logging
fiber_orientation_df_step_0 = pd.DataFrame(fiber_orientation_step_0.squeeze(0).reshape(-1, fiber_orientation_step_0.shape[-1]))
wandb.log({"Step 0 Fiber Orientation": wandb.Table(dataframe=fiber_orientation_df_step_0)})

# Set up the optimizer (SGD)
optimizer = optim.SGD([initial_fiber_orientation], lr=wandb.config.learning_rate)

# Initialize the ReduceLROnPlateau scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2500, verbose=True)

# Define the loss function
if wandb.config.loss == "MSE":
    loss_fn = nn.MSELoss()
elif wandb.config.loss == "L1":
    loss_fn = nn.L1Loss()
elif wandb.config.loss == "Geodesic":
    loss_fn = geodesic_loss
elif wandb.config.loss == "Cosine":
    loss_fn = cosine_similarity_loss
elif wandb.config.loss == "DotProduct":
    loss_fn = dot_product_loss
else:
    raise ValueError(f"Unknown loss function: {wandb.config.loss}")

print(f"Loss: {loss_fn}")

# Calculate target surface tangents and normals
print("----Target Surface----")
target_u, target_v, target_normal_vectors = calculate_tangent_normal_xyz(torch.tensor(reshaped_data, dtype=torch.float32))

# Visualize the target surface
visualize_xyz_and_derivatives(reshaped_data, target_u.numpy(), target_v.numpy(), target_normal_vectors.numpy(), "target", plot_name="Target Surface")

# Make sure the target normal vectors are on the same device as the predicted ones
target_normal_vectors = torch.tensor(target_normal_vectors, dtype=torch.float32).to(device)

# Make sure the target normal vectors are on the correct device, and clone & detach them
target_normal_vectors = target_normal_vectors.clone().detach().to(device)

# Set Optimization Parameters
max_iterations = wandb.config.max_iterations
desired_threshold = 0.01

patience = 20000  # Number of steps with no improvement after which to stop
best_loss = float('inf')
no_improvement_steps = 0

if is_optimize == True:
    # Optimization loop
    for step in range(max_iterations):
        optimizer.zero_grad()

        print("----Starting Optimization----")

        # Get the predicted XYZ from the model
        predicted_xyz = model(initial_fiber_orientation.to(device))

        # Adjust the shape for calculation
        predicted_xyz = predicted_xyz.permute(0, 2, 3, 1).squeeze(0)  # Adjusted shape (1, 3, 20, 15) to (20, 15, 3)

        # print(f"Predicted XYZ shape: {predicted_xyz.shape}")  # Check the shape here

        # Calculate the tangents and normal vectors for the predicted surface (no detachment here)
        tangent_u, tangent_v, normal_vector = calculate_tangent_normal_xyz(predicted_xyz)

        # Ensure tangents are on the same device
        tangent_u = tangent_u.to(device)
        tangent_v = tangent_v.to(device)
        target_u = target_u.to(device)
        target_v = target_v.to(device)

        # Log results for visualization in wandb (use detach just for logging/visualization)
        if step == 0:
            visualize_xyz_and_derivatives(
                predicted_xyz.detach().cpu().numpy(),  # Detach for visualization
                tangent_u.detach().cpu().numpy(),      # Detach for visualization
                tangent_v.detach().cpu().numpy(),      # Detach for visualization
                normal_vector.detach().cpu().numpy(),  # Detach for visualization
                step,
                plot_name="Step 0"
            )

            # Assuming predicted_xyz, tangent_u, tangent_v, normal_vector are already computed
            visualize_xyz_with_permutations(
                predicted_xyz.detach().cpu().numpy(),
                tangent_u.detach().cpu().numpy(),
                tangent_v.detach().cpu().numpy(),
                normal_vector.detach().cpu().numpy(),
                step,
                plot_name="Predicted_XYZ_Permutations"
            )

        # # Ensure both tangent_u and target_u are consistently shaped
        # if tangent_u.shape != target_u.shape:
        #     tangent_u = tangent_u.permute(1, 0, 2)  # Transpose the dimensions
        #
        # # Ensure both tangent_v and target_v are consistently shaped
        # if tangent_v.shape != target_v.shape:
        #     tangent_v = tangent_v.permute(1, 0, 2)  # Transpose the dimensions

        # Compute the loss by comparing predicted normal vectors to the target normals (stay in graph)
        loss_u = loss_fn(tangent_u, target_u)  # Compare tangents in u direction
        loss_v = loss_fn(tangent_v, target_v)  # Compare tangents in v direction

        # Combine the two losses
        loss = (loss_u + loss_v) / 2

        # Backpropagation
        loss.backward()

        # Calculate gradient norms (L2 norm)
        grad_norm = initial_fiber_orientation.grad.norm(2).item()

        # Log the gradient norm to wandb
        wandb.log({"loss": loss.item(), "step": step, "gradient_norm": grad_norm})

        optimizer.step()

        # Step the scheduler based on the loss
        scheduler.step(loss)

        # Log the learning rate
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr'], "step": step})

        print(f'Step {step + 1}, Loss: {loss.item()}, Gradient Norm: {grad_norm}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        # Clamp the values of the fiber orientation between 0 and 180
        # **Replace in-place clamp operation with out-of-place version**
        initial_fiber_orientation.data = torch.clamp(initial_fiber_orientation.data, 0, 1)


        if loss.item() < desired_threshold:
            print('Desired threshold reached. Stopping optimization.')
            break

        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improvement_steps = 0
        else:
            no_improvement_steps += 1

        if no_improvement_steps >= patience:
            print(f"No improvement for {patience} steps. Stopping early.")
            break

        if loss.item() < desired_threshold:
            print('Desired threshold reached. Stopping optimization.')
            break


# Log the final fiber orientation
fiber_orientation_step_final = initial_fiber_orientation.detach().cpu().numpy()  # Detach for logging
fiber_orientation_df_step_final = pd.DataFrame(fiber_orientation_step_final.squeeze(0).reshape(-1, fiber_orientation_step_final.shape[-1]))
wandb.log({"Final Step Fiber Orientation": wandb.Table(dataframe=fiber_orientation_df_step_final)})


visualize_xyz_and_derivatives(
    predicted_xyz.detach().cpu().numpy(),  # Detach for visualization
    tangent_u.detach().cpu().numpy(),  # Detach for visualization
    tangent_v.detach().cpu().numpy(),  # Detach for visualization
    normal_vector.detach().cpu().numpy(),  # Detach for visualization
    step,
    plot_name="Final Surface"
)

# print("Optimization complete. Result saved to Excel.")
visualize_fiber_orientation_heatmap(fiber_orientation_step_final, step=step, plot_name="final")
# Finish the wandb run
wandb.finish()
