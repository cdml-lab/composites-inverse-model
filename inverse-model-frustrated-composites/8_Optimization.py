import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from numpy.ma.extras import average

# ==========================================================
# Initial Definitions
# ==========================================================
model_path = r'C:\Users\user\OneDrive - Technion\Documents\GitHub\public-repo\inverse-model-frustrated-composites\saved_models_for_checks\17-24_Curvature_Features_20240923.pkl'
labels_channels = 3

# File Paths
excel_file_path = r'C:\Users\user\OneDrive - Technion\Documents\GitHub\public-repo\inverse-model-frustrated-composites\saved_models_for_checks\Dataset_Output_Test1.xlsx'
hdf5_file_path = r'C:\Users\user\OneDrive - Technion\Documents\GitHub\public-repo\inverse-model-frustrated-composites\saved_models_for_checks\test\test1.h5'
reshaped_hdf5_file_path = r'C:\Users\user\OneDrive - Technion\Documents\GitHub\public-repo\inverse-model-frustrated-composites\saved_models_for_checks\test\test1_reshaped.h5'

# Columns to preserve and split
preserve_columns_features = ['Location X', 'Location Y', 'Location Z'
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

# ==========================================================
# Functions and Classes
# ==========================================================

# Load Excel as DataFrame
def load_excel_as_dataframe(file_path):
    return pd.read_excel(file_path)

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
def clean_and_reshape_hdf5(category, preserve_columns, split_columns, remove_split_columns, new_shape, hdf5_file_path, reshaped_hdf5_file_path):
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

# Define the same model architecture as the one used for training
class OurModel(torch.nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
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
        self.dropout = torch.nn.Dropout(p=0.3)

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
        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout after every 3 layers
        x = self.conv_7(x)
        x = self.batch_norm_7(x)
        x = self.relu(x)
        x = self.conv_8(x)
        x = self.batch_norm_8(x)
        x = self.relu(x)
        x = self.conv_9(x)
        x = self.batch_norm_9(x)
        x = self.relu(x)
        x = self.conv_10(x)
        x = self.batch_norm_10(x)
        x = self.relu(x)
        x = self.conv_11(x)
        return x  # No activation on the final layer for regression

# ==========================================================
# Main Code - Load Model and Data, Perform Optimization
# ==========================================================

# Load the trained model
model = OurModel()
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device('cuda'))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

if is_convert:
    # Convert Excel to HDF5, with splitting of vector columns
    excel_to_hdf5(excel_file_path, hdf5_file_path, 'Features', split_columns_features)
else:
    print("Not Converting to HDF5")

if is_clean:
    # Clean columns and reshape
    clean_and_reshape_hdf5('Features', preserve_columns_features, split_columns_features, [], new_shape, hdf5_file_path, reshaped_hdf5_file_path)
else:
    print("Not Cleaning File")

# Read dataset from the file
with h5py.File(reshaped_hdf5_file_path, 'r') as hdf:
    dataset_name = [None]

    # Find the first dataset
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

# Convert reshaped data to a tensor and permute to match the model's output shape
input_tensor = torch.tensor(reshaped_data, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)  # Shape [1, 5, 20, 15]

# Initialize the fiber orientation with random integers between 0 and 180
initial_fiber_orientation = torch.randint(0, 181, (1, 1, 20, 15), dtype=torch.float32, requires_grad=True)


# Set up the optimizer (SGD)
optimizer = optim.SGD([initial_fiber_orientation], lr=0.1)

# Define the loss function
loss_fn = nn.MSELoss()

# Optimization loop
max_iterations = 1000
desired_threshold = 0.01

for step in range(max_iterations):
    optimizer.zero_grad()
    predicted_images = model(initial_fiber_orientation)
    loss = loss_fn(predicted_images, input_tensor)
    loss.backward()

    # Multiplying gradients for faster convergence - re-examine with working model.
    initial_fiber_orientation.grad *= 10000  # Scale gradients by 1000

    # Print gradients
    # print(average(initial_fiber_orientation.grad))

    optimizer.step()

    # Clamp the input values to the range [0, 180]
    # with torch.no_grad():
    #     initial_fiber_orientation.clamp_(0, 180)

    print(f'Step {step + 1}, Loss: {loss.item()}')

    if loss.item() < desired_threshold:
        print('Desired threshold reached. Stopping optimization.')
        break

# Save the optimized fiber orientation to Excel
# Convert the optimized fiber orientation tensor to a 2D DataFrame and save to Excel
optimized_fiber_orientation_df = pd.DataFrame(initial_fiber_orientation.detach().numpy().squeeze(0).squeeze(0))
optimized_fiber_orientation_df.to_excel('optimized_fiber_orientation.xlsx', index=False)

print("Optimization complete. Result saved to Excel.")

