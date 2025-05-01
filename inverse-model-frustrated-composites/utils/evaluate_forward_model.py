# ┌───────────────────────────────────────────────────────────────────────────┐
# │                                Imports                                    │
# └───────────────────────────────────────────────────────────────────────────┘

import torch
import sys
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                           Definitions                                     │
# └───────────────────────────────────────────────────────────────────────────┘
# Get the script's directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent


model_path = project_root / "saved_models" / "iconic-microwave.pkl"
plot_dir = project_root / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

# Dataset directory (datasets are parallel to the code folder)
dataset_dir = Path(__file__).resolve().parents[2] / "frustrated-composites-dataset"

# Set dataset name
dataset_name="60-83_no-smooth_no-69_xyz"

# PAY ATTENTION: since this is a forward models the files are flipped and the labels file will be the original features
# file! and the same foe feature will be the original labels file, meant for in inverse model.
# Defines the training files
labels_file = f"{dataset_dir}/{dataset_name}/{dataset_name}_Merged_Features.h5"
features_file = f"{dataset_dir}/{dataset_name}/{dataset_name}_Merged_Labels.h5"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features_channels = 1
labels_channels = 3

# XYZ
global_label_max = [20.4, 20.4, 20.0]
global_label_min = [-20.4, -20.4, -0.2]

global_feature_max = [180.0]
global_feature_min = [0.0]



# ┌───────────────────────────────────────────────────────────────────────────┐
# │                   Adjust Path to Import Training Components               │
# └───────────────────────────────────────────────────────────────────────────┘

# Append the project root to sys.path to import from sibling directory

sys.path.append(str(project_root))

from forward_model import (
    OurVgg16InstanceNorm2d,
    FolderHDF5Data,
    MaskedLossWrapper,
    PointDistanceLoss
)



# ┌───────────────────────────────────────────────────────────────────────────┐
# │                             Data Loading                                  │
# └───────────────────────────────────────────────────────────────────────────┘

val_dataset = FolderHDF5Data(
    features_file=features_file,
    labels_file=labels_file,
    feature_main_group='Labels',
    label_main_group='Features',
    category='Test',
    global_feature_min=global_feature_min,
    global_feature_max=global_feature_max,
    global_label_min=global_label_min,
    global_label_max=global_label_max
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                            Model Loading                                  │
# └───────────────────────────────────────────────────────────────────────────┘

model = OurVgg16InstanceNorm2d().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                         Evaluation + Plotting                             │
# └───────────────────────────────────────────────────────────────────────────┘
# ┌───────────────────────────────────────────────────────────────────────────┐
# │                     Custom Evaluation + Scatter Plots                     │
# └───────────────────────────────────────────────────────────────────────────┘

def denormalize_tensor(tensor, mins, maxs):
    denorm = []
    for c in range(tensor.shape[0]):
        ch = tensor[c]
        ch = ch * (maxs[c] - mins[c]) + mins[c]
        denorm.append(ch)
    return torch.stack(denorm)

all_preds_flat = []
all_labels_flat = []
per_channel_preds = [[] for _ in range(labels_channels)]
per_channel_labels = [[] for _ in range(labels_channels)]

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        preds = model(inputs)

        # Remove batch dimension
        preds = preds[0].cpu()
        labels = labels[0].cpu()

        preds_denorm = denormalize_tensor(preds, global_label_min, global_label_max)
        labels_denorm = denormalize_tensor(labels, global_label_min, global_label_max)

        all_preds_flat.append(preds_denorm.flatten())
        all_labels_flat.append(labels_denorm.flatten())

        # Per-channel separation
        for c in range(labels_channels):
            per_channel_preds[c].append(preds_denorm[c].flatten())
            per_channel_labels[c].append(labels_denorm[c].flatten())

# Concatenate all flattened data
global_preds = torch.cat(all_preds_flat).numpy()
global_labels = torch.cat(all_labels_flat).numpy()

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                     channel X peoblem                    │
# └───────────────────────────────────────────────────────────────────────────┘



examples_dir = plot_dir / "channel1_errors"
examples_dir.mkdir(exist_ok=True)

model.eval()
saved = 0
max_examples = 25
threshold_true = 1.0     # True value far from zero
threshold_pred = 0.3     # Predicted value near zero

with torch.no_grad():
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        labels = labels[0].cpu()
        preds = model(inputs)[0].cpu()

        preds_denorm = denormalize_tensor(preds, global_label_min, global_label_max)
        labels_denorm = denormalize_tensor(labels, global_label_min, global_label_max)

        gt = labels_denorm[0]
        pred = preds_denorm[0]

        mask = (gt.abs() > threshold_true) & (pred.abs() < threshold_pred)
        if mask.sum() > 0 and saved < max_examples:
            input_img = inputs[0, 0].cpu().numpy()
            gt_img = gt.numpy()
            pred_img = pred.numpy()

            plt.imsave(examples_dir / f"input_{i}.png", input_img, cmap="gray")
            plt.imsave(examples_dir / f"gt_c1_{i}.png", gt_img, cmap="plasma")
            plt.imsave(examples_dir / f"pred_c1_{i}.png", pred_img, cmap="plasma")
            print(f"Saved channel-1 error example {saved} (sample {i})")
            saved += 1

        if saved >= max_examples:
            break

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                        Plot Global Scatter Plot                           │
# └───────────────────────────────────────────────────────────────────────────┘

def plot_scatter(true, pred, save_path, title="True vs Predicted"):
    plt.figure(figsize=(8, 8))
    plt.scatter(true, pred, alpha=0.05, s=1)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

plot_scatter(global_labels, global_preds, plot_dir / "scatter_all_channels.png")

# ┌───────────────────────────────────────────────────────────────────────────┐
# │                   Plot Per-Channel Scatter Plots                          │
# └───────────────────────────────────────────────────────────────────────────┘

for c in range(labels_channels):
    pred_c = torch.cat(per_channel_preds[c]).numpy()
    label_c = torch.cat(per_channel_labels[c]).numpy()
    save_path = plot_dir / f"scatter_channel_{c+1}.png"
    plot_scatter(label_c, pred_c, save_path, title=f"Channel {c+1}: True vs Predicted")

print("✅ Custom evaluation complete. Scatter plots saved.")
