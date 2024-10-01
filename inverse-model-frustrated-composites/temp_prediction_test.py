import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb


# Assuming you have already defined your model architecture
class OurModel(torch.nn.Module):
    def __init__(self, features_channels=1, labels_channels=3, dropout=0.3):
        super(OurModel, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=features_channels, out_channels=32, kernel_size=3, padding=1)
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
        x = self.dropout(x)

        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.relu(x)

        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.relu(x)

        x = self.conv_7(x)
        x = self.batch_norm_7(x)
        x = self.relu(x)

        x = self.conv_8(x)
        x = self.batch_norm_8(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv_9(x)
        x = self.batch_norm_9(x)
        x = self.relu(x)

        x = self.conv_10(x)
        x = self.batch_norm_10(x)
        x = self.relu(x)

        x = self.conv_11(x)
        return x


# Generate a random fiber orientation with 3x4 patches
def generate_random_fiber_orientation():
    random_orientations = torch.randint(0, 181, (3, 4), dtype=torch.float32)
    initial_fiber_orientation = torch.zeros((1, 1, 15, 20), dtype=torch.float32)

    for i in range(3):
        for j in range(4):
            initial_fiber_orientation[:, 0, i * 5:(i + 1) * 5, j * 5:(j + 1) * 5] = random_orientations[i, j]

    return initial_fiber_orientation


# Function to visualize the predicted XYZ surface
def visualize_predicted_xyz(points_xyz, title="Predicted XYZ Surface"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the predicted points
    ax.scatter(points_xyz[:, :, 0], points_xyz[:, :, 1], points_xyz[:, :, 2], color='b', label='Predicted XYZ')

    ax.set_title(title)
    plt.show()


# Main program to generate fiber orientation, predict XYZ, and visualize it
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model and load it onto the device
    model = OurModel().to(device)

    # Load the trained model weights (make sure to specify the correct path)
    model.load_state_dict(torch.load(r'C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models_for_checks\17-24_Location_Features_20241001.pkl', map_location=device))

    model.eval()  # Set the model to evaluation mode

    # Generate random fiber orientation and move it to the device
    initial_fiber_orientation = generate_random_fiber_orientation().to(device)

    #print(initial_fiber_orientation)
    print(initial_fiber_orientation.shape)

    # Predict XYZ using the model
    with torch.no_grad():
        predicted_xyz = model(initial_fiber_orientation)
        print(f"predicted shape: {predicted_xyz.shape}")
        predicted_xyz = predicted_xyz.permute(0,3,2,1).cpu().numpy().squeeze()
        print(f"predicted shape: {predicted_xyz.shape}")


    # Visualize the predicted XYZ surface
    visualize_predicted_xyz(predicted_xyz)
