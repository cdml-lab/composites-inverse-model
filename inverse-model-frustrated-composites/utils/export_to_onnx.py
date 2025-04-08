
# ┌───────────────────────────────────────────────────────────────────────────┐
# │                          Imports and Inputs                               |
# └───────────────────────────────────────────────────────────────────────────┘

import torch

model_path = r"C:\Gal_Msc\Ipublic-repo\inverse-model-frustrated-composites\saved_models\fiery-cosmos.pkl"
features_channels = 1
labels_channels = 3
height = 30
width = 30


# ┌───────────────────────────────────────────────────────────────────────────┐
# │                             Model Classes                                 |
# └───────────────────────────────────────────────────────────────────────────┘


class ReducedWidth(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(ReducedWidth, self).__init__()

        self.conv_1 = torch.nn.Conv2d(features_channels, 64, kernel_size=3, padding=1)
        self.conv_2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_3 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_5 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_6 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_7 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_8 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_9 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_10 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_11 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_12 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_13 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_14 = torch.nn.Conv2d(64, labels_channels, kernel_size=3, padding=1)

        self.batch_norm_1 = torch.nn.BatchNorm2d(64)
        self.batch_norm_2 = torch.nn.BatchNorm2d(128)
        self.batch_norm_3 = torch.nn.BatchNorm2d(128)
        self.batch_norm_4 = torch.nn.BatchNorm2d(128)
        self.batch_norm_5 = torch.nn.BatchNorm2d(128)
        self.batch_norm_6 = torch.nn.BatchNorm2d(256)
        self.batch_norm_7 = torch.nn.BatchNorm2d(256)
        self.batch_norm_8 = torch.nn.BatchNorm2d(256)
        self.batch_norm_9 = torch.nn.BatchNorm2d(256)
        self.batch_norm_10 = torch.nn.BatchNorm2d(128)
        self.batch_norm_11 = torch.nn.BatchNorm2d(128)
        self.batch_norm_12 = torch.nn.BatchNorm2d(64)
        self.batch_norm_13 = torch.nn.BatchNorm2d(64)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.relu(self.batch_norm_2(self.conv_2(x)))
        x = self.relu(self.batch_norm_3(self.conv_3(x)))
        x = self.relu(self.batch_norm_4(self.conv_4(x)))
        x = self.relu(self.batch_norm_5(self.conv_5(x)))
        x = self.relu(self.batch_norm_6(self.conv_6(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm_7(self.conv_7(x)))
        x = self.relu(self.batch_norm_8(self.conv_8(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm_9(self.conv_9(x)))
        x = self.relu(self.batch_norm_10(self.conv_10(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm_11(self.conv_11(x)))
        x = self.relu(self.batch_norm_12(self.conv_12(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm_13(self.conv_13(x)))
        x = self.conv_14(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x




# Make sure the model class is defined or imported
model = ReducedWidth()  # or OurVgg16() depending on which you're using
model.load_state_dict(torch.load(model_path))
model.eval()

dummy_input = torch.randn(1, features_channels, height, width)
torch.onnx.export(model, dummy_input, "C:/Gal_Msc/Ipublic-repo/inverse-model-frustrated-composites/onnx/model.onnx")
