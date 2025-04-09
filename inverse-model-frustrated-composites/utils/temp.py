import torch
import torch.nn as nn


class AngularL1Loss(nn.Module):
    def __init__(self):
        super(AngularL1Loss, self).__init__()

    def forward(self, predictions, labels):

        # Compute the absolute difference
        diff = torch.abs(predictions - labels)

        # Wrap the differences to ensure they are between 0° and 90°
        wrapped_diff = torch.minimum(diff, 180 - diff)

        # Take the mean (L1 loss)
        loss = wrapped_diff.mean()

        # Debugging prints
        print(f"predictions {predictions}")
        print(f"labels {labels}")
        print(f"Diff: {diff}")
        print(f"Wrapped Diff: {wrapped_diff}")
        print(f"Loss: {loss}")

        return loss


# Test

# Instantiate the loss
loss_fn = AngularL1Loss()

# Sample predictions and labels in range 0–180
predictions = torch.tensor([10, 170, 30, 90, 90, 90], dtype=torch.float32)
labels = torch.tensor([20, 10, 160, 95, 180, 0], dtype=torch.float32)

# Compute the loss
loss = loss_fn(predictions, labels)
print(f"Final loss value: {loss.item()}")
