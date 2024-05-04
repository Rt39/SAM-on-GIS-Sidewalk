import torch
import numpy as np
import tifffile
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50

def calculate_centroids(image_label: np.ndarray) -> np.ndarray:
    width, height = image_label.shape

    # Dimensions of each grid cell
    grid_size_x = width // 8
    grid_size_y = height // 8

    # Initialize the result array with -1 and flag 0
    result_array = np.full((8, 8, 3), [-1, -1, 0]).astype(np.float32)

    # Process each grid cell
    for i in range(8):
        for j in range(8):
            x_start = j * grid_size_x
            y_start = i * grid_size_y
            x_end = x_start + grid_size_x
            y_end = y_start + grid_size_y

            # Extract the cell from the binary mask
            cell = image_label[y_start:y_end, x_start:x_end]

            # Find pixels belonging to the object in this cell
            points = np.column_stack(np.where(cell > 0))

            if points.size > 0:
                # Calculate the centroid of these points
                centroid = np.mean(points, axis=0)
                # Adjust centroid to the coordinate in the full image
                result_array[i, j] = [centroid[1] + x_start, centroid[0] + y_start, 1]

    return result_array

class SidewalkPrompter(nn.Module):
    def __init__(self):
        super(SidewalkPrompter, self).__init__()
        self.resnet = resnet50()
        self.resnet.fc = nn.Linear(in_features=2048, out_features=192, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), 8, 8, 3)
        x[:, :, :, 2] = self.sigmoid(x[:, :, :, 2])
        return x
    
class LossFn(nn.Module):
    """
    Loss function for SidewalkPrompter
    
    Details:
    - Ground truth is a 8x8x3 tensor with the last dimension being [x, y, flag]
    - flag is 1 if there is a centroid in the grid cell, 0 otherwise.
    - Loss fn = MSE([x_hat, y_hat], [x, y]) (if flag == 1) + \lambda BCE(flag_hat, flag)
    - \lambda is a hyperparameter, default value is 5.
    """
    def __init__(self):
        super(LossFn, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')  # Set reduction to 'sum'
        self.bce = nn.BCELoss(reduction='sum')  # Set reduction to 'sum'
        
    def forward(self, pred, target):
        # Extract x, y, flag from the target tensor
        x, y, flag = target[:, :, :, 0], target[:, :, :, 1], target[:, :, :, 2]
        
        # Extract x_hat, y_hat, flag_hat from the pred tensor
        x_hat, y_hat, flag_hat = pred[:, :, :, 0], pred[:, :, :, 1], pred[:, :, :, 2]
        
        # Calculate MSE loss only for grid cells where flag is 1
        mse_loss = self.mse(x_hat * flag, x * flag) + self.mse(y_hat * flag, y * flag)
        
        # Calculate BCE loss
        bce_loss = self.bce(flag_hat, flag)

        return mse_loss + 5 * bce_loss