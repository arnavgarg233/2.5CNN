import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from tqdm import tqdm

import time
from torch.utils.data import DataLoader
# Assuming CTScansDataset is your custom dataset class
from dataloader import CTScansDataset, TRANSFORMS, ROOT_DIRS, DATASETS

def memory_utilization(activation_shapes):
    individual_activations = np.array([np.prod(shape) for shape in activation_shapes]) * 4 / (1024 ** 3)
    print(individual_activations.round(3))
    total_activations_size = sum(individual_activations)
    print("Total activations memory size: {:.6f} GB".format(total_activations_size))

class Simple2DCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.0):
        super(Simple2DCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Adjust the input channels as needed
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjust the input features to match your data
        self.fc2 = nn.Linear(512, 128)  # Adjust the input features to match your data
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        self.dropout2d = nn.Dropout2d(dropout)

        # Batch Norm
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(self.dropout2d(F.relu(self.bn1(self.conv1(x)))))
        # print(x.shape)
        x = self.pool(self.dropout2d(F.relu(self.bn2(self.conv2(x)))))
        # print(x.shape)
        x = self.pool(self.dropout2d(F.relu(self.bn3(self.conv3(x)))))
        # print(x.shape)

        # Flattening the output for the fully connected layer
        x = x.view(-1, 128 * 16 * 16)  # Adjust the dimensions to match your data

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
    
class SliceCNN(nn.Module):
    def __init__(self, num_channels, dropout):
        super(SliceCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Adjust the input channels as needed
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjust the input features to match your data
        self.fc2 = nn.Linear(512, 128)  # Adjust the input features to match your data
        self.fc3 = nn.Linear(128, num_channels)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        self.dropout2d = nn.Dropout2d(dropout)

        # Batch Norm
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        shapes = []
        shapes.append(x.shape)

        x = self.pool(self.dropout2d(F.relu(self.bn1(self.conv1(x)))))
        shapes.append(x.shape)
        # print(x.shape)
        x = self.pool(self.dropout2d(F.relu(self.bn2(self.conv2(x)))))
        shapes.append(x.shape)
        # print(x.shape)
        x = self.pool(self.dropout2d(F.relu(self.bn3(self.conv3(x)))))
        shapes.append(x.shape)
        # print(x.shape)

        # Flattening the output for the fully connected layer
        x = x.view(-1, 128 * 16 * 16)  # Adjust the dimensions to match your data

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        shapes.append(x.shape)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        shapes.append(x.shape)
        x = self.dropout(x)
        return x, shapes
    
    
class VolumeCNNSharedWeights(nn.Module):
    def __init__(self, num_slices=64, num_channels=128, num_classes=5, dropout=0):
        super(VolumeCNNSharedWeights, self).__init__()
        self.num_slices = num_slices
        self.slice_cnn = SliceCNN(num_channels, dropout=dropout)  # Single instance
        self.fc1 = nn.Linear(num_channels * num_slices, 512)  # Adjust size accordingly
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x is the entire volume: (N, num_slices, num_channels, H, W)
        batch_size = x.size(0)
        processed_slices = []

        # Apply the same SliceCNN to each slice
        for i in range(self.num_slices):
            slice_output, _ = self.slice_cnn(x[:, :, :, :, i])
            processed_slices.append(slice_output.view(batch_size, -1))

        # Concatenate the output features from all slices
        concatenated_slices = torch.cat(processed_slices, dim=1)

        # Pass through the DNN
        concatenated_slices = F.relu(self.fc1(concatenated_slices))
        concatenated_slices = F.relu(self.fc2(concatenated_slices))
        out = self.fc3(concatenated_slices)
        return out
    
    def forward_with_memory(self, x):
        # x is the entire volume: (N, num_slices, num_channels, H, W)
        batch_size = x.size(0)
        processed_slices = []
        shapes = None
        # Apply the same SliceCNN to each slice
        for i in range(self.num_slices):
            slice_output, shapes = self.slice_cnn(x[:, :, :, :, i])
            processed_slices.append(slice_output.view(batch_size, -1))

        new_shapes = []
        for shape in shapes:
            shape = np.array(shape)
            shape[0] = self.num_slices * shape[0]
            new_shapes.append(shape)
        shapes = new_shapes
        # Concatenate the output features from all slices
        concatenated_slices = torch.cat(processed_slices, dim=1)
        shapes.append(concatenated_slices.shape)

        # Pass through the DNN
        concatenated_slices = F.relu(self.fc1(concatenated_slices))
        shapes.append(concatenated_slices.shape)
        concatenated_slices = F.relu(self.fc2(concatenated_slices))
        shapes.append(concatenated_slices.shape)
        out = self.fc3(concatenated_slices)
        return out, shapes

class CustomResNet(nn.Module):
    def __init__(self, num_classes, freeze_pretrained=True, dropout=0.0):
        super(CustomResNet, self).__init__()
        # Load a pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)

        # Freeze the parameters of the pre-trained model if required
        if freeze_pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = True

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() 

        # Add custom layers
        self.custom_layers = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # print(x.shape)
        # Pass x through the pre-trained layers, then the custom layers
        x = self.resnet(x)
        x = self.custom_layers(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 5 
    # model = VolumeCNNSharedWeights(64, 128, num_classes=num_classes)
    model = CustomResNet(num_classes=num_classes)
    model = model.to(device)

    dataset_id = '2d'
    my_transform = TRANSFORMS[dataset_id]
    root_dir = ROOT_DIRS[dataset_id]
    dataset = DATASETS[dataset_id]

    # Run through a full batch
    # dataset = dataset(root_dir=root_dir, split='train', transform=my_transform.get_val_transforms())
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # for batch in tqdm(dataloader, desc=f"Epoch  (Validation)"):
    #     images, labels = batch
    #     images, labels = images.to(device), labels.to(device)
    #     outputs = model(images)


    # Do memory footprinting
    inputs = torch.randn(16, 3, 128, 128).to(device)
    out = model(inputs)
    # out, activations = model(inputs)
    # memory = memory_utilization(activations)
    # print(memory)

"""
num_classes = 5  # Adjust based on your dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Simple2DCNN(num_classes=num_classes)
model = model.to(device)
# inputs = torch.randn(1, 1, 128, 128).to(device)  # Adjust the input channels to match your data
# outputs = model(inputs)

dataset_id = '2d'
my_transform = TRANSFORMS[dataset_id]
root_dir = ROOT_DIRS[dataset_id]
dataset = DATASETS[dataset_id]
dataset = dataset(root_dir=root_dir, split='train', transform=my_transform.get_val_transforms())
# size = 5
# dataset = torch.utils.data.Subset(dataset, list(range(size)) + list(range(-size, 0)))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

for batch in tqdm(dataloader, desc=f"Epoch  (Validation)"):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
"""
    