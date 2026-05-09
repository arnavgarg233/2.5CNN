import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        self.fc1 = nn.Linear(256 * 16 * 16 * 8, 512)  # Adjust the input features
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 16 * 16 * 8)  # Adjust based on your input dimensions
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward_with_memory_footprint(self, x):
        input_shape = x.shape

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        activation_shape_1 = x.shape

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        activation_shape_2 = x.shape

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        activation_shape_3 = x.shape

        x = x.view(-1, 256 * 16 * 16 * 8)  # Adjust based on your input dimensions
        flatten_shape = x.shape

        x = F.relu(self.fc1(x))
        fc1_shape = x.shape

        # x = self.dropout(x)
        x = self.fc2(x)
        fc2_shape = x.shape

        return x, [input_shape, activation_shape_1, activation_shape_2, activation_shape_3, flatten_shape, fc1_shape, fc2_shape]
    
def memory_utilization(activation_shapes):
    individual_activations = np.array([np.prod(shape) for shape in activation_shapes]) * 4 / (1024 ** 3)
    print(individual_activations.round(3))
    total_activations_size = sum(individual_activations)
    print("Total activations memory size: {:.6f} GB".format(total_activations_size))

class Simple3DCNNBN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNNBN, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        self.fc1 = nn.Linear(256 * 16 * 16 * 8, 512)  # Adjust the input features
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 16 * 16 * 8)  # Adjust based on your input dimensions
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def forward_with_memory_footprint(self, x):
        input_shape = x.shape

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        activation_shape_1 = x.shape

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        activation_shape_2 = x.shape

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        activation_shape_3 = x.shape

        x = x.view(-1, 256 * 16 * 16 * 8)



class Simple3DCNNBN_Modular(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNNBN_Modular, self).__init__()
        self.conv1 = Conv3DBlock(1, 32)
        self.pool = nn.MaxPool3d(2)

        self.conv2 = Conv3DBlock(32, 64)
        self.pool = nn.MaxPool3d(2)

        self.conv3 = Conv3DBlock(64, 128)
        self.pool = nn.MaxPool3d(2)

        self.conv4 = Conv3DBlock(128, 256)
        self.pool = nn.MaxPool3d(2)

        self.conv5 = Conv3DBlock(256, 512)
        self.pool = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(512 * 4 * 4 * 2, 1024)  # Adjust the input features
        self.fc2 = nn.Linear(1024, 512)  # Adjust the input features
        self.fc3 = nn.Linear(512, num_classes)
        # self.dropout = nn.Dropout(p=0.5)

        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))

        x = x.view(-1, 512 * 4 * 4 * 2)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        x = self.fc3(x)

        return x
    
    def forward_with_memory_footprint(self, x):
        input_shape = x.shape

        x = self.pool(self.conv1(x))
        activation_shape_1 = x.shape
        print(activation_shape_1)

        x = self.pool(self.conv2(x))
        activation_shape_2 = x.shape
        print(activation_shape_2)

        x = self.pool(self.conv3(x))
        activation_shape_3 = x.shape
        print(activation_shape_3)

        x = self.pool(self.conv4(x))
        activation_shape_4 = x.shape
        print(activation_shape_4)

        x = self.pool(self.conv5(x))
        activation_shape_5 = x.shape
        print(activation_shape_5)

        x = x.view(-1, 512 * 4 * 4 * 2)
        flatten_shape = x.shape

        x = F.relu(self.fc1(x))
        fc1_shape = x.shape

        # x = self.dropout(x)
        x = self.fc2(x)
        fc2_shape = x.shape

        # x = self.dropout(x)
        x = self.fc3(x)
        fc3_shape = x.shape

        return x, [input_shape, activation_shape_1, activation_shape_2, activation_shape_3, activation_shape_4, activation_shape_5, flatten_shape, fc1_shape, fc2_shape, fc3_shape]

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CNN3DWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(CNN3DWithAttention, self).__init__()
        self.conv1 = Conv3DBlock(1, 32)
        self.pool = nn.MaxPool3d(2)
        self.se1 = SEBlock(32)

        self.conv2 = Conv3DBlock(32, 64)
        self.se2 = SEBlock(64)

        self.conv3 = Conv3DBlock(64, 128)
        self.se3 = SEBlock(128)
        
        self.conv4 = Conv3DBlock(128, 256)
        self.se4 = SEBlock(256)

        self.conv5 = Conv3DBlock(256, 512)
        self.se5 = SEBlock(512)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.se1(x)
        # print(x.shape)

        x = self.pool(self.conv2(x))
        x = self.se2(x)
        # print(x.shape)

        x = self.pool(self.conv3(x))
        x = self.se3(x)
        # print(x.shape)
        
        x = self.pool(self.conv4(x))
        x = self.se4(x)
        # print(x.shape)
        
        x = self.pool(self.conv5(x))
        x = self.se5(x)
        # print(x.shape)

        x = self.avg_pool(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = F.relu(self.fc1(x))
        # print(x.shape)

        x = F.relu(self.fc2(x))
        # print(x.shape)

        x = self.fc3(x)
        # print(x.shape)

        return x
    
    def forward_with_memory_footprint(self, x):
        input_shape = x.shape

        x = self.pool(self.conv1(x))
        activation_shape_1 = x.shape
        x = self.se1(x)
        print(x.shape)

        x = self.pool(self.conv2(x))
        activation_shape_2 = x.shape
        x = self.se2(x)
        print(x.shape)

        x = self.pool(self.conv3(x))
        activation_shape_3 = x.shape
        x = self.se3(x)
        print(x.shape)

        x = self.pool(self.conv4(x))
        activation_shape_4 = x.shape
        x = self.se4(x)
        print(x.shape)

        x = self.pool(self.conv5(x))
        activation_shape_5 = x.shape
        x = self.se5(x)
        print(x.shape)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        print(x.shape)

        flatten_shape = x.shape
        x = F.relu(self.fc1(x))
        fc1_shape = x.shape
        print(x.shape)

        x = F.relu(self.fc2(x))
        fc2_shape = x.shape
        print(x.shape)

        x = self.fc3(x)
        fc3_shape = x.shape
        print(x.shape)

        return x, [input_shape, activation_shape_1, activation_shape_2, activation_shape_3, activation_shape_4, activation_shape_5, flatten_shape, fc1_shape, fc2_shape, fc3_shape]



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 5  # Adjust based on your dataset
    # model = Simple3DCNNBN_Modular(num_classes).to(device)
    model = CNN3DWithAttention(num_classes).to(device)

    # Example input
    inputs = torch.randn(16, 1, 128, 128, 64).to(device)
    x, activation_shapes = model.forward_with_memory_footprint(inputs)
    out = model(inputs)
    Simple3DCNN.memory_utilization(activation_shapes)



