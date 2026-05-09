import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import CTScansDataset

class Normalize3D:
    """Normalize a 3D image with mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std

# Example transformations
transform = transforms.Compose([
    Normalize3D(mean=0.5, std=0.5),  # Example values, adjust as needed
    transforms.ToTensor(),  # If your images are not already in tensor format
])


if __name__ == '__main__':
    root_dir = 'post-processed'  # Adjust as necessary
    train_dataset = CTScansDataset(root_dir=root_dir, split='train', transform=transform)
    val_dataset = CTScansDataset(root_dir=root_dir, split='val', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
