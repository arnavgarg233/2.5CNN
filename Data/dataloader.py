import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Grayscale
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandRotate90d,
    RandFlipd,
    ToTensord
)
from PIL import Image
from monai.transforms import Transform
import random


CLASS_TRAIN_COUNTS = {
    0: 203,
    1: 546,
    2: 100,
    3: 36,
    4: 1
}

class SliceDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, include_classes=None, class_mapping=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train' or 'val' to specify the split of dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            include_classes (list, optional): List of classes to include (e.g., [0, 1]).
            class_mapping (dict, optional): Dictionary to map original classes to new labels.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.file_list = []
        self.labels = []
        self.transform = transform
        self.include_classes = include_classes

        if class_mapping and not all(k in include_classes for k in class_mapping.keys()):
            raise ValueError("All keys in class_mapping must be in include_classes")

        # Iterate over each CT type directory
        for ct_type in sorted(os.listdir(self.root_dir)):
            print(ct_type)
            class_label = int(ct_type.split('_')[-2])
            if self.include_classes is not None and class_label not in self.include_classes:
                continue
            if class_mapping:
                class_label = class_mapping[class_label]
            ct_type_dir = os.path.join(self.root_dir, ct_type)
            if os.path.isdir(ct_type_dir):
                ct_type_files = sorted(os.listdir(ct_type_dir))
                for file in ct_type_files:
                    if file.endswith('.npy'):
                        self.file_list.append(os.path.join(ct_type_dir, file))
                        self.labels.append(class_label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path = self.file_list[idx]
        label = self.labels[idx]

        # Load the numpy array (2D slice)
        image = np.load(image_path)

        # Applying transformation if any
        if self.transform:
            image = self.transform(image)

        # Convert the numpy array to a PyTorch tensor and add a channel dimension
        image = image.clone().detach().requires_grad_(False)

        return image, torch.tensor(label)

class ApplyGrayscaleTo3DSlices(Transform):
    def __init__(self, keys):
        self.keys = keys
        self.grayscale_transform = Grayscale(num_output_channels=3)

    def __call__(self, data):
        for key in self.keys:
            volume = data[key]
            # Apply the grayscale transform to each slice
            processed_slices = [self.grayscale_transform(volume[:, :, :, i]) for i in range(volume.shape[3])]
            # Stack the processed slices back into a volume
            data[key] = torch.stack(processed_slices, dim=3)
        return data
    
class Normalize3D(Transform):
    def __init__(self, keys, mean, std):
        self.keys = keys
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key in self.keys:
            volume = data[key]
            # Normalize each channel
            for c in range(volume.shape[0]):  # Iterate over channels
                volume[c, :, :, :] = (volume[c, :, :, :] - self.mean[c]) / self.std[c]
            data[key] = volume
        return data
    
class ReplicateChannel(Transform):
    def __init__(self, keys, num_channels=3):
        self.keys = keys
        self.num_channels = num_channels

    def __call__(self, data):
        for key in self.keys:
            volume = data[key]
            # Replicate the single channel across 'num_channels'
            replicated_channels = [volume] * self.num_channels
            data[key] = torch.cat(replicated_channels, dim=0)  # Concatenate along the channel dimension
        return data
class CTScansDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, include_classes=None, class_mapping=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train' or 'val' to specify the split of dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            include_classes (list, optional): List of classes to include (e.g., [0, 1]).
            class_mapping (dict, optional): Dictionary to map original classes to new labels.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.file_list = []
        self.labels = []
        self.transform = transform
        self.include_classes = include_classes

        if class_mapping and not all(k in include_classes for k in class_mapping.keys()):
            raise ValueError("All keys in class_mapping must be in include_classes")

        # Iterate over each CT type directory
        for ct_type in sorted(os.listdir(self.root_dir)):
            class_label = int(ct_type.split('_')[-1])
            if self.include_classes is not None and class_label not in self.include_classes:
                continue
            if class_mapping:
                class_label = class_mapping[class_label]
            ct_type_dir = os.path.join(self.root_dir, ct_type)
            if os.path.isdir(ct_type_dir):
                ct_type_files = sorted(os.listdir(ct_type_dir))
                for file in ct_type_files:
                    if file.endswith('.npy'):
                        self.file_list.append(os.path.join(ct_type_dir, file))
                        self.labels.append(class_label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = {'image': self.file_list[idx]}
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)
        else:
            data['image'] = np.load(data['image'])
            data['image'] = torch.tensor(data['image']).unsqueeze(0).float()

        return data['image'], torch.tensor(label)
    
class NumpyToPIL:
    def __call__(self, numpy_array):
        return Image.fromarray(np.uint8(numpy_array * 255), 'L')  # 'L' mode for grayscale

# Normalization (assuming your data is already scaled between 0 and 1)
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return transforms.functional.normalize(tensor, self.mean, self.std)

class AkshTransforms():
    train_transforms = None
    val_transforms = None

    @classmethod
    def get_train_transforms(cls):
        return cls.train_transforms
    
    @classmethod
    def get_val_transforms(cls):
        return cls.val_transforms

class Transforms3D(AkshTransforms):
    def __ini__(self):
        super().__init__()

    # Class variables for train and validation transforms
    # train_transforms = Compose([
    #     LoadImaged(keys=['image']),
    #     EnsureChannelFirstd(keys=['image']),
    #     ScaleIntensityRanged(
    #         keys=['image'], a_min=-1000, a_max=400,
    #         b_min=0.0, b_max=1.0, clip=True
    #     ),
    #     RandRotate90d(keys=['image'], prob=0.5, spatial_axes=[0, 1]),
    #     RandFlipd(keys=['image'], spatial_axis=0, prob=0.5),
    #     ToTensord(keys=['image'])
    # ])

    train_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image']),
        ReplicateChannel(keys=['image'], num_channels=3),
        Normalize3D(keys=['image'], mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
        ToTensord(keys=['image'])
    ])
    
    # val_transforms = Compose([
    #     LoadImaged(keys=['image']),
    #     EnsureChannelFirstd(keys=['image']),
    #     ScaleIntensityRanged(
    #         keys=['image'], a_min=-1000, a_max=400,
    #         b_min=0.0, b_max=1.0, clip=True
    #     ),
    #     ToTensord(keys=['image'])
    # ])

    val_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image']),
        ReplicateChannel(keys=['image'], num_channels=3),
        Normalize3D(keys=['image'], mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
        ToTensord(keys=['image'])
    ])

    @classmethod
    def get_train_transforms(cls):
        return cls.train_transforms

    @classmethod
    def get_val_transforms(cls):
        return cls.val_transforms

class Transforms2D(AkshTransforms):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, numpy_array):
        return Image.fromarray(np.uint8(numpy_array * 255), 'L')  # 'L' mode for grayscale
    
    train_transform_pipeline = transforms.Compose([
        NumpyToPIL(),
        transforms.Grayscale(num_output_channels=3), # Convert to grayscale
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10),
        transforms.ToTensor(), # Convert the PIL Image to a tensor
        Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]) # Normalization values
    ])

    val_transform_pipeline = transforms.Compose([
        NumpyToPIL(),
        transforms.Grayscale(num_output_channels=3), # Convert to grayscale
        transforms.Resize((128, 128)), # Resize the image to 256x256
        transforms.ToTensor(), # Convert the PIL Image to a tensor
        Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]) # Normalization values
    ])

    @classmethod
    def get_train_transforms(cls):
        return cls.train_transform_pipeline

    @classmethod
    def get_val_transforms(cls):
        return cls.val_transform_pipeline

# Normalization (assuming your data is already scaled between 0 and 1)
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return transforms.functional.normalize(tensor, self.mean, self.std)


TRANSFORMS = {
    '2d': Transforms2D,
    '3d': Transforms3D
}


DATASETS = {
    '2d': SliceDataset,
    '3d': CTScansDataset
}

# ROOT_DIRS = {
#     '2d': 'post-processed-slices',
#     '3d': 'post-processed-slides'
# }

ROOT_DIRS = {
    '2d': 'post-processed-slices',
    '3d': 'post-processed'
}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Example Usage
if __name__ == '__main__':
    # Set seeds for reproducibility
    
    
    sequences = []
    for i in range(10):
        g = torch.Generator()
        g.manual_seed(42)  # You can change this seed value
        
        dataset_id = '2d'
        my_transform = TRANSFORMS[dataset_id]
        root_dir = ROOT_DIRS[dataset_id]
        dataset = DATASETS[dataset_id]
        
        train_dataset = dataset(root_dir=root_dir, split='train', transform=my_transform.get_train_transforms())
        val_dataset = dataset(root_dir=root_dir, split='val', transform=my_transform.get_val_transforms())
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=32, 
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )

        # test the include_classes flags
        train_dataset = dataset(root_dir=root_dir, split='train', transform=my_transform.get_train_transforms(), include_classes=[0, 1])
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # test the class_mapping flag
        train_dataset = dataset(root_dir=root_dir, split='train', transform=my_transform.get_train_transforms(), include_classes=[0, 1], class_mapping={0: 0, 1: 0})
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # test the class_mapping flag with a class not in include_classes
        train_dataset = dataset(root_dir=root_dir, split='train', transform=my_transform.get_train_transforms(), include_classes=[0, 1], class_mapping={0: 0, 1: 2})
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        seq_i = []
        for i, (images, labels) in enumerate(train_dataloader):
            seq_i.append(labels)
            break
        sequences.append(seq_i)

    print(sequences)

    breakpoint()
