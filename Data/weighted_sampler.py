from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from dataset import CTScansDataset

# Assuming your dataset object is train_dataset
class_train_counts = {
    0: 203,
    1: 546,
    2: 100,
    3: 36,
    4: 1
}
class_val_counts = {
    0: 51,
    1: 137,
    2: 25,
    3: 9,
    4: 1
}

class_counts = class_train_counts
num_samples = sum(class_counts.values())

# Extract labels from the dataset
train_dataset = CTScansDataset(root_dir=root_dir, split='train', transform=train_transforms, include_classes=classes_to_include, class_mapping=class_mapping)
labels = np.array([label for _, label in train_dataset])

# Create weights for each sample
weights = [num_samples/class_counts[label] for label in labels]
sampler = WeightedRandomSampler(weights, num_samples, replacement=True)

# DataLoader with the sampler
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
