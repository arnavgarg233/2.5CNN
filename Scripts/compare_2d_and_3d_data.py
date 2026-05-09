import torch
from torch.utils.data import DataLoader, SequentialSampler
from dataloader import TRANSFORMS, ROOT_DIRS, DATASETS

def get_dataloader(dataset_id, batch_size=64, shuffle_blocks=True, num_workers=0, mode='val', 
                  include_classes=None, class_mapping=None, seed=42):
    """
    Returns a DataLoader that yields (images, labels) in blocks of 'batch_size'.
    If shuffle_blocks is True, the blocks of size batch_size are shuffled.
    
    Args:
        ...
        seed (int): Random seed for reproducibility
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    
    # Prepare the dataset
    my_transform = TRANSFORMS[dataset_id]
    root_dir = ROOT_DIRS[dataset_id]
    dataset_cls = DATASETS[dataset_id]
    
    if mode == 'val':
        transform = my_transform.get_val_transforms()
    elif mode == 'train':
        transform = my_transform.get_train_transforms()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    dataset = dataset_cls(
        root_dir=root_dir, 
        split=mode, 
        transform=transform,
        include_classes=include_classes,
        class_mapping=class_mapping
    )

    # Create indices
    n_samples = len(dataset)
    indices = torch.arange(n_samples)
    # Group indices into blocks of batch_size
    batch_indices = indices.view(-1, batch_size)

    # Optionally shuffle the batch blocks as a whole
    if shuffle_blocks:
        # Use the same seed for shuffling
        generator = torch.Generator()
        generator.manual_seed(seed)
        # This randomizes the order of the blocks, but keeps items in each block together
        batch_indices = batch_indices[torch.randperm(len(batch_indices), generator=generator)]

    # Flatten back to 1D
    final_indices = batch_indices.flatten()

    # Define sampler
    class OrderedSubsetSampler(torch.utils.data.Sampler):
        def __init__(self, indices):
            self.indices = indices
        def __iter__(self):
            return iter(self.indices)
        def __len__(self):
            return len(self.indices)

    sampler = OrderedSubsetSampler(final_indices)

    # Build the DataLoader with the generator
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        generator=generator  # Use the same generator for worker initialization
    )
    return dataloader

if __name__ == "__main__":
    # Example usage
    dataloader_2d = get_dataloader('2d', batch_size=64, shuffle_blocks=True)
    for i, (images, labels) in enumerate(dataloader_2d):
        print(images.shape)
        print(f"Batch {i}, labels: {labels}")
        # if i >= 3:
        #     break

