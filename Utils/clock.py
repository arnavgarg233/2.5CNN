import time
from torch.utils.data import DataLoader
# Assuming CTScansDataset is your custom dataset class
from dataloader import CTScansDataset, TRANSFORMS, ROOT_DIRS, DATASETS


# Create your DataLoader
dataset_id = '3d'
my_transform = TRANSFORMS[dataset_id]
root_dir = ROOT_DIRS[dataset_id]
dataset = DATASETS[dataset_id]
dataset = dataset(root_dir=root_dir, split='train', transform=my_transform.get_val_transforms())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Start timing
start_time = time.time()

# Load all batches
for i, (images, labels) in enumerate(dataloader):
    pass  # Replace with actual processing if needed

# End timing
end_time = time.time()

# Calculate and print total time taken
total_time = end_time - start_time
print(f"Total time for loading data: {total_time} seconds")
