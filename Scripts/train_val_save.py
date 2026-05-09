import os
import numpy as np
from sklearn.model_selection import train_test_split

def save_ct_slices(ct_array, ct_type, base_dir="post-processed", train_size=0.8):
    # Split the array into training and validation sets
    train_ct, val_ct = train_test_split(ct_array, train_size=train_size)

    # Save function for slices
    def save_slices(slices, set_type):
        set_dir = os.path.join(base_dir, set_type, f"ct_{ct_type}")
        os.makedirs(set_dir, exist_ok=True)

        for i, slice in enumerate(slices):
            slice_path = os.path.join(set_dir, f"slice_{i}.npy")
            np.save(slice_path, slice)

    # Save training and validation slices
    save_slices(train_ct, "train")
    save_slices(val_ct, "val")

home_path = os.path.abspath("/ifs/loni/faculty/dduncan/agarg")
processed_scans_path = os.path.join(home_path, 'processed_scans')
ct_paths = [os.path.join(processed_scans_path, f'ct_{i}.npy') for i in range(5)]

cts = [np.load(path) for path in ct_paths]

for idx, ct in enumerate(cts):
    save_ct_slices(ct, ct_type=idx)
