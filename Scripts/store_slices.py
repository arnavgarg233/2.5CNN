import numpy as np
import os
from sklearn.model_selection import train_test_split

def save_ct_slices(ct_array, ct_type, base_dir="post-processed-slices", train_size=0.8):
    # Split the array into training and validation sets
    train_ct, val_ct = train_test_split(ct_array, train_size=train_size)

    # Save function for slices
    def save_slices(slices, set_type):
        set_dir = os.path.join(base_dir, set_type, f"ct_{ct_type}_slice")
        os.makedirs(set_dir, exist_ok=True)

        counter = 0
        for i, volume in enumerate(slices):
            depth = volume.shape[2]
            for i in range(depth):
                slice = volume[:, :, i]
                slice_file_name = f"slice_{counter}.npy"
                slice_path = os.path.join(set_dir, slice_file_name)
                np.save(slice_path, slice)
                counter += 1

    # Save training and validation slices
    save_slices(train_ct, "train")
    save_slices(val_ct, "val")


home_path = os.path.abspath("/ifs/loni/faculty/dduncan/agarg")
processed_scans_path = os.path.join(home_path, 'processed_scans')
ct_paths = [os.path.join(processed_scans_path, f'ct_{i}.npy') for i in range(5)]

cts = [np.load(path) for path in ct_paths]

for idx, ct in enumerate(cts):
    save_ct_slices(ct, ct_type=idx)