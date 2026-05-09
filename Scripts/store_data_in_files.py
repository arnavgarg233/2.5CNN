import os
import zipfile
import numpy as np
import nibabel as nib
from scipy import ndimage


home_path = os.path.abspath("/ifs/loni/faculty/dduncan/agarg")
processed_scans_path = os.path.join(home_path, 'processed_scans')
ct_paths = [os.path.join(processed_scans_path, f'ct_{i}.npy') for i in range(5)]

# Load Data
cts = [np.load(path) for path in ct_paths]

def save_ct_slices(ct_array, ct_type, save_dir="post-processed"):
    ct_type_dir = os.path.join(save_dir, f"ct_type_{ct_type}")
    os.makedirs(ct_type_dir, exist_ok=True)

    for i, slice in enumerate(ct_array):
        slice_path = os.path.join(ct_type_dir, f"ct_{i}.npy")
        np.save(slice_path, slice)

for idx, ct in enumerate(cts):
    save_ct_slices(ct, ct_type=idx)
