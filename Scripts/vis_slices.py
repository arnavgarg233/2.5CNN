from PIL import Image
import numpy as np
import os
import time

if __name__ == '__main__':
    base_folder = "/workspace/new/ct_severity/"
    for i in range(5):
        for j in range(1):
            for centering in range(10,15):
                slice_to_pick = 64*j + centering
                scan_file = os.path.join(base_folder, f"post-processed-slices/train/ct_{i}_slice/slice_{slice_to_pick}.npy")
                scan = np.load(scan_file)
                ct_folder = os.path.join(base_folder, f"imgs/ct_{i}")
                img_folder = os.path.join(base_folder, "imgs")

                img_path = os.path.join(base_folder, f"{ct_folder}/slice_{slice_to_pick}.jpg")
                scan *= 255   
                scan = scan.astype(np.uint8)
                img = Image.fromarray(scan)
                img.save(img_path)
            