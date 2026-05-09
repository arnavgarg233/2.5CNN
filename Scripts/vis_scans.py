import numpy as np
import imageio
import os

def create_gif_from_3d_scan(scan_data, gif_path, duration=0.1):
    """
    Create a GIF from a 3D CT scan.

    :param scan_data: 3D numpy array of the CT scan (shape: slices x height x width)
    :param gif_path: Path to save the GIF
    :param duration: Duration of each frame in the GIF
    """
    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
        print(scan_data.shape)
        for i in range(scan_data.shape[2]):
            writer.append_data(scan_data[:, : ,i])
    print(f"GIF saved at {gif_path}")





if __name__ == '__main__':
    scan_file = "/workspace/new/ct_severity/post-processed/train/ct_0/slice_1.npy"
    base_folder = "/workspace/new/ct_severity/"
    for i in range(5):
        for j in range(10):
            scan_file = os.path.join(base_folder, f"post-processed/train/ct_{i}/slice_{j}.npy")
            scan = np.load(scan_file)
            gif_folder = os.path.join(base_folder, "gifs")
            os.makedirs(gif_folder, exist_ok=True)
            ct_folder = os.path.join(base_folder, f"gifs/ct_{i}")
            os.makedirs(ct_folder, exist_ok=True)
            gif_path = os.path.join(base_folder, f"{ct_folder}/slice_{j}.gif")
            create_gif_from_3d_scan(scan, gif_path)