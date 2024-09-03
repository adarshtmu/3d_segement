import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
from scipy.ndimage import zoom
import numpy as np


def preprocess_image(image_path, target_shape=(128, 128, 128)):
    """
    Load and preprocess a single image.
    """
    img = nib.load(image_path).get_fdata()
    img_resized = resize_volume(img, target_shape)
    return np.expand_dims(img_resized, axis=0)  # Add channel dimension


def resize_volume(volume, target_shape):
    """
    Resize the volume to the target shape.
    """
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, zoom_factors, order=1)  # Use order=1 for bilinear interpolation


def plot_3d_volume(volume, title="3D Volume"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    x, y, z = volume.nonzero()
    ax.scatter(x, y, z, c=volume[x, y, z], cmap='viridis')
    plt.show()


if __name__ == "__main__":
    image_folder = '../data/raw/images/'

    for i in range(1, 51):  # Loop from _0001 to _0050
        file_name = f'FLARE22_Tr_{i:04d}_0000.nii'
        volume_path = os.path.join(image_folder, file_name)

        if os.path.exists(volume_path):
            print(f"Processing {file_name}...")
            volume = preprocess_image(volume_path)
            plot_3d_volume(volume[0, :, :, :], title=file_name)
        else:
            print(f"File {file_name} not found.")
