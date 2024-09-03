import os
import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, TensorDataset

# Import your model class
from vnet import VNet  # Make sure to replace 'your_model_file' with the actual file name


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


def load_model(model_path, num_classes):
    """
    Load the trained model.
    """
    model = VNet(in_channels=1, num_classes=num_classes)

    # Determine the map_location based on CUDA availability
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')

    # Load the state dict with appropriate map_location
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def predict(model, image):
    """
    Make a prediction for a single image.
    """
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.cpu().numpy()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set paths
    model_path = '../models/vnet_model.pth'  # Replace with your model path
    image_dir = '../data/raw/images'  # Replace with your test image directory
    num_classes = 14  # Replace with the number of classes in your model

    # Load model
    model = load_model(model_path, num_classes)
    model = model.to(device)

    # Process each image in the directory
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.nii'):
            image_path = os.path.join(image_dir, image_file)

            # Preprocess the image
            preprocessed_image = preprocess_image(image_path)
            image_tensor = torch.from_numpy(preprocessed_image).float().unsqueeze(0).to(device)

            # Make prediction
            prediction = predict(model, image_tensor)

            print(f"Prediction for {image_file}: {prediction}")


#if __name__ == " main_":
main()