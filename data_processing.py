import os
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom
from vnet import VNet  # Import VNet here

def preprocess_image(image_path, target_shape=(128, 128, 128)):
    img = nib.load(image_path).get_fdata()
    img_resized = resize_volume(img, target_shape)
    return np.expand_dims(img_resized, axis=0)

def resize_volume(volume, target_shape):
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, zoom_factors, order=1)

def load_model(model_path, num_classes):
    model = VNet(in_channels=1, num_classes=num_classes)
    map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(model, image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.cpu().numpy()

def dice_score(pred, target):
    smooth = 1e-6
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    return (2. * intersection + smooth) / (union + smooth)

def plot_3d_volume(volume, title="3D Volume"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    x, y, z = np.nonzero(volume)
    ax.scatter(x, y, z, c=volume[x, y, z], cmap='gray')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '../models/vnet_model.pth'
    image_dir = '../data/raw/images'
    label_dir = '../data/raw/labels'
    num_classes = 14

    model = load_model(model_path, num_classes)
    model = model.to(device)

    all_dice_scores = {i: [] for i in range(num_classes)}

    for image_file in os.listdir(image_dir):
        if image_file.endswith('.nii'):
            image_path = os.path.join(image_dir, image_file)
            preprocessed_image = preprocess_image(image_path)
            image_tensor = torch.from_numpy(preprocessed_image).float().unsqueeze(0).to(device)
            prediction = predict(model, image_tensor)

            label_file = image_file.replace('.nii', '.nii')
            label_path = os.path.join(label_dir, label_file)
            label = preprocess_image(label_path)

            if prediction.shape != label.shape:
                label = resize_volume(label, prediction.shape[2:])

            dice = dice_score(prediction, label)

            for i in range(num_classes):
                class_dice = dice_score(prediction == i, label == i)
                all_dice_scores[i].append(class_dice)

            plot_3d_volume(prediction[0, :, :, :], title=f"Prediction for {image_file}")

    for i in range(num_classes):
        avg_dice = np.mean(all_dice_scores[i])
        print(f"Average Dice score for class {i}: {avg_dice}")

if __name__ == "__main__":
    main()
