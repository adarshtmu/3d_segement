import torch
import nibabel as nib
from src.preprocess import resize_volume
from src.visualize import visualize_3d, save_visualization_video
from src.model import VNet
from src.evaluate import dice_score  # Import if you want to calculate dice score on a single prediction

# Load the model
model = VNet(in_channels=1, num_classes=4)
model.load_state_dict(torch.load('../models/vnet_model.pth'))
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load and preprocess a new CT scan
new_image_path = '../data/images/new_ct_scan.nii'  # Adjust path as needed
new_image = nib.load(new_image_path).get_fdata()
processed_image = resize_volume(new_image, target_shape=(128, 128, 128))

# Convert to PyTorch tensor
image_tensor = torch.from_numpy(processed_image).float().unsqueeze(0).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    output = model(image_tensor)
    prediction = output.argmax(1).squeeze().cpu().numpy()

# Visualize the result
visualize_3d(processed_image, prediction=prediction)

# Optionally, save the visualization as a video
save_visualization_video(processed_image, label=None, prediction=prediction, filename='3d_segmentation.mp4')

print("Inference complete!")

# Example: Calculate Dice score (if you have ground truth)
# ground_truth_path = '../data/labels/new_ct_scan_label.nii'
# ground_truth = nib.load(ground_truth_path).get_fdata()
# ground_truth_tensor = torch.from_numpy(ground_truth).long().to(device)
# dice = dice_score(torch.from_numpy(prediction).float(), ground_truth_tensor.float())
# print(f'Dice Score: {dice}')
