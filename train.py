import os
import numpy as np  # Make sure to import numpy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from vnet import VNet
from data_processing import preprocess_image

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.nii')]
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = preprocess_image(image_path)

        # Extract the base filename to locate the label file
        base_filename = os.path.basename(image_path).replace('_0000.nii', '').replace('.nii', '')
        label_folder = os.path.join(self.label_dir, base_filename + '.nii')
        label_path = os.path.join(label_folder, base_filename + '.nii')

        # Debug prints
        print(f"Image path: {image_path}")
        print(f"Base filename: {base_filename}")
        print(f"Label folder: {label_folder}")
        print(f"Label path: {label_path}")

        # Ensure label path is correct
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"No such file or no access: '{label_path}'")

        label = preprocess_image(label_path)

        # Convert labels to integers (if they are not already)
        label = np.squeeze(label)  # Remove channel dimension

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)  # Ensure dtype=torch.long for labels

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Debugging prints
            print(f"Inputs shape: {inputs.shape}")
            print(f"Labels shape: {labels.shape}")

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

def main():
    image_dir = '../data/raw/images'
    label_dir = '../data/raw/labels'
    num_classes = 14

    model = VNet(in_channels=1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    dataset = CustomDataset(image_dir=image_dir, label_dir=label_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
