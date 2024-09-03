# 3D Segmentation Model for CT Abdomen Organs

## Overview
This project involves building a 3D segmentation model to segment abdominal organs (Liver, Right Kidney, Left Kidney, and Spleen) from CT scans using a VNet architecture.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>

2. Install dependencies:
pip install -r requirements.txt


3.Download the dataset and place the images in data/raw/images/ and labels in data/raw/labels/.


4.Train the model:

python src/train.py

5.Run inference and evaluation:

python src/run.py



6. **3D Visualization**

Create a script to visualize the 3D segments if you haven't already. You can use libraries like `matplotlib` or `vtk` to render and save videos. Save it as `visualize.py` in `src/`:


