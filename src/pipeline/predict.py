import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.image import imread # Changed to use cv2.imread

# Import the UNetPP network from a custom module
from pipeline.network import UNetPP # Assuming this relative import works based on how engine.py calls it
from argparse import ArgumentParser
from albumentations import Resize, Normalize
from albumentations.core.composition import Compose

# Define an image validation transform using Albumentations library
val_transform = Compose([
    Resize(256, 256),  # Resize images to 256x256
    Normalize(),       # Normalize the image (expects uint8 [0,255] or float32 [0,1] if max_pixel_value=1.0)
])

# Define a function to load and preprocess an image
def image_loader(image_name):
    # Read the image using OpenCV (BGR format, uint8)
    img = cv2.imread(image_name)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_name}")
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply the defined validation transform
    # Normalize() by default expects input [0,255] for uint8 or [0,1] for float32 if max_pixel_value=1.0
    # Since img is uint8 [0,255], default Normalize() is fine.
    img = val_transform(image=img)["image"] # img is now HWC, float32, standardized
    
    # The line `img = img.astype('float32') / 255` is REMOVED.
    
    # Transpose the image dimensions to match the expected PyTorch format (C, H, W)
    img = img.transpose(2, 0, 1)

    return img