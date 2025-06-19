import os,sys, yaml
import cv2
from glob import glob
import torch
from tqdm import tqdm
import numpy as np
from albumentations import Resize, Normalize
from albumentations.core.composition import Compose
import matplotlib.pyplot as plt

# 
from pipeline.network import UNetPP
from pipeline.predict import image_loader
from pipeline.constants import *


# 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'predictions')
os.makedirs(OUTPUT_DIR, exist_ok=True)

config_file_path = os.path.join(SCRIPT_DIR, "config.yaml")
config:dict = yaml.load(open(config_file_path), Loader=yaml.SafeLoader)

# HELPER FUNC:
def resolve_config_path(config_path_value):
        return os.path.abspath(os.path.join(SCRIPT_DIR, config_path_value))

extn = config["extn"]
mask_path = resolve_config_path(config["mask_path"])

INPUT_IMAGE_DIR = resolve_config_path(config["image_path"])
IMAGES_LIST = [f'{i}.png' for i in range(50,52) ]
MODEL_PATH = resolve_config_path(config["model_path"])


# Load the Trained Model:
def load_model(model_path:str = MODEL_PATH) -> UNetPP:
    """Load the trained model.
    
    :param model_path(str): path to the model
    
    Returns:
        UNetPP
    """
    
    model = UNetPP(num_classes=1,
                   input_channels=3,
                   deep_supervision=True)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE_STR))
    model.to(DEVICE)
    model.eval()
    return model


# Make Predictions:::::
def predict_image(model, image_path):
    """Generate the Predicted Mask"""

    image = image_loader(image_path)
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)
    
    with torch.no_grad():
        mask_outputs = model(image_tensor)
        
    output_mask = mask_outputs[-1] if isinstance(mask_outputs, list) else mask_outputs
    output_mask = output_mask.squeeze().cpu().numpy() # remove batch channels & dims
    
    # POST-PROCESS: Apply sigmoid and threshold to get binary mask.
    output_mask_sigmoid = 1 / (1 + np.exp(-output_mask))          # Sigmoid Apply
    binary_mask = (output_mask_sigmoid >0.5).astype(np.uint8) * 255    # Threshold: 0.5
    
    resized_mask = cv2.resize(binary_mask, 
                              (config["im_width"], config["im_height"]),
                              interpolation=cv2.INTER_NEAREST)
    return resized_mask
    
    
def save_prediction(mask, output_path):
    plt.imsave(output_path,
               mask,
               cmap='gray')
    

def main(first_n_images:int = 10):
    
    model = load_model()
    
    # image_paths = glob(os.path.join(INPUT_IMAGE_DIR, '*.png'))[:first_n_images]
    image_paths = [os.path.join(INPUT_IMAGE_DIR, image_path) for image_path in IMAGES_LIST]
    
    # print(image_paths)
    if not image_paths:
        print(f"No images found in {INPUT_IMAGE_DIR}")
        return
    
    pbar = tqdm(len(image_paths))
    # Each Image Processing:
    for image_path in image_paths:
        
        predicted_mask = predict_image(model, image_path)
        image_name = os.path.basename(image_path)
        output_path = os.path.join(OUTPUT_DIR, image_name)

        # Save predicted image
        save_prediction(mask=predicted_mask,
                        output_path=output_path)

        # print(f"Saved Prediction for {image_name} to {output_path}")
        pbar.update(1)
    pbar.close()
    

if __name__ == "__main__":
    
    main(first_n_images=10)