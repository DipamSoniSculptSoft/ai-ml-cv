import os, sys
import yaml
import torch
import pandas as pd, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.amp import autocast
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from albumentations import (
    Resize,
    HorizontalFlip,
    VerticalFlip,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
    RandomRotate90
)
from albumentations.core.composition import Compose, OneOf
# 
from pipeline.utils import AverageMeter, iou_score
from pipeline.network import UNetPP
from pipeline.dataset import DataSet
from pipeline.train import train
from pipeline.validate import validate
from pipeline.predict import image_loader
from pipeline.constants import *


# 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)




def main():
    # Load configuration from a YAML file located in SCRIPT_DIR
    config_file_path = os.path.join(SCRIPT_DIR, "config.yaml")
    config:dict = yaml.load(open(config_file_path), Loader=yaml.SafeLoader)

    # Helper function to resolve paths from config relative to SCRIPT_DIR
    def resolve_config_path(config_path_value):
        return os.path.abspath(os.path.join(SCRIPT_DIR, config_path_value))

    # Extract and resolve configuration values
    extn = config["extn"]
    epochs = config["epochs"]
    log_path = resolve_config_path(config["log_path"])
    mask_path = resolve_config_path(config["mask_path"])
    image_path = resolve_config_path(config["image_path"])
    model_path = resolve_config_path(config["model_path"])
    
    
    # Ensure output directories exist for logs and models
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"""{'-'*75}
Batch Size: {config.get("batch_size", 4)}
Total Workers: {config.get("num_workers", 8)}
Device: {DEVICE_STR}
{'-'*75}""")

    # Create an ordered dictionary to store training logs
    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0

    torch.backends.cudnn.benchmark = True # cudNN optimzations    

    # Find image files with the specified extension
    extn_ = f"*{extn}"
    # image_path is now an absolute path
    img_files_found = glob(os.path.join(image_path, extn_))

    if not img_files_found:
        print(f"Error: No image files found in {image_path} with extension {extn_}")
        print(f"Please check the 'image_path' in config.yaml and the directory structure.")
        sys.exit(1)

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_files_found]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=42) # Added random_state for reproducibility

    # Define data augmentation transformations for training and validation
    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        # RandomRotate90(p=0.5),
        # HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.5),
        OneOf([
            HueSaturationValue(),
            RandomBrightnessContrast(),
        ], p=1),
        Resize(256, 256),
        Normalize(),
    ])

    val_transform_engine = Compose([ # Renamed to avoid conflict if predict.val_transform was imported
        Resize(256, 256),
        Normalize(),
    ])


    # Create training and validation datasets
    train_dataset = DataSet(
        img_ids=train_img_ids,
        img_dir=image_path, # Use resolved absolute path
        mask_dir=mask_path, # Use resolved absolute path
        img_ext=extn,
        mask_ext=extn,
        transform=train_transform)

    val_dataset = DataSet(
        img_ids=val_img_ids,
        img_dir=image_path, # Use resolved absolute path
        mask_dir=mask_path, # Use resolved absolute path
        img_ext=extn,
        mask_ext=extn,
        transform=val_transform_engine)

    # Create data loaders for training and validation
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        drop_last=True,
        num_workers=config.get("num_workers", 8), 
        pin_memory=True
        )


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        drop_last=False,
        num_workers=config.get("num_workers", 8), # Get num_workers from config or default
        pin_memory=True
        )

    # Create the UNet++ model
    model = UNetPP(num_classes=1, input_channels=3, deep_supervision=True) # Ensure num_classes is 1 for binary
    model.to(DEVICE)

    # Loss function (Binary Cross Entropy with Logits)
    criterion = nn.BCEWithLogitsLoss()

    # Define the optimizer (Adam)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=config.get("learning_rate", 1e-3), weight_decay=config.get("weight_decay", 1e-4))
    scaler = GradScaler(device=DEVICE) # For Mixed Precision

    #* Training loop #################################################
    for epoch in range(epochs):
        print(f'Epoch [{epoch}/{epochs}]')
        
        #* TRAIN
        train_log : OrderedDict = train(deep_sup=True,
                                        train_loader=train_loader,
                                        model=model,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        scaler=scaler)
        
        
        
        #* VALIDATE [@ every 5 epochs]
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_log : OrderedDict = validate(deep_sup=True, 
                                             val_loader=val_loader, 
                                             model=model, 
                                             criterion=criterion)
            
            print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                  % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        else:
            val_log = OrderedDict([('loss', float('nan')), ('iou', float('nan'))])
        

        # Log the training and validation metrics
        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Save the log to a CSV file
            pd.DataFrame(log).to_csv(log_path, index=False)

        trigger += 1

        # Save the model if the validation IoU is improved
        if val_log is not None and val_log['iou'] > best_iou:
            torch.save(model.state_dict(), model_path)
            best_iou = val_log['iou']
            print(f"=> saved best model to {model_path} with val_iou: {best_iou:.4f}")
            trigger = 0

        # Early stopping (optional)
        # if trigger >= config.get("early_stopping_patience", 10): # Example: stop if no improvement for 10 epochs
        #     print("Early stopping triggered.")
        #     break

    print("Training finished.")
    print(f"Best model saved at: {model_path}")
    print(f"Logs saved at: {log_path}")
    print('-'*75)

    # Prediction part (can be run optionally or as a separate script)
    print("\n--- Running Prediction ---")

    # Resolve default test image path relative to SCRIPT_DIR
    default_test_img_path_relative = "../input/PNG/Original/50.png" # As in original ArgumentParser
    default_test_img_path_absolute = os.path.abspath(os.path.join(SCRIPT_DIR, default_test_img_path_relative))

    parser = ArgumentParser()
    parser.add_argument("--test_img", default=default_test_img_path_absolute, help="path to test image")
    opt = parser.parse_args()

    # im_width and im_height are from config for resizing output, not input to model
    im_width = config["im_width"]
    im_height = config["im_height"]
    # model_path for prediction is the one saved during training (already resolved)
    # output_path for prediction image
    output_pred_path = resolve_config_path(config["output_path"])
    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)


    # Load the trained model (best one)
    pred_model = UNetPP(num_classes=1, input_channels=3, deep_supervision=True)
    # Load state dict, ensuring map_location for CPU if model was trained on GPU and predicting on CPU
    if not torch.cuda.is_available():
        pred_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        pred_model.load_state_dict(torch.load(model_path))

    pred_model.to(DEVICE)
    pred_model.eval()

    # Load and preprocess the test image
    if not os.path.exists(opt.test_img):
        print(f"Error: Test image not found at {opt.test_img}")
        sys.exit(1)

    print(f"Loading test image from: {opt.test_img}")
    image = image_loader(opt.test_img) # image_loader from ML_Pipeline.predict
    image_tensor = torch.from_numpy(image).unsqueeze(0) # Add batch dimension
    image_tensor = image_tensor.to(DEVICE)

    # Perform inference
    with torch.no_grad():
        mask_outputs = pred_model(image_tensor)

    # If deep_supervision is True, model(image) returns a list of masks.
    # The last one is typically the most refined.
    output_mask = mask_outputs[-1] if isinstance(mask_outputs, list) else mask_outputs

    output_mask = output_mask.squeeze().cpu().numpy() # Remove batch and channel (if 1), move to CPU, convert to numpy

    # Threshold the mask and resize it
    # The threshold value (-2.5) might need adjustment depending on model output range (logits)
    # A more common approach after sigmoid (if not included in loss) is 0.5
    # Since BCEWithLogitsLoss is used, output_mask is logits. Apply sigmoid then threshold.
    output_mask_sigmoid = 1 / (1 + np.exp(-output_mask)) # Apply sigmoid manually
    binary_mask = (output_mask_sigmoid > 0.5).astype(np.uint8) * 255 # Threshold at 0.5

    # Resize to original dimensions specified in config (im_width, im_height)
    resized_mask = cv2.resize(binary_mask, (im_width, im_height), interpolation=cv2.INTER_NEAREST)

    # Save the output mask as an image
    plt.imsave(output_pred_path, resized_mask, cmap="gray")
    print(f"Prediction mask saved to: {output_pred_path}")

    # Optionally display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    original_test_img_display = cv2.imread(opt.test_img)
    plt.imshow(cv2.cvtColor(original_test_img_display, cv2.COLOR_BGR2RGB))
    plt.title("Original Test Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(resized_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")
    plt.show()
    
    
if __name__ == '__main__':
    main()