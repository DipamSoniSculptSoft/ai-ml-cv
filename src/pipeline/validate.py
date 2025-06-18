import os
import yaml
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from pipeline.utils import AverageMeter, iou_score
from albumentations import Resize
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf
# from albumentations.augmentations.transforms import RandomRotate90
from pipeline.network import UNetPP, VGGBlock
from pipeline.dataset import DataSet
from pipeline.train import train
from pipeline.constants import *

# Define a validation function
def validate(deep_sup:bool, 
             val_loader:torch.utils.data.DataLoader, 
             model:UNetPP, 
             criterion: nn.BCEWithLogitsLoss) -> OrderedDict:
    
    """
    Validation pipeline.
    Evaluating model on test/validation data.
    
    :param deep_sup(bool):
    :param val_loader(torch.utils.data.DataLoader): Validation Data Loader 
    :param model(UNetPP): UNet++ defined model
    :parm criterion(nn.BCEWithLogitsLoss): Loss criterion
    """
    
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))  # progress bar
        for input, target, _ in val_loader:
            input, target = input.to(DEVICE), target.to(DEVICE)
            
            with torch.amp.autocast(device_type=DEVICE_STR):
                
                if deep_sup:
                    outputs = model(input)
                    loss = sum(criterion(out, target) for out in outputs) / len(outputs)
                    iou = iou_score(outputs[-1], target)
                else: # standard single-output case
                    output = model(input)
                    loss = criterion(output, target)
                    iou = iou_score(output, target)

            # Update the average meters with loss and IoU values
            avg_meters['loss'].update(loss.item() if isinstance(loss, torch.Tensor) else loss, input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            # Create a dictionary of values to display in the progress bar
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()  

    # Return a dictionary of average loss and IoU values
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
