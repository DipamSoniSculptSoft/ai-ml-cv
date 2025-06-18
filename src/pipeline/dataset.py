import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class DataSet(Dataset):
    
    def __init__(self,
                 img_ids, 
                 img_dir, mask_dir,
                 img_ext, mask_ext,
                 transform=None
                 ):
        
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.images = {}
        self.masks = {}
        
        #!
        for img_id in tqdm(img_ids, desc="Preloading dataset"):
            img_path = os.path.join(img_dir, img_id + img_ext)
            mask_path = os.path.join(mask_dir, img_id + mask_ext)

            self.images[img_id] = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            self.masks[img_id] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[...,None]
        #!
        
    def __len__(self):
        return len(self.img_ids)

    # def __getitem__(self, idx):
    #     img_id = self.img_ids[idx]
        
    #     img = cv2.imread(os.path.join(self.img_dir, img_id+self.img_ext))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     mask = []
    #     mask.append(cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
    #     mask = np.dstack(mask)

    #     # Apply data augmentation and preprocessing transformations
    #     if self.transform is not None:
    #         augmented = self.transform(image=img, mask=mask) # transform includes Normalize for img
    #         img = augmented['image']    # img is now HWC, float32, standardized
    #         mask = augmented['mask']

    #     # Transpose the image and mask data for compatibility with PyTorch
    #     # The line `img = img.astype('float32') / 255` is REMOVED as `augmented['image']` is already processed by Normalize()
    #     img = img.transpose(2, 0, 1)  # Channels-first format

    #     # For mask, scale to [0,1] if not already handled by transform
    #     mask = mask.astype('float32') / 255 # Assuming mask needs to be [0,1]
    #     mask = mask.transpose(2, 0, 1)  # Channels-first format

    #     # Return the image, mask, and associated metadata
    #     return img, mask, {'img_id': img_id}
    
    def __getitem__(self, idx):
        
        img_id = self.img_ids[idx]
        img = self.images[img_id].copy()
        mask = self.masks[img_id].copy()
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        
        img = img.transpose(2,0,1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2,0,1) if mask.ndim == 3 else mask[None, :,:]
        return img, mask, {'img_id':img_id}
