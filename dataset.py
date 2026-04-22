import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from config import ANN_INPUT_SIZE

class GrazDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_dict_path, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # Filter out hidden Mac files
        self.images = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
        
        # --- FIX: Build Color-to-Class Mapping from CSV ---
        self.color_to_id = {}
        with open(class_dict_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]): # Skip header row
                parts = line.strip().split(',')
                # Format: [name, r, g, b]
                r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
                self.color_to_id[(r, g, b)] = idx
                
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Resize(ANN_INPUT_SIZE),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def rgb_to_class(self, mask_rgb):
        """Converts the colorful RGB mask into a 2D array of Class IDs (0-22)"""
        mask_class = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.int64)
        for (r, g, b), class_id in self.color_to_id.items():
            matches = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
            mask_class[matches] = class_id
        return mask_class

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # Load RGB image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- FIX: Load mask as RGB and map to Class IDs ---
        mask_rgb = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask_class = self.rgb_to_class(mask_rgb)
        
        image = self.img_transform(image)
        mask_tensor = torch.from_numpy(mask_class).long()
        mask_tensor = T.Resize(ANN_INPUT_SIZE, interpolation=T.InterpolationMode.NEAREST)(mask_tensor.unsqueeze(0)).squeeze(0)
        
        return image, mask_tensor