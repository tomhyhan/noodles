import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
        # Calculate dataset statistics BEFORE transforms
        means = []
        stds = []
        
        for path in image_paths:
            img = np.array(Image.open(path)).astype(np.float32)
            # Convert to channel-first format (C, H, W)
            img = img.transpose(2, 0, 1)
            # Calculate statistics per channel
            means.append(img.mean(axis=(1, 2)))
            stds.append(img.std(axis=(1, 2)))
            
        # Average across all images
        self.means = np.mean(means, axis=0)  # [channel1_mean, channel2_mean, channel3_mean]
        self.stds = np.mean(stds, axis=0)    # [channel1_std, channel2_std, channel3_std]
        
        # Create normalization transform
        self.normalize = transforms.Normalize(mean=self.means, std=self.stds)
        
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        
        if self.transform:
            image = self.transform(image)
            
        # Apply normalization after other transforms
        image = self.normalize(image)
        return image

# Example usage
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Scales pixels to [0,1]
])

dataset = ImageDataset(
    image_paths=['path/to/images'],
    transform=transform
)