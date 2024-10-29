from torch.utils.data import Dataset
import cv2
import albumentations  as A
from albumentations.pytorch import ToTensorV2 

class PastaData(Dataset):
    def __init__(self, image_paths, labels, transform_list=None):
        if not transform_list:
            raise ValueError("Transform_list must be provided")
        self.image_paths = image_paths
        self.labels = labels
        self.transform_list = transform_list
    
    def __len__(self):
        if self.transform_list:
            return len(self.image_paths) * len(self.transform_list) 
    
    def __getitem__(self, idx):
        original_idx = idx // len(self.transform_list)
        transform_idx = idx % len(self.transform_list)
        
        image_path = self.image_paths[original_idx]
        label = self.labels[original_idx]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transform_image = self.transform_list[transform_idx](image=image)["image"]
        
        return transform_image, label

def create_train_transforms(image_size=224):
    # calculated seperately
    mean = [0.6614, 0.5616, 0.4240]
    std  = [0.2281, 0.2389, 0.2696]

    transform_list = [
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
           ToTensorV2()
        ]),
        
        A.Compose([
            A.Resize(int(image_size * 1.2), int(image_size * 1.2)),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]),
        
        A.Compose([
            A.Resize(int(image_size * 1.2), int(image_size * 1.2)),
            A.RandomCrop(image_size, image_size),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5, 
                    sat_shift_limit=20,
                    val_shift_limit=20
                ),
            ], p=0.7),  
            A.GaussianBlur(blur_limit=(3, 5), p=0.3), 
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]),
    ]
    return transform_list

def create_test_transforms(image_size=224):
    # calculated seperately
    mean = [0.6614, 0.5616, 0.4240]
    std  = [0.2281, 0.2389, 0.2696]

    transform_list = [
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
           ToTensorV2()
        ]),
    ]
    return transform_list


