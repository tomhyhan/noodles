import albumentations  as A
from albumentations.pytorch import ToTensorV2 
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from model.data_model import PastaData
from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 42

def create_transforms(image_size=224):
    # calculated seperately
    mean = [0.6614, 0.5616, 0.4240]
    std  = [0.2281, 0.2389, 0.2696]

    transform_list = [
        # Transform 1: Basic resize and normalize
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
           ToTensorV2()
        ]),
        
        # Transform 2: Center crop
        A.Compose([
            A.Resize(int(image_size * 1.2), int(image_size * 1.2)),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]),
        
        # Transform 3: Random crop with color augmentation
        A.Compose([
            A.RandomResizedCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]),
        
        # Transform 4: Strong augmentation
        # A.Compose([
        #     A.RandomResizedCrop(image_size, image_size),
        #     A.HorizontalFlip(p=0.5),
        #     A.OneOf([
        #         A.RandomBrightnessContrast(),
        #         A.ColorJitter(),
        #     ], p=0.3),
        #     A.OneOf([
        #         A.GaussNoise(),
        #         A.GaussianBlur(),
        #     ], p=0.2),
        #     A.Normalize(mean=mean, std=std),
        #     ToTensorV2
        # ])
    ]
    return transform_list

data = pd.read_csv("./pasta_data.csv")
image_paths, labels = data["img_path"], data["label"]

X, test_data, y, test_label = train_test_split(image_paths.values, labels.values, train_size=0.9, random_state=SEED, shuffle=True, stratify=labels)

N=50
perm_indices = np.random.permutation(N)

X = X[perm_indices]
y = y[perm_indices]


transform_list = create_transforms()
img = PastaData(image_paths=X, labels=y, transform_list=transform_list)

print(len(img))

loader = DataLoader(img, batch_size=32, shuffle=False, drop_last=True)
print(len(loader))
