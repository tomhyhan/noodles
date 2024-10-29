import albumentations  as A
from albumentations.pytorch import ToTensorV2 
from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from torchvision.models import swin_s, Swin_S_Weights
from config.config_manager import ConfigManager
from collections import Counter
from model.data_model import PastaData
from model.train import trainer
from model.utils import reset_seed
from model.data import CLASS_ENCODER
from model.viz import class_imblance

SEED = 42

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

data = pd.read_csv("./pasta_data.csv")
image_paths, labels = data["img_path"], data["label"]

X, test_data, y, test_label = train_test_split(image_paths.values, labels.values, train_size=0.9, random_state=SEED, shuffle=True, stratify=labels)

# class_imblance(y)


print(type(X), type(y))

N=20
perm_indices = np.random.permutation(N)

X = X[perm_indices]
y = y[perm_indices]

print(y[:3])

train_transform_list = create_train_transforms()
img = PastaData(image_paths=X, labels=y, transform_list=train_transform_list)

print(len(img))

cm = ConfigManager("./config/config.yml")
num_epochs = cm.config.swin.train_args.num_epochs
batch_size = cm.config.swin.train_args.batch_size
lr=cm.config.swin.train_args.lr

loader = DataLoader(img, batch_size=batch_size, shuffle=False, drop_last=True)

model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, 16)

print(len(loader))

# StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
for (train_i, val_i) in k_fold.split(X,y):
    print(len(train_i), len(val_i))
    trainer(
        model, 
        loader, 
        None,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=0.01,
    )
