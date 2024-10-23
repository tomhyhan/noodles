import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

SEED = 42

data = pd.read_csv("./pasta_data.csv")
image_paths, labels = data["img_path"], data["label"]

X, test_data, y, test_label = train_test_split(image_paths, labels, train_size=0.9, random_state=SEED, shuffle=True, stratify=labels)

means = []
stds = []
for path in image_paths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.tensor(image, dtype=torch.float32).div(255)
    img = img.permute(2, 0, 1)
    means.append(img.mean(dim=(1, 2)))
    stds.append(img.std(dim=(1, 2)))

means = torch.stack(means)
stds = torch.stack(stds)

mean = means.mean(dim=0)
std = stds.mean(dim=0)

print(mean, std)
# Train Data (RGB)
# mean - tensor([0.6614, 0.5616, 0.4240]) 
# std  - tensor([0.2281, 0.2389, 0.2696])