import albumentations  as A
import cv2
from torch.utils.data import DataLoader, Dataset
import numpy
import matplotlib.pyplot as plt

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    # A.CenterCrop(224,224),
    # A.Resize(224,244)
])

class SingleDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.transform = transform  
              
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        if self.transform:
            print("transform")
            return self.transform(image=self.image)["image"]
        print("asdf")
img = SingleDataset("./Penne_2.jpg", transform=transform)

loader = DataLoader(img, batch_size=1, shuffle=False)

for batch in loader:
    print(len(batch))
    augmented_image = batch[0].numpy()  # Convert to numpy array
    
    # Display original and augmented images side by side
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img.image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image)
    plt.title('Augmented Image')
    plt.axis('off')
    
    plt.show()
    
# import albumentations as A
# import cv2
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt

# # Create a custom dataset class
# class SingleImageDataset(Dataset):
#     def __init__(self, image_path, transform=None):
#         self.image = cv2.imread(image_path)
#         self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
#         self.transform = transform
    
#     def __len__(self):
#         return 1  # Since we only have one image
    
#     def __getitem__(self, idx):
#         if self.transform:
#             augmented = self.transform(image=self.image)
#             image = augmented['image']
#         else:
#             image = self.image
#         return image

# # Define transformations
# transform = A.Compose([
#     A.RandomCrop(width=256, height=256),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
# ])

# # Create dataset and dataloader
# dataset = SingleImageDataset("./Penne_0.jpg", transform=transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# # Get and display the augmented image
# for batch in dataloader:
#     augmented_image = batch[0].numpy()  # Convert to numpy array
    
#     # Display original and augmented images side by side
#     plt.figure(figsize=(10, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(dataset.image)
#     plt.title('Original Image')
#     plt.axis('off')
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(augmented_image)
#     plt.title('Augmented Image')
#     plt.axis('off')
    
#     plt.show()
#     break  # Since we only want to show one augmented version