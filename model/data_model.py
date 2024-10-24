from torch.utils.data import Dataset
import cv2

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
    
