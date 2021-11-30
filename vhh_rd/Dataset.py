import cv2
import os
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, paths, transform=None):
        """
        Args:
            paths: a list of paths to the images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = cv2.imread(path)
        img = self.transform(img)
        
        # Get image filename without extension  
        img_name = os.path.split(path)[-1].replace(".png", "")
        return {"img": img, "img_name": img_name}
