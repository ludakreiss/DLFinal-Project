import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class GeoDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        """
        csv_path: path to a CSV with columns filename, country, iso, lat, lng
        img_dir: folder containing the actual .jpg files
        transform: torchvision transform pipeline (mainly for data augmentaion purposes)
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        # normalize lng and lat
        lat_norm = row["lat"] / 90.0
        lng_norm = row["lng"] / 180.0
        target = torch.tensor([lat_norm, lng_norm], dtype=torch.float32)

        return image, target