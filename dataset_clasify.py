# dataset_classify.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class GeoDatasetClassify(Dataset):
    def __init__(self, csv_path, img_dir, country_to_idx, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.country_to_idx = country_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.country_to_idx[row["country"]]
        return image, torch.tensor(label, dtype=torch.long)