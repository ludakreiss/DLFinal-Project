# dataset_hierarchical.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class GeoDatasetHierarchical(Dataset):
    def __init__(self, csv_path, img_dir, country_to_idx, centroids, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.country_to_idx = country_to_idx
        self.centroids = centroids  # DataFrame indexed by country, columns lat/lng

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        country = row["country"]
        country_label = self.country_to_idx[country]

        centroid_lat = self.centroids.loc[country, "lat"]
        centroid_lng = self.centroids.loc[country, "lng"]

        # offset target: true coord minus TRUE country's centroid (teacher forcing)
        delta_lat = row["lat"] - centroid_lat
        delta_lng = row["lng"] - centroid_lng
        offset_target = torch.tensor([delta_lat, delta_lng], dtype=torch.float32)

        return image, torch.tensor(country_label, dtype=torch.long), offset_target