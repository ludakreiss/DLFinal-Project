# dataset_classify.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class GeoDatasetClassify(Dataset):
    def __init__(self, csv_path, img_dir, country_to_idx, transform=None):
        self.df = pd.read_csv(csv_path)                          # read the CSV using pandas
        self.img_dir = img_dir
        self.transform = transform
        self.country_to_idx = country_to_idx    # the mapping dict, passed in from outside

    def __len__(self):
        return len(self.df)                     # how many rows does the dataset have?

    def __getitem__(self, idx):
        row = self.df.iloc[idx]                               # get the idx-th row from self.df
                                                 # hint: pandas has a method for "get row by position"

        img_path = os.path.join(self.img_dir, row["filename"])   # which CSV column holds the filename?
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        country_str = row["country"]                # which column holds the country name?
        label = self.country_to_idx[country_str]         # convert the string to an integer using the mapping

        return image, torch.tensor(label, dtype=torch.long)