# train_classify.py
from ctypes.wintypes import RGB
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataset19 import GeoDatasetClassify
from utils import haversine_km, count_parameters
from model19 import GeoClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3

# ---- build country_to_idx mapping (you already wrote this logic) ----
train_df = pd.read_csv("geo_dataset/train_split.csv")
centroids = pd.read_csv("geo_dataset/country_centroids.csv").set_index("country")
val_df = pd.read_csv("geo_dataset/val_split.csv")
countries = sorted(train_df["country"].unique())   
country_to_idx = {country: idx for idx, country in enumerate(countries)}   
idx_to_country = {idx: country for country, idx in country_to_idx.items()}  

# ---- transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- datasets and dataloaders ----
train_ds = GeoDatasetClassify(csv_path="geo_dataset/train_split.csv", img_dir="geo_dataset/train", country_to_idx=country_to_idx, transform=transform)
val_ds = GeoDatasetClassify(csv_path="geo_dataset/val_split.csv", img_dir="geo_dataset/train", country_to_idx=country_to_idx, transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=5)   # should training data be shuffled?
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,  num_workers=5)       # should validation data be shuffled?

# ---- model, loss, optimizer ----
model = GeoClassifier(num_countries=len(countries)).to(DEVICE)
print("Trainable parameters:", count_parameters(model))
criterion = nn.CrossEntropyLoss()          # which PyTorch loss function is standard for multi-class classification?
optimizer = torch.optim.Adam(model.parameters(), lr=LR)         # torch.optim has an optimizer class you've used before


# add below your setup code from Stage 1

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}")

    train_loss /= len(train_ds)
    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f}")

    # ---- validation half ----
    model.eval()
    correct = 0
    all_dists = []

    with torch.no_grad():   # which context manager turns off gradient tracking?
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)   # which dimension has the 12 country scores?
            preds = preds.cpu().numpy()
            labels_np = labels.numpy()

            correct += (preds == labels_np).sum()   # compare predictions to true labels

            batch_start = i * BATCH_SIZE
            for j, pred_idx in enumerate(preds):
                pred_country = idx_to_country[pred_idx]     # convert index -> country name

                pred_lat = centroids.loc[pred_country, "lat"]
                pred_lng = centroids.loc[pred_country, "lng"]

                true_row = val_df.iloc[batch_start+j]              # which row? (hint: batch_start + j)
                true_lat, true_lng = true_row["lat"], true_row["lng"]

                dist = haversine_km(pred_lat, pred_lng, true_lat, true_lng)  # what order do haversine's 4 args go in?
                all_dists.append(dist)

    val_acc = correct / len(val_ds)
    median_haversine = np.median(all_dists)   # numpy function for median
    print(f"          val_acc={val_acc:.3f} | val_median_km={median_haversine:.1f}")