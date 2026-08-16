# confusion_check.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from dataset_hierarchical import GeoDatasetHierarchical
from model import GeoCNNHierarchical
from transforms_config import transform_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("geo_dataset/train_split.csv")
countries = sorted(train_df["country"].unique())
country_to_idx = {c: i for i, c in enumerate(countries)}
idx_to_country = {i: c for c, i in country_to_idx.items()}

centroids = pd.read_csv("geo_dataset/country_centroids.csv").set_index("country")
with open("geo_dataset/offset_stats.txt") as f:
    offset_std_lat, offset_std_lng = map(float, f.read().split(","))

val_ds = GeoDatasetHierarchical("geo_dataset/val_split.csv", "geo_dataset/train", country_to_idx, centroids, (offset_std_lat, offset_std_lng), transform=transform_val)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = GeoCNNHierarchical(num_countries=len(countries)).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best_model_hierarchical.pt", map_location=DEVICE))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for images, country_labels, _ in val_loader:
        images = images.to(DEVICE)
        class_logits, _ = model(images)
        preds = class_logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(country_labels.numpy())

cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(countries)), countries, rotation=90)
plt.yticks(range(len(countries)), countries)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Country Confusion Matrix")
plt.colorbar()
for i in range(len(countries)):
    for j in range(len(countries)):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
plt.tight_layout()
plt.savefig("checkpoints/confusion_matrix.png", dpi=150)
print("Saved confusion matrix to checkpoints/confusion_matrix.png")
print("Per-country accuracy:")
for i, c in enumerate(countries):
    acc = cm[i, i] / cm[i].sum()
    print(f"  {c}: {acc:.2%}")