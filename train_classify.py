# train_classify.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset_classify import GeoDatasetClassify
from model import GeoCNN
from utils import haversine_km, count_parameters
from transforms_config import transform_none, transform_val

BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- build country <-> index mapping ----
train_df = pd.read_csv("geo_dataset/train_split.csv")
countries = sorted(train_df["country"].unique())
country_to_idx = {c: i for i, c in enumerate(countries)}
idx_to_country = {i: c for c, i in country_to_idx.items()}
print("Countries:", country_to_idx)

# ---- load centroids for haversine eval ----
centroids = pd.read_csv("geo_dataset/country_centroids.csv").set_index("country")

# ---- data ----
train_ds = GeoDatasetClassify("geo_dataset/train_split.csv", "geo_dataset/train", country_to_idx, transform=transform_none)
val_ds = GeoDatasetClassify("geo_dataset/val_split.csv", "geo_dataset/train", country_to_idx, transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---- model: same backbone, output_dim = num countries ----
model = GeoCNN(output_dim=len(countries)).to(DEVICE)
print("Trainable parameters:", count_parameters(model))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_haversine = float("inf")
history = {"train_loss": [], "val_acc": [], "val_median_km": [], "val_mean_km": []}

# ---- need true lat/lng for val haversine, not just country label ----
val_df = pd.read_csv("geo_dataset/val_split.csv")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_ds)

    # ---- validate ----
    model.eval()
    correct = 0
    all_dists = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()

            correct += (preds == labels_np).sum()

            # map predicted country index -> centroid coords
            batch_start = i * BATCH_SIZE
            for j, p in enumerate(preds):
                pred_country = idx_to_country[p]
                pred_lat = centroids.loc[pred_country, "lat"]
                pred_lng = centroids.loc[pred_country, "lng"]

                true_row = val_df.iloc[batch_start + j]
                true_lat, true_lng = true_row["lat"], true_row["lng"]

                dist = haversine_km(pred_lat, pred_lng, true_lat, true_lng)
                all_dists.append(dist)

    val_acc = correct / len(val_ds)
    median_haversine = np.median(all_dists)
    mean_haversine = np.mean(all_dists)

    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    history["val_median_km"].append(median_haversine)
    history["val_mean_km"].append(mean_haversine)

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | "
          f"val_acc={val_acc:.3f} | val_median_km={median_haversine:.1f} | val_mean_km={mean_haversine:.1f}")

    if median_haversine < best_val_haversine:
        best_val_haversine = median_haversine
        torch.save(model.state_dict(), "checkpoints/best_model_classify.pt")
        print(f"  -> saved new best classify model (median={median_haversine:.1f} km)")

print("Training complete. Best val median haversine (classify):", best_val_haversine)

epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(epochs_range, history["train_loss"], marker="o", color="tab:blue")
axes[0].set_title("Training Loss (CrossEntropy)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["val_acc"], marker="o", color="tab:purple")
axes[1].set_title("Validation Country Accuracy")
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, history["val_median_km"], marker="o", label="Median", color="tab:green")
axes[2].plot(epochs_range, history["val_mean_km"], marker="o", label="Mean", color="tab:orange")
axes[2].set_title("Validation Haversine (km) via Centroid")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("checkpoints/training_curves_classify.png", dpi=150)
print("Saved training curves to checkpoints/training_curves_classify.png")