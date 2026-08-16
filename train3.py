# train_hierarchical.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset_2 import GeoDatasetHierarchical
from newmodel import GeoCNNHierarchical
from utils import haversine_km, count_parameters
from transform_config import transform_none, transform_val

BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
CLASS_LOSS_WEIGHT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("geo_dataset/train_split.csv")
countries = sorted(train_df["country"].unique())
country_to_idx = {c: i for i, c in enumerate(countries)}
idx_to_country = {i: c for c, i in country_to_idx.items()}

centroids = pd.read_csv("geo_dataset/country_centroids.csv").set_index("country")

train_ds = GeoDatasetHierarchical("geo_dataset/train_split.csv", "geo_dataset/train", country_to_idx, centroids, transform=transform_none)
val_ds = GeoDatasetHierarchical("geo_dataset/val_split.csv", "geo_dataset/train", country_to_idx, centroids, transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = GeoCNNHierarchical(num_countries=len(countries)).to(DEVICE)
print("Trainable parameters:", count_parameters(model))

classify_criterion = nn.CrossEntropyLoss()
offset_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_haversine = float("inf")
history = {"train_loss": [], "val_acc": [], "val_median_km": [], "val_mean_km": []}

val_df = pd.read_csv("geo_dataset/val_split.csv")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, country_labels, offset_targets in train_loader:
        images = images.to(DEVICE)
        country_labels = country_labels.to(DEVICE)
        offset_targets = offset_targets.to(DEVICE)

        optimizer.zero_grad()
        class_logits, offset_preds = model(images)

        loss_class = classify_criterion(class_logits, country_labels)
        loss_offset = offset_criterion(offset_preds, offset_targets)
        loss = CLASS_LOSS_WEIGHT * loss_class + loss_offset

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_ds)

    # ---- validate: use PREDICTED country's centroid + predicted offset ----
    model.eval()
    correct = 0
    all_dists = []
    with torch.no_grad():
        for i, (images, country_labels, offset_targets) in enumerate(val_loader):
            images = images.to(DEVICE)
            class_logits, offset_preds = model(images)

            preds_class = class_logits.argmax(dim=1).cpu().numpy()
            offset_preds_np = offset_preds.cpu().numpy()
            correct += (preds_class == country_labels.numpy()).sum()

            batch_start = i * BATCH_SIZE
            for j, pred_country_idx in enumerate(preds_class):
                pred_country = idx_to_country[pred_country_idx]
                centroid_lat = centroids.loc[pred_country, "lat"]
                centroid_lng = centroids.loc[pred_country, "lng"]

                final_lat = centroid_lat + offset_preds_np[j, 0]
                final_lng = centroid_lng + offset_preds_np[j, 1]

                true_row = val_df.iloc[batch_start + j]
                true_lat, true_lng = true_row["lat"], true_row["lng"]

                dist = haversine_km(final_lat, final_lng, true_lat, true_lng)
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
        torch.save(model.state_dict(), "checkpoints/best_model_hierarchical.pt")
        print(f"  -> saved new best hierarchical model (median={median_haversine:.1f} km)")

print("Training complete. Best val median haversine (hierarchical):", best_val_haversine)

epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(epochs_range, history["train_loss"], marker="o", color="tab:blue")
axes[0].set_title("Training Loss (weighted sum)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["val_acc"], marker="o", color="tab:purple")
axes[1].set_title("Validation Country Accuracy")
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, history["val_median_km"], marker="o", label="Median", color="tab:green")
axes[2].plot(epochs_range, history["val_mean_km"], marker="o", label="Mean", color="tab:orange")
axes[2].set_title("Validation Haversine (km)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("checkpoints/training_curves_hierarchical.png", dpi=150)
print("Saved training curves to checkpoints/training_curves_hierarchical.png")