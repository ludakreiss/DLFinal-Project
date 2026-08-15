# train_sphere.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from dataset_sphere import GeoDatasetSphere
from model import GeoCNN
from utils import haversine_km, count_parameters, xyz_to_latlng

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = GeoDatasetSphere("geo_dataset/train_split2.csv", "geo_dataset/train", transform=transform)
val_ds = GeoDatasetSphere("geo_dataset/val_split2.csv", "geo_dataset/train", transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = GeoCNN(output_dim=3).to(DEVICE)
print("Trainable parameters:", count_parameters(model))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def cosine_loss(pred, target):
    # pred, target: [batch, 3], target already unit length
    pred_norm = F.normalize(pred, dim=1)
    cos_sim = (pred_norm * target).sum(dim=1)  # dot product of unit vectors
    return (1 - cos_sim).mean()

best_val_haversine = float("inf")
history = {"train_loss": [], "val_median_km": [], "val_mean_km": []}

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        preds = model(images)
        loss = cosine_loss(preds, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_ds)

    model.eval()
    all_dists = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            preds = model(images)
            preds_norm = F.normalize(preds, dim=1).cpu().numpy()
            targets_np = targets.numpy()

            pred_lat, pred_lng = xyz_to_latlng(preds_norm[:, 0], preds_norm[:, 1], preds_norm[:, 2])
            true_lat, true_lng = xyz_to_latlng(targets_np[:, 0], targets_np[:, 1], targets_np[:, 2])

            dists = haversine_km(pred_lat, pred_lng, true_lat, true_lng)
            all_dists.extend(dists)

    median_haversine = np.median(all_dists)
    mean_haversine = np.mean(all_dists)

    history["train_loss"].append(train_loss)
    history["val_median_km"].append(median_haversine)
    history["val_mean_km"].append(mean_haversine)

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | "
          f"val_median_km={median_haversine:.1f} | val_mean_km={mean_haversine:.1f}")

    if median_haversine < best_val_haversine:
        best_val_haversine = median_haversine
        torch.save(model.state_dict(), r"checkpoints/unit sphere regression/best_model_sphere.pt")
        print(f"  -> saved new best sphere model (median={median_haversine:.1f} km)")

print("Training complete. Best val median haversine (sphere):", best_val_haversine)

epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(epochs_range, history["train_loss"], marker="o", color="tab:blue")
axes[0].set_title("Training Loss (1 - cosine similarity)")
axes[0].set_xlabel("Epoch")
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["val_median_km"], marker="o", label="Median", color="tab:green")
axes[1].plot(epochs_range, history["val_mean_km"], marker="o", label="Mean", color="tab:orange")
axes[1].set_title("Validation Haversine Distance (km) - Sphere Model")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"checkpoints/unit sphere regression/training_curves_sphere.png", dpi=150)
print("Saved training curves to checkpoints/unit sphere regression/training_curves_sphere.png")