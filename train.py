# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from dataset import GeoDataset
from model import GeoCNN
from utils import haversine_km, count_parameters

# ---- config ----
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- transforms ----
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- data ----
train_ds = GeoDataset("geo_dataset/train_split.csv", "geo_dataset/train", transform=train_transform)
val_ds = GeoDataset("geo_dataset/val_split.csv", "geo_dataset/train", transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---- model ----
model = GeoCNN().to(DEVICE)
print("Trainable parameters:", count_parameters(model))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_haversine = float("inf")

# ---- history tracking for plots ----
history = {
    "train_loss": [],
    "val_median_km": [],
    "val_mean_km": [],
}

for epoch in range(EPOCHS):
    # ---- train ----
    model.train()
    train_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_ds)

    # ---- validate ----
    model.eval()
    all_dists = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            preds = model(images).cpu().numpy()
            targets = targets.numpy()

            pred_lat = preds[:, 0] * 90.0
            pred_lng = preds[:, 1] * 180.0
            true_lat = targets[:, 0] * 90.0
            true_lng = targets[:, 1] * 180.0

            dists = haversine_km(pred_lat, pred_lng, true_lat, true_lng)
            all_dists.extend(dists)

    median_haversine = np.median(all_dists)
    mean_haversine = np.mean(all_dists)

    # ---- record history ----
    history["train_loss"].append(train_loss)
    history["val_median_km"].append(median_haversine)
    history["val_mean_km"].append(mean_haversine)

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | "
          f"val_median_km={median_haversine:.1f} | val_mean_km={mean_haversine:.1f}")

    if median_haversine < best_val_haversine:
        best_val_haversine = median_haversine
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print(f"  -> saved new best model (median={median_haversine:.1f} km)")

print("Training complete. Best val median haversine:", best_val_haversine)

# ---- plot results ----
epochs_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# left plot: training loss
axes[0].plot(epochs_range, history["train_loss"], marker="o", color="tab:blue")
axes[0].set_title("Training Loss (MSE, normalized coords)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

# right plot: validation haversine distance
axes[1].plot(epochs_range, history["val_median_km"], marker="o", label="Median", color="tab:green")
axes[1].plot(epochs_range, history["val_mean_km"], marker="o", label="Mean", color="tab:orange")
axes[1].set_title("Validation Haversine Distance (km)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Distance (km)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("checkpoints/training_curves.png", dpi=150)
print("Saved training curves to checkpoints/training_curves.png")
plt.show()