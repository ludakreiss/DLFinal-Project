# predict.py
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from model import GeoCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOLDOUT_DIR = "geo_dataset/holdout_public"
CHECKPOINT_PATH = "checkpoints/best_model.pt"
OUTPUT_PATH = "predictions.csv"

# ---- transform (must match validation transform used in training) ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- load model ----
model = GeoCNN().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ---- get all holdout filenames ----
filenames = sorted(os.listdir(HOLDOUT_DIR))
print(f"Found {len(filenames)} holdout images")

results = []

with torch.no_grad():
    for fname in filenames:
        img_path = os.path.join(HOLDOUT_DIR, fname)
        image = Image.open(img_path)
        image = transform(image).unsqueeze(0).to(DEVICE)  # add batch dim

        pred = model(image).cpu().numpy()[0]  # [lat_norm, lng_norm]

        pred_lat = pred[0] * 90.0
        pred_lng = pred[1] * 180.0

        results.append({
            "filename": fname,
            "pred_lat": pred_lat,
            "pred_lng": pred_lng
        })

# ---- write predictions.csv ----
df = pd.DataFrame(results)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} predictions to {OUTPUT_PATH}")
print(df.head())