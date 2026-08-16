# compute_centroids.py
import pandas as pd

df = pd.read_csv("geo_dataset/train_labels.csv")
centroids = df.groupby("country")[["lat", "lng"]].mean()
centroids.to_csv("geo_dataset/country_centroids.csv")
print(centroids)

# compute offset spread for normalization
merged = df.merge(centroids, on="country", suffixes=("", "_centroid"))
delta_lat = merged["lat"] - merged["lat_centroid"]
delta_lng = merged["lng"] - merged["lng_centroid"]

offset_std_lat = delta_lat.std()
offset_std_lng = delta_lng.std()

print(f"Offset std: lat={offset_std_lat:.4f}, lng={offset_std_lng:.4f}")

with open("geo_dataset/offset_stats.txt", "w") as f:
    f.write(f"{offset_std_lat},{offset_std_lng}")