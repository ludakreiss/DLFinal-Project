# compute_centroids.py
import pandas as pd

df = pd.read_csv("geo_dataset/train_labels.csv")
centroids = df.groupby("country")[["lat", "lng"]].mean()
centroids.to_csv("geo_dataset/country_centroids.csv")
print(centroids)