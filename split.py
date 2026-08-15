# split.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("geo_dataset/train_labels.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["country"],
    random_state=66  # fixed seed so your split is reproducible
)

train_df.to_csv("geo_dataset/train_split.csv", index=False)
val_df.to_csv("geo_dataset/val_split.csv", index=False)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("\nTrain country distribution:")
print(train_df["country"].value_counts(normalize=True))
print("\nVal country distribution:")
print(val_df["country"].value_counts(normalize=True))