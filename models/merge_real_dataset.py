import pandas as pd
from pathlib import Path

# File paths
SIMULATED_FILE = Path("data/processed/training_dataset.csv")
REAL_FILE = Path("data/raw/real_normal/real_normal_features.csv")

TRAIN_OUTPUT = Path("data/processed/training_dataset_mixed.csv")
HOLDOUT_OUTPUT = Path("data/processed/real_normal_holdout.csv")

print("\nAETHERHOUND DATASET MERGING")
print("=" * 40)

# Check files
if not SIMULATED_FILE.exists():
    raise FileNotFoundError(f"Missing file: {SIMULATED_FILE}")

if not REAL_FILE.exists():
    raise FileNotFoundError(f"Missing file: {REAL_FILE}")

# Load datasets
simulated_df = pd.read_csv(SIMULATED_FILE)
real_df = pd.read_csv(REAL_FILE)

# Separate real recordings for independent testing
real_train = real_df.iloc[:8].copy()
real_holdout = real_df.iloc[8:].copy()

# Match feature columns
feature_columns = [
    col for col in simulated_df.columns if col != "label"
]

missing = set(feature_columns) - set(real_train.columns)

if missing:
    raise ValueError(f"Missing real audio features: {missing}")

# Keep only matching columns
real_train = real_train[feature_columns + ["label"]]
real_holdout = real_holdout[feature_columns + ["label"]]

# Combine simulated data and real normal training data
merged_df = pd.concat(
    [simulated_df, real_train],
    ignore_index=True
)

# Clean and save
merged_df = merged_df.dropna()
merged_df.to_csv(TRAIN_OUTPUT, index=False)

# Save independent real recordings separately
real_holdout.to_csv(HOLDOUT_OUTPUT, index=False)

print("\nDataset merging completed!")
print("Simulated dataset:", len(simulated_df))
print("Real normal training samples:", len(real_train))
print("Independent real normal test samples:", len(real_holdout))
print("Combined dataset shape:", merged_df.shape)
print("Saved training dataset:", TRAIN_OUTPUT)
print("Saved holdout dataset:", HOLDOUT_OUTPUT)
print("\nLabel distribution:")
print(merged_df["label"].value_counts())