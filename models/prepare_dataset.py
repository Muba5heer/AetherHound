
import pandas as pd
from pathlib import Path

# Input and output paths
input_path = Path("data/processed/training_dataset.csv")
output_path = Path("data/processed/cleaned_dataset.csv")

print("AETHERHOUND DATASET PREPARATION")
print("=" * 45)

# Check whether training dataset exists
if not input_path.exists():
    raise FileNotFoundError(f"Dataset not found: {input_path}")

# Load dataset
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows containing missing values
df = df.dropna()

# Separate features and label
if "label" not in df.columns:
    raise ValueError("The dataset does not contain a 'label' column.")

# Save cleaned dataset
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\nDataset cleaned successfully!")
print("Cleaned dataset shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts().sort_index())

print("\nSaved at:", output_path)
