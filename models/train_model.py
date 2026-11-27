import os
import pandas as pd
import joblib
from pathlib import Path



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Find project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load generated dataset
dataset_path = os.path.join(
    BASE_DIR, "data", "processed", "training_dataset.csv"
)

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train model
print("\nTraining model...")
model.fit(X_train, y_train)
# Save test dataset
# Save test dataset
TEST_FILE = Path(BASE_DIR) / "data" / "processed" / "test_dataset.csv"

TEST_FILE.parent.mkdir(parents=True, exist_ok=True)

test_output = X_test.copy()
test_output["label"] = y_test

test_output.to_csv(TEST_FILE, index=False)

print("Test dataset saved:", TEST_FILE)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Anomalous"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save trained model and feature column order
model_folder = os.path.join(BASE_DIR, "models")
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, "anomaly_model.pkl")

joblib.dump({
    "model": model,
    "feature_columns": list(X.columns)
}, model_path)

print("\nModel saved successfully!")
print("Location:", model_path)
