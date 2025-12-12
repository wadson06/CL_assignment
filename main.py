import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 1. DATA COLLECTION
print("\n" + "="*60)
print("1. DATA COLLECTION")
print("="*60)

df = pd.read_csv("breast-cancer-wisconsin-data.csv")
print(f"\n✓ Dataset loaded successfully")
print(f"  Total records: {len(df)}")
print(f"  Total features: {len(df.columns)}")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Column Information ---")
print(df.info())


# 2. DATA EXPLORATION & ANALYSIS
print("\n" + "="*60)
print("2. DATA EXPLORATION & ANALYSIS")
print("="*60)

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
print("✓ Converted diagnosis (M=1, B=0)")
print(f"Unique values: {df['diagnosis'].unique()}")

print("\n--- Descriptive Statistics ---")
print(df.describe())

print(f"\nDataset Shape: {df.shape}")

print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing)
print(f"Total missing values: {missing.sum()}")

print("\n--- Duplicate Rows ---")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Diagnosis distribution BEFORE cleaning/encoding
print("\n--- Diagnosis Distribution (Raw) ---")
print(df["diagnosis"].value_counts())
print(df["diagnosis"].value_counts(normalize=True))

# EDA VISUALIZATIONS
# Class distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x=df["diagnosis"])
plt.title("Breast Cancer Diagnosis Distribution")
plt.xlabel("Diagnosis (0 = Benign, 1 = Malignant)")
plt.ylabel("Number of Samples")
plt.show()

# Correlation heatmap
plt.figure(figsize=(14, 12))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Print top correlated features with diagnosis
print("\n--- Top Correlations with Diagnosis ---")
print(corr["diagnosis"].sort_values(ascending=False).head(15))

# Histograms
df.hist(figsize=(14,12))
plt.tight_layout()
plt.show()


# 3. DATA CLEANING
print("\n" + "="*60)
print("3. DATA CLEANING")
print("="*60)

# Remove id column
if "id" in df.columns:
    df = df.drop(columns=["id"])
    print("✓ Removed 'id' column")


# Missing values (again)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Remove duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✓ Removed {duplicates} duplicate rows")
else:
    print("✓ No duplicate rows found")

print("\n--- Final Dataset After Cleaning ---")
print(df.info())
print(f"Shape: {df.shape}")


# DATA SPLITTING (70 / 15 / 15)
print("\n" + "="*60)
print("4. DATA SPLITTING")
print("="*60)

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Train: 70% | Temp: 30%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Validation: 15% | Test: 15%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("\n--- Dataset Splits ---")
print(f"Training size:   {X_train.shape}")
print(f"Validation size: {X_val.shape}")
print(f"Testing size:    {X_test.shape}")



# FEATURE SCALING
print("\n" + "="*60)
print("5. FEATURE SCALING")
print("="*60)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print("✓ Features scaled successfully")
print(f"Train mean: {X_train.mean():.4f}, Train std: {X_train.std():.4f}")

print("\n" + "="*60)
print("✓ DATA PIPELINE COMPLETE — READY FOR MODEL TRAINING")
print("="*60)

print("\nFinal shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_val:   {y_val.shape}")
print(f"  y_test:  {y_test.shape}")


# CONVERT DATA TO TORCH TENSORS
print("\n" + "="*60)
print("6. CONVERT DATA TO TORCH TENSORS")
print("="*60)

# Convert numpy arrays → torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.long)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.long)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)

print("✓ Tensor conversion complete")
print(f"Train tensor shape: {X_train_t.shape}")
print(f"Val tensor shape:   {X_val_t.shape}")
print(f"Test tensor shape:  {X_test_t.shape}")

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

print("✓ DataLoaders created successfully")

# ANN MODEL 
print("\n" + "="*60)
print("7. BUILD ANN MODEL")
print("="*60)

class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(30, 16),  
            nn.ReLU(),
            nn.Linear(16, 8),   
            nn.ReLU(),
            nn.Linear(8, 1)     
        )

    def forward(self, x):
        return self.network(x)

model = ANNModel()
print(model)


# TRAINING LOOP (with validation)

print("\n" + "="*60)
print("8. TRAIN ANN MODEL")
print("="*60)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def calculate_accuracy(logits, labels):
    preds = torch.sigmoid(logits)
    preds = (preds > 0.5).long().squeeze()
    return (preds == labels).float().mean().item()

for epoch in range(num_epochs):
    # Training 
    model.train()
    total_train_loss = 0
    total_train_acc = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch).squeeze()  # logits
        loss = criterion(outputs, y_batch.float())

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_acc += calculate_accuracy(outputs, y_batch)

    train_losses.append(total_train_loss / len(train_loader))
    train_accuracies.append(total_train_acc / len(train_loader))

    # Validation 
    model.eval()
    total_val_loss = 0
    total_val_acc = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.float())

            total_val_loss += loss.item()
            total_val_acc += calculate_accuracy(outputs, y_batch)

    val_losses.append(total_val_loss / len(val_loader))
    val_accuracies.append(total_val_acc / len(val_loader))

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_losses[-1]:.4f}, "
          f"Val Loss: {val_losses[-1]:.4f}, "
          f"Train Acc: {train_accuracies[-1]:.4f}, "
          f"Val Acc: {val_accuracies[-1]:.4f}")


# PLOT TRAINING CURVES
# Loss Curve 
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Accuracy Curve 
plt.figure(figsize=(8,5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# TEST SET EVALUATION + CONFUSION MATRIX
print("\n" + "="*60)
print("10. TEST SET EVALUATION")
print("="*60)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).long()
        
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.tolist())

test_acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {test_acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0 (Benign)", "Pred 1 (Malignant)"],
            yticklabels=["Actual 0 (Benign)", "Actual 1 (Malignant)"])
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

