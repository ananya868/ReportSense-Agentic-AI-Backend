"""
Lung Cancer Classification Training Script

This script loads the 4-class lung CT scan dataset, applies aggressive image
augmentation, trains a ResNet18 model on the data, evaluates on validation,
and saves the best performing model as 'lung_cancer_model.pth'.
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import copy

"""
Configuration
"""

# Path to your dataset containing 'train/' and 'valid/' folders
DATA_DIR = os.path.expanduser("/Users/js/Desktop/ReportSense-Agentic-AI-Backend/diagnosis-agent/data/LungCancer_Data")

# Output model path
MODEL_PATH = "/Users/js/Desktop/ReportSense-Agentic-AI-Backend/diagnosis-agent/image_models/weights/lung_cancer_model.pth"

# Training hyperparameters
BATCH_SIZE = 16
IMG_SIZE = 224
NUM_EPOCHS = 10
LR = 1e-4

"""
Device Selection
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

"""
Image Preprocessing & Augmentation
"""

# Aggressive augmentation pipeline for training data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),   # Random crop + resize
    transforms.RandomHorizontalFlip(p=0.5),                     # Flip horizontally
    transforms.RandomRotation(degrees=20),                      # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),       # Brightness/contrast
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),   # Slight shifts
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)), # Soft blur
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],                 # ImageNet mean
                         [0.229, 0.224, 0.225])                 # ImageNet std
])

# Simpler transform for validation data (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

"""
Load Training & Validation Data
"""

train_path = os.path.join(DATA_DIR, "train")
valid_path = os.path.join(DATA_DIR, "valid")

train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(valid_path, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print(f"\nClass Names: {class_names}")

"""
Model Definition: ResNet18
"""

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Replace the final fully connected layer with one for our 4 classes
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(DEVICE)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# To store best model weights
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

"""
Training Loop
"""

print("\nTraining Started...\n")
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    """
    Validation Loop
    """

    model.eval()
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)

    val_acc = val_corrects.double() / len(val_dataset)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

"""
Save Best Model
"""

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nTraining complete. Best Validation Accuracy: {best_acc:.4f}")
print(f"Model saved to: {MODEL_PATH}")
