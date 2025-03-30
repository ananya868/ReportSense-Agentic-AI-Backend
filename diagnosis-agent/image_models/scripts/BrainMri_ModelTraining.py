"""
Training script for Brain Tumor Classification using EfficientNet-B3.
This script loads the dataset, applies preprocessing, trains the model,
evaluates it on a test set, and saves the trained model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import os

"""
Configuration
"""

TRAIN_DIR = "/Users/js/Desktop/ReportSense-Agentic-AI-Backend/diagnosis-agent/data/BrainMRI_Data/Training"
TEST_DIR = "/Users/js/Desktop/ReportSense-Agentic-AI-Backend/diagnosis-agent/data/BrainMRI_Data/Testing"
BATCH_SIZE = 8
EPOCHS = 5
NUM_CLASSES = 4  # glioma, meningioma, pituitary, no_tumor

"""
Device Selection
"""

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("MPS not available, falling back to CPU")

"""
Image Preprocessing
"""

transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Resize to match EfficientNet-B3 input size
    transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

"""
Load Training & Testing Data
"""

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_ds = datasets.ImageFolder(TEST_DIR, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

"""
Display Dataset Information
"""

print(f"\nDataset Summary")
print(f"Train Samples: {len(train_ds)}")
print(f"Test Samples: {len(test_ds)}")
print(f"Classes: {train_ds.classes}")
print(f"Device: {DEVICE}")

"""
Model Definition
"""

model = EfficientNet.from_pretrained('efficientnet-b3')
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

"""
Training Loop
"""

print("\nTraining Started...\n")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
    for i, (images, labels) in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")

"""
Evaluation on Test Set
"""

print("\nEvaluating on Test Set...")
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

"""
Save Trained Model
"""

os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/brain_mri_model.pt")
print("Saved model to: weights/brain_mri_model.pt")
