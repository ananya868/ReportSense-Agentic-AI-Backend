import torch
from torchvision import models
import os

# Create the image_models folder if it doesn't exist
os.makedirs("image_models", exist_ok=True)

# Load DenseNet121 (base model)
model = models.densenet121(pretrained=False)

# Modify classifier for 14 classes (CheXNet)
model.classifier = torch.nn.Linear(model.classifier.in_features, 14)

state_dict = torch.hub.load_state_dict_from_url(
    'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    progress=True
)

for key in list(state_dict.keys()):
    if key.startswith("classifier"):
        del state_dict[key]

model.load_state_dict(state_dict, strict=False)

save_path = "chexnet_model.pth"
torch.save(model.state_dict(), save_path)

print(f"CheXNet weights saved to: {save_path}")
