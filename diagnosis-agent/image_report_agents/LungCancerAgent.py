"""
LungCancerAgent: Agent for Lung CT Scan Analysis

This agent receives a lung CT scan image file path via a message,
uses a trained ResNet18 model to predict the cancer type,
and returns the predicted label to a connected agent.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from uagents import Agent, Context

from agent_models.lung_models import LungRequest, LungResponse

"""
Agent Configuration
"""

# Create the LungCancerAgent and assign it a port and endpoint
lung_agent = Agent(name="LungCancerAgent", port=8005, endpoint="http://localhost:8005/submit")

# Set the address of the agent that will receive the prediction response
REPORT_HANDLER_AGENT_ADDRESS = "agent1qfteffcpfqhrsj9mpcjxvza42axkr5y9zva0fnmgztzmpaaxse00garhcdv"

"""
Model Configuration
"""

# Path to the trained ResNet18 model weights
MODEL_PATH = "/Users/js/Desktop/LungCancerDetection/lung_cancer_model.pth"

# Class labels in the same order as used during training
CLASS_NAMES = ['adenocarcinoma', 'large cell carcinoma', 'normal', 'squamous cell carcinoma']

# Select appropriate device (Apple MPS if available, else CPU)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the ResNet18 model and modify the classifier layer
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

"""
Image Preprocessing
"""

# Transform pipeline to resize, normalize, and convert the image to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

"""
Lung CT Scan Handler

Triggered when a LungRequest is received.
Processes the image using the trained model and sends the prediction.
"""

@lung_agent.on_message(model=LungRequest)
async def handle_lung_ct(ctx: Context, sender: str, message: LungRequest):
    """
    Handles incoming CT scan classification requests.

    Args:
        ctx (Context): UAgents context for sending messages
        sender (str): Address of the sending agent
        message (LungRequest): Message containing image file path
    """
    file_path = message.file_path
    ctx.logger.info(f"Received lung CT scan from {sender}: {file_path}")

    prediction = classify_lung_ct(file_path)

    ctx.logger.info(f"Prediction result: {prediction}")
    response = LungResponse(cancer_prediction=prediction)
    await ctx.send(REPORT_HANDLER_AGENT_ADDRESS, response)

"""
Prediction Function

Takes an image file path, processes it, and predicts the class using the model.
"""

def classify_lung_ct(file_path: str):
    """
    Classifies a lung CT image using the trained ResNet18 model.

    Args:
        file_path (str): Path to the CT scan image

    Returns:
        str: Predicted cancer type or error message
    """
    try:
        image = Image.open(file_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
            return CLASS_NAMES[pred.item()]
    except Exception as e:
        return f"Error: {str(e)}"

"""
Main Execution

Prints the agent address and starts the UAgents runtime.
"""

if __name__ == "__main__":
    print(f"LungCancerAgent Address: {lung_agent.address}")
    lung_agent.run()
